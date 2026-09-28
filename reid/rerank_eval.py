"""Val evaluation of k-reciprocal re-ranking + query-side TTA (W2-1 / W2-2).

Runs on the frozen champion checkpoint (``runs/exp-0007/best.pt``, DINOv2-B,
EMA) over the hold-out val split (``reid.data.splits.holdout_val`` seed=42).

    python -m reid.rerank_eval --checkpoint runs/exp-0007/best.pt \
        --out runs/exp-0011_rerank --json reports/exp-0011.json

What it computes, all through the OFFICIAL ``evaluate.py`` (reid.eval.harness):
    baseline      raw 224 query/gallery, cosine ranking
    rerank        per-query k-reciprocal on the raw 224 embeddings
    tta_only      query-side multiscale fusion (224 + 280), no re-rank
    rerank_tta    multiscale-fused query + k-reciprocal

``embeddings.npy`` is kept RAW for the baseline/rerank runs (re-rank only
changes the submission order). For the TTA runs the query side of
``embeddings.npy`` holds the fused query (TTA is inference, not re-rank).

No hflip anywhere. Each query is processed independently; this script never
builds a query-gallery graph that spans more than one query.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from reid import rerank
from reid.data.aug import build_eval_transform
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models import build_model
from reid.train import extract_embeddings, resolve_crop_cache, set_seed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRID = {
    "k1": [5, 10, 20, 40],
    "k2": [2, 3],
    # lam=1.0 is the identity anchor (reproduces the baseline exactly)
    "lam": [0.0, 0.3, 0.5, 0.7, 1.0],
    "pool_size": 100,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _write_submission(path, q_ids, g_ids, orders, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            row = [qid] + [g_ids[j] for j in orders[i, :top_k]]
            w.writerow(row)


def _write_candidates(path, q_ids, g_ids, orders, scores):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def _write_gt(val_query, val_gallery, path):
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy()
    q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy()
    g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def _metrics(rep):
    r = rep.get("ranking", {})
    fr = rep.get("full_ranking", {})
    return {
        "mAP@10": r.get("mAP@10"),
        "Rank-1": r.get("Rank-1"),
        "Rank-5": r.get("Rank-5"),
        "mAP_full": fr.get("mAP_full"),
        "mINP": fr.get("mINP"),
        "n_scored": r.get("n_scored"),
        "n_openset_excluded": r.get("n_openset_excluded"),
    }


def _emit_run(run_dir, gt, query_csv, gallery_csv, submission, candidates,
              embeddings, tag):
    """Copy/inspect one variant with the official evaluate.py; return metrics."""
    rep = run_official(
        gt_csv=gt, submission=submission, candidates=candidates,
        embeddings=embeddings, query=query_csv, gallery=gallery_csv,
        json_out=os.path.join(run_dir, f"official_{tag}.json"),
    )
    return _metrics(rep)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------
def _load_model(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = ck.get("config", {})
    sd = ck["state_dict"]
    num_classes = int(sd["arcface.weight"].shape[0])
    model = build_model(
        backbone=ck.get("backbone", cfg.get("backbone", "dinov2_b")),
        num_classes=num_classes,
        emb_dim=int(ck.get("emb_dim", cfg.get("emb_dim", 512))),
        pretrained=False,
        margin=float(cfg.get("margin", 0.3)),
        scale=float(cfg.get("scale", 30.0)),
        gem_p=float(cfg.get("gem_p", 3.0)),
        image_size=int(cfg.get("image_size", 224)),
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, cfg


def _extract(model, df, dataset_dir, size, device, cache_dir, cfg):
    tf = build_eval_transform(size)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("amp"))
    return extract_embeddings(
        model, df, dataset_dir, tf, size, device,
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 8)),
        amp_dtype=amp, cache_dir=cache_dir,
        draft_factor=float(cfg.get("draft_factor", 2.0)),
    )


def _grid_combos(grid):
    for k1 in grid["k1"]:
        for k2 in grid["k2"]:
            for lam in grid["lam"]:
                yield int(k1), int(k2), float(lam)


def _rank_all(queries, gallery, k1, k2, lam, pool_size):
    """Per-query re-ranking (prepare once per query, score one combo)."""
    n_q = queries.shape[0]
    n_g = gallery.shape[0]
    orders = np.empty((n_q, n_g), dtype=np.int64)
    scores = np.empty((n_q, n_g), dtype=np.float32)
    for i in range(n_q):
        prep = rerank.prepare_query(queries[i], gallery, pool_size=pool_size)
        orders[i], scores[i] = rerank.rank_prepared(prep, k1=k1, k2=k2, lam=lam)
    return orders, scores


def _verify_streaming(queries, gallery, k1, k2, lam, pool_size, idx=(0, 5, 17)):
    """Re-check the streaming invariant at runtime on the real embeddings."""
    ok = True
    for t in idx:
        t = int(t) % len(queries)
        single = rerank.k_reciprocal_order(queries[t], gallery, k1=k1, k2=k2,
                                           lam=lam, pool_size=pool_size)[0]
        perm = np.random.default_rng(123).permutation(len(queries))
        new_t = int(np.where(perm == t)[0][0])
        shuffled = rerank.k_reciprocal_order(queries[perm][new_t], gallery,
                                             k1=k1, k2=k2, lam=lam,
                                             pool_size=pool_size)[0]
        only = rerank.k_reciprocal_order(queries[[t]][0], gallery, k1=k1, k2=k2,
                                         lam=lam, pool_size=pool_size)[0]
        ok = ok and np.array_equal(single, shuffled) and np.array_equal(single, only)
    return bool(ok)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="k-reciprocal + TTA val eval (W2-1/2)")
    ap.add_argument("--checkpoint", default="runs/exp-0007/best.pt")
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--out", default="runs/exp-rerank")
    ap.add_argument("--scales", default="224,280",
                    help="base,tta scale (must be /14 for DINOv2)")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_GRID["pool_size"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args(argv)

    if not args.dataset:
        print("ERROR: --dataset or DATASET_DIR required", file=sys.stderr)
        return 2

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.abspath(args.out)
    run_dir = os.path.join(out_dir, "val")
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    model, cfg = _load_model(args.checkpoint, device)
    print(f"checkpoint {args.checkpoint} loaded on {device}", flush=True)

    train_full = read_csv(os.path.join(args.dataset, "train.csv"),
                          required=TRAIN_COLUMNS)
    _, val_query, val_gallery = holdout_val(
        train_full,
        val_fraction=float(cfg.get("val_fraction", 0.2)),
        open_set_fraction=float(cfg.get("open_set_fraction", 0.2)),
        seed=42,
    )
    print(f"val: query={len(val_query)} gallery={len(val_gallery)}", flush=True)

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    base = scales[0]
    cache_dir = resolve_crop_cache(cfg)

    # ---- embeddings (base + TTA views of the query) ----------------------
    t0 = time.time()
    q_base = _extract(model, val_query, args.dataset, base, device, cache_dir, cfg)
    g_base = _extract(model, val_gallery, args.dataset, base, device, cache_dir, cfg)
    extra_q = {}
    for s in scales[1:]:
        extra_q[s] = _extract(model, val_query, args.dataset, s, device, cache_dir, cfg)
    print(f"embeddings extracted in {time.time() - t0:.1f}s", flush=True)

    if len(scales) > 1:
        q_tta = np.stack([
            rerank.fuse_embeddings(
                [q_base[i]] + [extra_q[s][i] for s in scales[1:]])
            for i in range(len(q_base))
        ]).astype(np.float32)
    else:
        q_tta = q_base

    q_ids = val_query["image_id"].tolist()
    g_ids = val_gallery["image_id"].tolist()
    gt = os.path.join(run_dir, "gt.csv")
    _write_gt(val_query, val_gallery, gt)
    val_query.to_csv(os.path.join(run_dir, "query.csv"), index=False)
    val_gallery.to_csv(os.path.join(run_dir, "gallery.csv"), index=False)
    np.save(os.path.join(run_dir, "raw_embeddings.npy"),
            np.vstack([q_base, g_base]).astype(np.float32))

    emb_raw = os.path.join(run_dir, "embeddings.npy")
    np.save(emb_raw, np.vstack([q_base, g_base]).astype(np.float32))

    sub = os.path.join(run_dir, "submission.csv")
    cand = os.path.join(run_dir, "candidates.csv")
    query_csv = os.path.join(run_dir, "query.csv")
    gallery_csv = os.path.join(run_dir, "gallery.csv")

    # ---- baseline: raw cosine -------------------------------------------
    sims0 = q_base @ g_base.T
    ord0 = np.argsort(-sims0, axis=1, kind="stable")
    _write_submission(sub, q_ids, g_ids, ord0)
    _write_candidates(cand, q_ids, g_ids, ord0, sims0)
    baseline = _emit_run(run_dir, gt, query_csv, gallery_csv, sub, cand,
                         emb_raw, "baseline")
    print("baseline", json.dumps(baseline, ensure_ascii=False), flush=True)

    # ---- TTA-only (query-side multiscale, no re-rank) --------------------
    tta_only = {}
    if len(scales) > 1:
        sims_t = q_tta @ g_base.T
        ord_t = np.argsort(-sims_t, axis=1, kind="stable")
        _write_submission(sub, q_ids, g_ids, ord_t)
        _write_candidates(cand, q_ids, g_ids, ord_t, sims_t)
        np.save(emb_raw, np.vstack([q_tta, g_base]).astype(np.float32))
        tta_only = _emit_run(run_dir, gt, query_csv, gallery_csv, sub, cand,
                             emb_raw, "tta_only")
        print("tta_only", json.dumps(tta_only, ensure_ascii=False), flush=True)
        np.save(emb_raw, np.vstack([q_base, g_base]).astype(np.float32))

    # ---- grid over k1, k2, lam ------------------------------------------
    grid_table = []
    best = None
    if not args.skip_grid:
        for k1, k2, lam in _grid_combos(DEFAULT_GRID):
            t0 = time.time()
            orders, scores = _rank_all(q_base, g_base, k1, k2, lam,
                                       args.pool_size)
            _write_submission(sub, q_ids, g_ids, orders)
            _write_candidates(cand, q_ids, g_ids, orders, scores)
            m = _emit_run(run_dir, gt, query_csv, gallery_csv, sub, cand,
                          emb_raw, f"grid_k{k1}_k2{k2}_l{lam}")
            row = {"k1": k1, "k2": k2, "lam": lam,
                   "mAP@10": m["mAP@10"], "Rank-1": m["Rank-1"],
                   "Rank-5": m["Rank-5"]}
            grid_table.append(row)
            print(f"  k1={k1:>3} k2={k2} lam={lam:.1f}  "
                  f"mAP@10={m['mAP@10']:.4f} R1={m['Rank-1']:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            if best is None or (m["mAP@10"] or 0) > (best["mAP@10"] or 0):
                best = {**row, "metrics": m}

    if best is None:
        best = {"k1": DEFAULT_GRID["k1"][1], "k2": DEFAULT_GRID["k2"][0],
                "lam": DEFAULT_GRID["lam"][1], "mAP@10": None,
                "Rank-1": None, "Rank-5": None, "metrics": {}}

    bk1, bk2, blam = best["k1"], best["k2"], best["lam"]

    # ---- best re-rank (raw query) ---------------------------------------
    orders_r, scores_r = _rank_all(q_base, g_base, bk1, bk2, blam,
                                   args.pool_size)
    _write_submission(sub, q_ids, g_ids, orders_r)
    _write_candidates(cand, q_ids, g_ids, orders_r, scores_r)
    np.save(emb_raw, np.vstack([q_base, g_base]).astype(np.float32))
    rerank_metrics = _emit_run(run_dir, gt, query_csv, gallery_csv, sub, cand,
                               emb_raw, "rerank_best")
    print("rerank", json.dumps(rerank_metrics, ensure_ascii=False), flush=True)

    # ---- best re-rank + TTA ---------------------------------------------
    rerank_tta = {}
    if len(scales) > 1:
        orders_rt, scores_rt = _rank_all(q_tta, g_base, bk1, bk2, blam,
                                         args.pool_size)
        _write_submission(sub, q_ids, g_ids, orders_rt)
        _write_candidates(cand, q_ids, g_ids, orders_rt, scores_rt)
        np.save(emb_raw, np.vstack([q_tta, g_base]).astype(np.float32))
        rerank_tta = _emit_run(run_dir, gt, query_csv, gallery_csv, sub, cand,
                               emb_raw, "rerank_tta")
        print("rerank_tta", json.dumps(rerank_tta, ensure_ascii=False),
              flush=True)
        np.save(emb_raw, np.vstack([q_base, g_base]).astype(np.float32))

    # ---- streaming invariant on real embeddings -------------------------
    streaming_ok = _verify_streaming(q_base, g_base, bk1, bk2, blam,
                                     args.pool_size)
    print(f"streaming invariant: {'OK' if streaming_ok else 'FAIL'}", flush=True)

    def _delta(a, b):
        return None if (a is None or b is None) else float(a - b)

    report = {
        "agent": "rerank-agent",
        "task": "W2-1/W2-2",
        "checkpoint": args.checkpoint,
        "split": {"seed": 42, "n_query": int(len(val_query)),
                  "n_gallery": int(len(val_gallery))},
        "scales": scales,
        "grid": {"k1": DEFAULT_GRID["k1"], "k2": DEFAULT_GRID["k2"],
                 "lam": DEFAULT_GRID["lam"], "pool_size": args.pool_size},
        "best": {"k1": bk1, "k2": bk2, "lam": blam},
        "metrics": {
            "baseline": baseline,
            "rerank": rerank_metrics,
            "tta_only": tta_only,
            "rerank_tta": rerank_tta,
        },
        "deltas": {
            "rerank_mAP@10": _delta(rerank_metrics.get("mAP@10"),
                                    baseline.get("mAP@10")),
            "rerank_Rank-1": _delta(rerank_metrics.get("Rank-1"),
                                    baseline.get("Rank-1")),
            "rerank_mINP": _delta(rerank_metrics.get("mINP"),
                                  baseline.get("mINP")),
            "rerank_tta_mAP@10": _delta(rerank_tta.get("mAP@10"),
                                        baseline.get("mAP@10")),
            "rerank_tta_Rank-1": _delta(rerank_tta.get("Rank-1"),
                                        baseline.get("Rank-1")),
            "rerank_tta_mINP": _delta(rerank_tta.get("mINP"),
                                      baseline.get("mINP")),
            "tta_only_mAP@10": _delta(tta_only.get("mAP@10"),
                                      baseline.get("mAP@10")),
        },
        "grid_table": grid_table,
        "streaming_ok": streaming_ok,
        "timing_s": round(time.time() - t_start, 1),
        "notes": ("embeddings.npy raw for baseline/rerank; fused query for "
                  "TTA runs (TTA is inference, not re-rank). No hflip; each "
                  "query independent. Re-rank affects submission only."),
    }

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"report -> {args.json}", flush=True)
    print(f"done in {report['timing_s']}s", flush=True)
    return 0 if streaming_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
