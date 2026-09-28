"""Post-processing "white magic" on the deployed fusion embeddings (CPU-only).

Takes the two component feature banks of the champion pipeline on the frozen
hold-out val split (seed 42):
    sig  = SigLIP2 NaFlex 512d   (frozen, no TTA)
    dino = DINOv2-B 512d         (exp-0007, TTA baked in: 224+280)
and evaluates variants of the *post-processing* that turn them into a ranking:

    fuse(w) -> [per-component power-norm] -> L2 -> optional PCA-whitening
            -> optional power-norm -> L2 -> cosine / k-reciprocal

PCA-whitening is always FIT ON THE GALLERY ONLY (never on the query set), so
the ranking of a query still depends on that query + gallery alone — the
streaming invariant holds. Metrics come from the OFFICIAL ``evaluate.py`` via
``reid.eval.harness``.

    python -m reid.models.fusion_postproc --dataset "docs/<ds>/dataset" ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pnorm(x, p):
    if p == 1.0:
        return x
    return np.sign(x) * np.power(np.abs(x), p)


def fit_whiten(X, k=None, eps=1e-4):
    """PCA-whitening fit on X. Returns (mu, W). Reduces to k dims if given."""
    mu = X.mean(axis=0)
    Xc = X - mu
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(1, n - 1)
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], eps, None)
    evecs = evecs[:, order]
    if k is not None and k > 0:
        evals = evals[:k]
        evecs = evecs[:, :k]
    W = (evecs / np.sqrt(evals)[None, :]).astype(np.float32)
    return mu.astype(np.float32), W


def flat(rep):
    r = rep.get("ranking", {})
    fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"), "mAP_full": fr.get("mAP_full"),
            "mINP": fr.get("mINP"), "n_scored": r.get("n_scored")}


def write_sub(path, q_ids, g_ids, orders, scores, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])
    with open(os.path.join(os.path.dirname(path), "candidates.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def build_post(sig, dino, nq, cfg):
    """Return fused (q, g) after all post-processing. Whitening fit on gallery."""
    ps = pnorm(l2(sig), cfg.get("pow_sig", 1.0))
    pd = pnorm(l2(dino), cfg.get("pow_dino", 1.0))
    s = l2(ps)
    d = l2(pd)
    fused = l2(np.concatenate([s, cfg["w"] * d], axis=1))
    q, g = fused[:nq], fused[nq:]
    if cfg.get("whiten"):
        mu, W = fit_whiten(g, k=cfg.get("wk", None), eps=cfg.get("weps", 1e-4))
        q = (q - mu) @ W
        g = (g - mu) @ W
    p = cfg.get("pow_after", 1.0)
    if p != 1.0:
        q, g = pnorm(q, p), pnorm(g, p)
    return np.ascontiguousarray(l2(q), dtype=np.float32), \
        np.ascontiguousarray(l2(g), dtype=np.float32)


def evaluate_cfg(cfg, sig, dino, nq, ids, out, evaldir, rr_cfg, tag):
    q_ids, g_ids, gt, qcsv, gcsv = ids
    q, g = build_post(sig, dino, nq, cfg)
    sub = os.path.join(evaldir, "submission.csv")
    emb = os.path.join(evaldir, "embeddings.npy")
    np.save(emb, np.vstack([q, g]).astype(np.float32))

    # baseline cosine
    sims = q @ g.T
    o0 = np.argsort(-sims, axis=1, kind="stable")
    write_sub(sub, q_ids, g_ids, o0, sims)
    m_base = flat(run_official(gt_csv=gt, submission=sub,
                               candidates=os.path.join(evaldir, "candidates.csv"),
                               embeddings=emb, query=qcsv, gallery=gcsv,
                               json_out=os.path.join(evaldir, "official_base.json")))

    # per-query k-reciprocal
    preps = [rerank.prepare_query(q[i], g, pool_size=rr_cfg["pool_size"])
             for i in range(nq)]
    o = np.empty((nq, len(g)), np.int64)
    sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=rr_cfg["k1"],
                                           k2=rr_cfg["k2"], lam=rr_cfg["lam"])
    write_sub(sub, q_ids, g_ids, o, sc)
    m_rr = flat(run_official(gt_csv=gt, submission=sub,
                             candidates=os.path.join(evaldir, "candidates.csv"),
                             embeddings=emb, query=qcsv, gallery=gcsv,
                             json_out=os.path.join(evaldir, "official_rr.json")))
    row = {"tag": tag, "cfg": cfg, "base_mAP@10": m_base["mAP@10"],
           "mAP@10": m_rr["mAP@10"], "Rank-1": m_rr["Rank-1"],
           "Rank-5": m_rr["Rank-5"], "mINP": m_rr["mINP"],
           "base_mINP": m_base["mINP"]}
    print(f"{tag:52s} base={m_base['mAP@10']:.4f} rr={m_rr['mAP@10']:.4f} "
          f"R1={m_rr['Rank-1']:.4f} mINP={m_rr['mINP']:.4f}", flush=True)
    return row


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k1", type=int, default=6)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--ref", default="runs/W2-5-val-fusion/embeddings.npy",
                    help="reference deployed fusion embeddings (repro check)")
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required (auto-detect failed)", file=sys.stderr)
            return 2
        args.dataset = os.path.dirname(cand[0])
    print(f"[dataset] {args.dataset}", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dino = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == dino.shape[0] == nq + len(vg), (sig.shape, dino.shape)

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    rr_cfg = {"k1": args.k1, "k2": args.k2, "lam": args.lam,
              "pool_size": args.pool}

    # --- repro check: our reconstruction of the deployed fusion ---
    if args.ref and os.path.exists(args.ref):
        ref = np.load(args.ref).astype(np.float32)
        q_, g_ = build_post(sig, dino, nq, {"w": 0.6})
        ours = np.vstack([q_, g_])
        cos = (ours * ref).sum(1) / (np.linalg.norm(ours, axis=1) *
                                     np.linalg.norm(ref, axis=1) + 1e-12)
        print(f"repro vs {args.ref}: shape={ref.shape} cos_min={cos.min():.6f} "
              f"cos_mean={cos.mean():.6f}", flush=True)

    evaldir = os.path.join(out, "eval")
    os.makedirs(evaldir, exist_ok=True)

    rows = []
    # 1) weight sweep (finer + higher, since the first pass peaked at w=0.8)
    for w in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.2, 1.5):
        rows.append(evaluate_cfg({"w": w}, sig, dino, nq, ids, out, evaldir,
                                 rr_cfg, f"w={w}"))
    best_w = max(rows, key=lambda r: r["mAP@10"] or 0)["cfg"]["w"]
    print(f"[stage2] rerank grid at best w={best_w}", flush=True)

    # 2) k-reciprocal grid at the best fusion weight
    import itertools
    grid = []
    for pool in (100, 200):
        for k1 in (4, 6, 8, 10):
            for k2 in (2, 3, 4):
                for lam in (0.5, 0.6, 0.7):
                    if k2 >= k1:
                        continue
                    rc = {"k1": k1, "k2": k2, "lam": lam, "pool_size": pool}
                    row = evaluate_cfg({"w": best_w}, sig, dino, nq, ids, out,
                                       evaldir, rc, f"w={best_w} {rc}")
                    row["rr_cfg"] = rc
                    grid.append(row)
    rows.extend(grid)
    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "fusion postproc", "rr": rr_cfg,
              "results": rows, "best": best}
    with open(os.path.join(out, "postproc_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
