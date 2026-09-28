"""Database-side augmentation (DBA) on the champion fusion gallery (CPU-only).

Hypothesis: augmenting every *gallery* feature with a weighted mean of its
top-k gallery neighbours (alpha-QE / DBA, e.g. Arandjelovic et al.) sharpens
the database manifold and improves retrieval, especially when the query is
"cleaner" than a gallery item (occlusion / odd viewpoint).

Streaming invariant (INTERFACES.md §5): ONLY gallery rows are touched; the
query feature is never expanded against the gallery (that would be query
expansion, which is forbidden). A gallery feature's neighbours are computed
inside the gallery alone, so rank(q) depends only on q + the gallery.

Pipeline per config:
    fusion(w) -> [per-component power-norm] -> L2 -> gallery DBA
              -> L2 -> cosine / k-reciprocal

Metrics are produced by the OFFICIAL ``evaluate.py`` via ``reid.eval.harness``.

    python -m reid.models.dba_postproc --out runs/exp-0052-dba \
        --json reports/exp-0052-dba.json
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
from reid.models.fusion_postproc import build_post, flat

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def dba(g, alpha, k, sim_pow=1.0, use_self=True):
    """Augment gallery rows with their top-k gallery neighbours.

    g        : (N, D) float32 gallery embeddings
    alpha    : weight of the neighbour term
    k        : number of neighbours
    sim_pow  : cosine similarity is raised to this power (weights)
    use_self : include the row itself (mixture stays anchored) -- for pure DBA
               False reproduces the classic 'neighbour mean' variant.
    """
    if alpha <= 0 or k <= 0:
        return np.ascontiguousarray(g, dtype=np.float32)
    gn = l2(g)
    sim = gn @ gn.T                                   # (N, N) cosine
    if not use_self:
        np.fill_diagonal(sim, -np.inf)
    kk = int(min(k, g.shape[0]))
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :kk]   # self first if use_self
    w = np.take_along_axis(sim, idx, axis=1)
    if sim_pow != 1.0:
        w = np.clip(w, 0.0, None) ** float(sim_pow)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    nb = np.einsum("nk,nkd->nd", w.astype(np.float32), gn[idx])
    out = (1.0 - alpha) * g + alpha * nb
    return np.ascontiguousarray(l2(out), dtype=np.float32)


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


def eval_dba(q, g, ids, evaldir, rr_cfg, tag):
    q_ids, g_ids, gt, qcsv, gcsv = ids
    os.makedirs(evaldir, exist_ok=True)
    sub = os.path.join(evaldir, "submission.csv")
    emb = os.path.join(evaldir, "embeddings.npy")
    np.save(emb, np.vstack([q, g]).astype(np.float32))

    sims = q @ g.T
    o0 = np.argsort(-sims, axis=1, kind="stable")
    write_sub(sub, q_ids, g_ids, o0, sims)
    m_base = flat(run_official(gt_csv=gt, submission=sub,
                               candidates=os.path.join(evaldir, "candidates.csv"),
                               embeddings=emb, query=qcsv, gallery=gcsv,
                               json_out=os.path.join(evaldir, "official_base.json")))

    preps = [rerank.prepare_query(q[i], g, pool_size=rr_cfg["pool_size"])
             for i in range(q.shape[0])]
    o = np.empty((q.shape[0], g.shape[0]), np.int64)
    sc = np.empty((q.shape[0], g.shape[0]), np.float32)
    for i in range(q.shape[0]):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=rr_cfg["k1"],
                                           k2=rr_cfg["k2"], lam=rr_cfg["lam"])
    write_sub(sub, q_ids, g_ids, o, sc)
    m_rr = flat(run_official(gt_csv=gt, submission=sub,
                             candidates=os.path.join(evaldir, "candidates.csv"),
                             embeddings=emb, query=qcsv, gallery=gcsv,
                             json_out=os.path.join(evaldir, "official_rr.json")))
    print(f"{tag:44s} base={m_base['mAP@10']:.4f} rr={m_rr['mAP@10']:.4f} "
          f"R1={m_rr['Rank-1']:.4f} mINP={m_rr['mINP']:.4f}", flush=True)
    return {"tag": tag, "base_mAP@10": m_base["mAP@10"], "mAP@10": m_rr["mAP@10"],
            "Rank-1": m_rr["Rank-1"], "Rank-5": m_rr["Rank-5"],
            "mINP": m_rr["mINP"], "base_mINP": m_base["mINP"]}


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
    ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr)
            return 2
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dino = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == dino.shape[0] == nq + len(vg)

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    rr_cfg = {"k1": args.k1, "k2": args.k2, "lam": args.lam, "pool_size": args.pool}
    q, g = build_post(sig, dino, nq, {"w": args.w})
    print(f"[fusion] w={args.w} q={q.shape} g={g.shape} rr={rr_cfg}", flush=True)

    evaldir = os.path.join(out, "eval")
    rows = []
    # baseline: no DBA
    rows.append(eval_dba(q, g, ids, os.path.join(evaldir, "none"), rr_cfg, "DBA off"))

    # alpha x k sweep (plain mean, self-included)
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5):
        for k in (2, 3, 5, 10):
            gd = dba(g, alpha, k, sim_pow=1.0, use_self=True)
            tag = f"DBA a={alpha} k={k} mean"
            rows.append(eval_dba(q, gd, ids, os.path.join(evaldir, f"a{alpha}k{k}"),
                                 rr_cfg, tag))
    # similarity-weighted variant around the likely sweet spot
    for alpha in (0.2, 0.3, 0.4):
        for k in (3, 5, 10):
            gd = dba(g, alpha, k, sim_pow=3.0, use_self=True)
            tag = f"DBA a={alpha} k={k} w^3"
            rows.append(eval_dba(q, gd, ids, os.path.join(evaldir, f"a{alpha}k{k}p3"),
                                 rr_cfg, tag))

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "DBA on fusion gallery",
              "w": args.w, "rr": rr_cfg, "results": rows, "best": best}
    with open(os.path.join(out, "dba_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
