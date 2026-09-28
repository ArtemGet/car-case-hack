"""Independent re-verification of the champion fusion config (val, seed=42).

Reproduces ONLY the winning pipeline found in exp-0046:
    L2( concat[ L2(SigLIP2 frozen 512d), 0.8 * L2(DINOv2-B exp-0007 512d) ] )
    + per-query k-reciprocal (k1=8, k2=3, lam=0.5, pool=200)
and scores it with the OFFICIAL ``evaluate.py`` (via reid.eval.harness).

CPU-only post-processing over already-extracted, frozen embeddings: no model
forward, streaming invariant preserved (per-query ranking).
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


def flat(rep):
    r = rep.get("ranking", {}); fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"), "mAP_full": fr.get("mAP_full"),
            "mINP": fr.get("mINP"), "n_scored": r.get("n_scored")}


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
    ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr); return 2
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dino = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)
    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2,
                            seed=args.seed)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == dino.shape[0] == nq + len(vg), (sig.shape, dino.shape)
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()

    fused = l2(np.concatenate([l2(sig), args.w * l2(dino)], axis=1))
    q, g = fused[:nq], fused[nq:]
    emb = os.path.join(out, "embeddings.npy")
    np.save(emb, np.vstack([q, g]).astype(np.float32))
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)

    # baseline cosine
    sims = q @ g.T
    o0 = np.argsort(-sims, axis=1, kind="stable")
    sub = os.path.join(out, "submission.csv")
    write_sub(sub, q_ids, g_ids, o0, sims)
    m_base = flat(run_official(gt_csv=gt, submission=sub,
                               candidates=os.path.join(out, "candidates.csv"),
                               embeddings=emb, query=qcsv, gallery=gcsv,
                               json_out=os.path.join(out, "official_base.json")))

    preps = [rerank.prepare_query(q[i], g, pool_size=args.pool) for i in range(nq)]
    o = np.empty((nq, len(g)), np.int64); sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=args.k1, k2=args.k2,
                                           lam=args.lam)
    write_sub(sub, q_ids, g_ids, o, sc)
    m_rr = flat(run_official(gt_csv=gt, submission=sub,
                             candidates=os.path.join(out, "candidates.csv"),
                             embeddings=emb, query=qcsv, gallery=gcsv,
                             json_out=os.path.join(out, "official_rr.json")))

    result = {"agent": "ml-trainer", "task": "independent verify champion fusion",
              "config": {"w": args.w, "k1": args.k1, "k2": args.k2,
                         "lam": args.lam, "pool": args.pool, "seed": args.seed},
              "baseline": m_base, "rerank": m_rr,
              "sources": {"sig": os.path.join(args.sig_run, "embeddings.npy"),
                          "dino": os.path.join(args.dino_run, "embeddings.npy")}}
    print("BASE", json.dumps(m_base, ensure_ascii=False), flush=True)
    print("RR  ", json.dumps(m_rr, ensure_ascii=False), flush=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
