"""Offline test: drop likely same-camera duplicates (junk) BEFORE top-10.

Official protocol removes gallery items sharing BOTH vehicle_id AND camera_id
with the query *before* truncation to 10. At inference we have no labels, but
same-camera-same-vehicle pairs are near-duplicate views and score extremely high
on cosine. Masking those out of the ranking (per query, gallery only) lets the
top-10 be filled with valid cross-camera matches instead of slots the official
evaluator would silently discard. Threshold is tuned on val.

    python -m reid.models.strip_junk_eval --run runs/exp-0032-zs-siglip2 \
        --dataset "docs/<ds>/dataset"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_sub(path, q_ids, g_ids, orders, top_k=10, scores=None):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])


def _write_cand(path, q_ids, g_ids, orders, scores):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def _write_gt(val_query, val_gallery, path):
    import pandas as pd
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy(); q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy(); g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def _flat(rep):
    r = rep.get("ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"), "Rank-5": r.get("Rank-5")}


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--thresholds", default="1.01,0.999,0.995,0.99,0.98,0.95,0.90,0.80")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not args.dataset:
        print("ERROR: dataset required", file=sys.stderr); return 2

    out = os.path.abspath(args.run)
    emb = np.load(os.path.join(out, "raw_embeddings.npy"))
    train_full = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(train_full, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    q, g = emb[:nq], emb[nq:]
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()
    sims = q @ g.T

    # how much headroom: junk slots inside the baseline top-10
    junk = (vg["vehicle_id"].to_numpy()[None, :] == vq["vehicle_id"].to_numpy()[:, None]) & \
           (vg["camera_id"].to_numpy()[None, :] == vq["camera_id"].to_numpy()[:, None])
    order0 = np.argsort(-sims, axis=1, kind="stable")
    top10 = np.take_along_axis(order0, np.arange(10)[None, :], axis=1)
    n_junk_top10 = int(junk[np.arange(nq)[:, None], top10].sum())
    print(f"junk pairs inside baseline top-10: {n_junk_top10} / {nq*10}", flush=True)

    # similarity distribution of junk vs true positives
    jt = sims[junk]
    tp = (vg["vehicle_id"].to_numpy()[None, :] == vq["vehicle_id"].to_numpy()[:, None]) & ~junk
    tt = sims[tp]
    print(f"junk sim: mean={jt.mean():.4f} p5={np.percentile(jt,5):.4f} "
          f"p50={np.percentile(jt,50):.4f}", flush=True)
    print(f"true-pos sim: mean={tt.mean():.4f} p50={np.percentile(tt,50):.4f} "
          f"p95={np.percentile(tt,95):.4f}", flush=True)

    gt = os.path.join(out, "gt.csv"); _write_gt(vq, vg, gt)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    sub = os.path.join(out, "submission.csv"); cand = os.path.join(out, "candidates.csv")

    results = {}
    for T in [float(x) for x in args.thresholds.split(",")]:
        masked = np.where(sims >= T, -np.inf, sims)
        # keep one finite fallback per query to avoid all -inf
        orders = np.argsort(-masked, axis=1, kind="stable")
        _write_sub(sub, q_ids, g_ids, orders)
        _write_cand(cand, q_ids, g_ids, orders, sims)
        np.save(os.path.join(out, "embeddings.npy"), emb)
        rep = run_official(gt_csv=gt, submission=sub, candidates=cand,
                           embeddings=os.path.join(out, "embeddings.npy"),
                           query=qcsv, gallery=gcsv,
                           json_out=os.path.join(out, f"official_strip_{T}.json"))
        m = _flat(rep)
        removed = int((sims >= T).sum())
        results[str(T)] = {"metrics": m, "removed_pairs": removed}
        print(f"  T={T:<6} mAP@10={m['mAP@10']:.4f} R1={m['Rank-1']:.4f} "
              f"removed={removed}", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
