"""CPU grid: extend the champion k-reciprocal search to a larger candidate pool.

The frozen champion (exp-0046) fixed ``pool_size=200`` out of the two values
tried (100/200) on a gallery of 1528. A bigger pool may capture positive
gallery items that the cosine top-200 misses. This script re-evaluates the
champion fusion with larger pools and a few k1/k2/lam neighbours, using the
OFFICIAL evaluator. No feature changes -> same streaming invariant.

    python -m reid.models.rr_pool_grid --out runs/exp-0054-poolgrid \
        --json reports/exp-0054-poolgrid.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.dba_postproc import eval_dba
from reid.models.fusion_postproc import build_post

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    ap.add_argument("--w-list", default="", help="comma weights; overrides --w sweep")
    ap.add_argument("--fine", action="store_true", help="small pool/w refinement grid")
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dino = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)
    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    q, g = build_post(sig, dino, nq, {"w": args.w})
    grid = [
        (200, 8, 3, 0.5),    # champion reference
        (300, 8, 3, 0.5), (500, 8, 3, 0.5), (750, 8, 3, 0.5),
        (300, 12, 3, 0.5), (500, 12, 3, 0.5), (750, 12, 3, 0.5),
        (300, 16, 3, 0.5), (500, 16, 3, 0.5), (750, 16, 4, 0.5),
        (500, 10, 3, 0.6), (500, 12, 4, 0.4),
    ]
    rows = []
    wlist = [float(x) for x in args.w_list.split(",") if x.strip()] if args.w_list else [args.w]
    if args.fine:
        combos = []
        for w in wlist:
            for pool in (250, 300, 350, 400):
                combos.append((w, pool, 8, 3, 0.5))
    else:
        combos = [(args.w, p, k1, k2, lam) for (p, k1, k2, lam) in grid]
    for w, pool, k1, k2, lam in combos:
        q, g = build_post(sig, dino, nq, {"w": w})
        rr = {"k1": k1, "k2": k2, "lam": lam, "pool_size": pool}
        tag = f"w={w} pool={pool} k1={k1} k2={k2} lam={lam}"
        ed = os.path.join(out, f"w{w}_pool{pool}_k{k1}_{k2}_{lam}")
        row = eval_dba(q, g, ids, ed, rr, tag)
        row["rr"] = rr; row["w"] = w
        rows.append(row)

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    rep = {"agent": "ml-trainer", "task": "k-reciprocal pool grid", "w": args.w,
           "results": rows, "best": best}
    with open(os.path.join(out, "poolgrid_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
