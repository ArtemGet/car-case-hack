"""CPU side of the frozen-backbone challenger: fuse a new frozen bank with the
champion SigLIP2 bank and score via the OFFICIAL ``evaluate.py``.

    champion = L2(concat[ L2(sig), w * L2(new) ])  -> k-reciprocal (pool=300)

Also scores each new bank ALONE, and a 3-way fusion with the old DINOv2-B bank.
No feature is fit on the query set; k-reciprocal stays per-query. CPU-only.

    python -m reid.models.frozen_fusion_grid --new-run runs/exp-0066-dinov2l-val \
        --out runs/exp-0066-fusion --json reports/exp-0066-fusion.json
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--new-run", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--w-list", default="0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5")
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = l2(np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32))
    dino = l2(np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32))
    new = l2(np.load(os.path.join(args.new_run, "embeddings.npy")).astype(np.float32))
    assert sig.shape[0] == dino.shape[0] == new.shape[0], (sig.shape, dino.shape, new.shape)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == nq + len(vg)

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    rr = {"k1": args.k1, "k2": args.k2, "lam": args.lam, "pool_size": args.pool}
    evaldir = os.path.join(out, "eval")
    rows = []

    def run_features(feats, tag, sub):
        q = l2(feats[:nq])
        g = l2(feats[nq:])
        row = eval_dba(q, g, ids, os.path.join(evaldir, sub), rr, tag)
        rows.append(row)
        return row

    # new alone
    run_features(new, "new alone", "alone")
    # 2-way fusion
    for w in [float(x) for x in args.w_list.split(",") if x.strip()]:
        fused = l2(np.concatenate([sig, w * new], axis=1))
        run_features(fused, f"sig+{w}*new", f"w{w}")
    # 3-way: sig + 0.8*dinoB + w*new (convex-ish)
    sig_dino = l2(np.concatenate([sig, 0.8 * dino], axis=1))
    for w in (0.4, 0.6, 0.8, 1.0):
        fused = l2(np.concatenate([sig_dino, w * new], axis=1))
        run_features(fused, f"sig+0.8*dinoB+{w}*new", f"3way{w}")

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    rep = {"agent": "ml-trainer", "task": "frozen backbone fusion",
           "new_run": args.new_run, "rr": rr, "results": rows, "best": best}
    with open(os.path.join(out, "fusion_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
