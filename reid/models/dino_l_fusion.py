"""CPU fusion grids for the W4.5 in-domain large-backbone challenger.

Combines the champion's frozen SigLIP2 bank with the in-domain fine-tuned
DINOv2-L bank (``reid/models/sig_ft_train`` is SigLIP2; DINOv2-L is trained by
``reid.train`` with ``configs/dinov2_l.yaml`` then extracted via
``reid.models.extract_scales``):

    2-way:  L2( concat[ L2(sig), w * L2(dinoL) ] )
    3-way:  L2( concat[ L2(sig), a * L2(dinoB), b * L2(dinoL) ] )
      -> k-reciprocal (k1=8, k2=3, lam=0.5, pool=300)   [champion rr config]

CPU-only post-processing over stored val banks; the query set is never a fit
target (streaming invariant preserved). Metrics via the OFFICIAL ``evaluate.py``.

    python -m reid.models.dino_l_fusion --mode 3way --out runs/exp-0073-fusion-3way
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
CHAMPION = 0.7234035876255264


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2way", "3way"], required=True)
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--dinob-run", default="runs/W2-5-val-dino")
    ap.add_argument("--dinol-run", default="runs/exp-0073-dinov2l-ft")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--w-list", default="0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--a-list", default="0.6,0.8,1.0")
    ap.add_argument("--b-list", default="0.2,0.3,0.4,0.5,0.6,0.8,1.0")
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr)
            return 2
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = l2(np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32))
    dinoB = l2(np.load(os.path.join(args.dinob_run, "embeddings.npy")).astype(np.float32))
    dinoL = l2(np.load(os.path.join(args.dinol_run, "embeddings.npy")).astype(np.float32))
    assert sig.shape == dinoB.shape == dinoL.shape, (sig.shape, dinoB.shape, dinoL.shape)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
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

    def run(feats, tag):
        row = eval_dba(feats[:nq], feats[nq:], ids,
                       os.path.join(evaldir, tag.replace(".", "p")), rr, tag)
        rows.append(row)
        return row

    if args.mode == "2way":
        run(sig, "sig alone")
        run(dinoL, "dinoL alone")
        for w in [float(x) for x in args.w_list.split(",") if x.strip()]:
            run(l2(np.concatenate([sig, w * dinoL], axis=1)), f"sig+{w}dinoL")
    else:
        run(dinoL, "dinoL alone")
        for av in [float(x) for x in args.a_list.split(",") if x.strip()]:
            for bv in [float(x) for x in args.b_list.split(",") if x.strip()]:
                run(l2(np.concatenate([sig, av * dinoB, bv * dinoL], axis=1)),
                    f"sig+{av}dinoB+{bv}dinoL")

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    rep = {"agent": "ml-trainer", "task": f"in-domain DINOv2-L {args.mode} fusion",
           "rr": rr, "results": rows, "best": best,
           "champion_ref": CHAMPION,
           "delta_vs_champion": (best["mAP@10"] or 0) - CHAMPION}
    with open(os.path.join(out, "fusion_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
