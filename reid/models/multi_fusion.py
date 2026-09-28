"""Multi-backbone fusion grid (CPU-only postproc).

Champion = 2-backbone concat (SigLIP2 + 0.8*DINOv2-TTA) + k-reciprocal.  Here we
test whether already-available public backbones add complementary signal when
appended to the concat with a tunable weight:

    banks:  sig   SigLIP2 NaFlex 512d   (frozen, exp-0032)
            dino  DINOv2-B 512d, TTA(224+280)  (exp-0007 / W2-5-val-dino)
            d224/d280/d336  DINOv2-B single scales (exp-0053 features.npz)
            clip  CLIP-ReID ViT 512d    (exp-0031, weak alone 0.218)
            rn    resnet34-VeRi 512d    (exp-0030, weak alone 0.091)

    fused = L2( concat[ L2(sig), w_d*L2(dino), w_c*L2(clip), w_r*L2(rn), ... ] )

The k-reciprocal pass is the champion one (k1=8,k2=3,lam=0.5,pool=300).  Every
method touches only query + gallery (streaming invariant).  Metrics are the
OFFICIAL evaluator.

    python -m reid.models.multi_fusion --out runs/exp-0060-multifusion \
        --json reports/exp-0060-multifusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.dba_postproc import eval_dba
from reid.models.fusion_postproc import build_post

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def build_multi(banks, nq, weights):
    """banks: list of (N, D) arrays. weights: list of floats. L2 concat."""
    parts = [w * l2(b.astype(np.float32)) for b, w in zip(banks, weights)]
    fused = l2(np.concatenate(parts, axis=1))
    return np.ascontiguousarray(fused[:nq], np.float32), \
        np.ascontiguousarray(fused[nq:], np.float32)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default="runs/exp-0032-zs-siglip2/raw_embeddings.npy")
    ap.add_argument("--dino", default="runs/W2-5-val-dino/embeddings.npy")
    ap.add_argument("--clip", default="runs/exp-0031-zs-clipreid/embeddings.npy")
    ap.add_argument("--rn", default="runs/exp-0030-zs-resnet34/embeddings.npy")
    ap.add_argument("--dinores", default="runs/exp-0053-dinores/features.npz")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pool", type=int, default=300)
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    import pandas as pd
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    sig = np.load(args.sig).astype(np.float32)
    dino = np.load(args.dino).astype(np.float32)
    clip = np.load(args.clip).astype(np.float32)
    rn = np.load(args.rn).astype(np.float32)
    z = np.load(args.dinores)
    d224 = np.vstack([z["q_224"], z["g_224"]]).astype(np.float32)
    d280 = np.vstack([z["q_280"], z["g_280"]]).astype(np.float32)
    d336 = np.vstack([z["q_336"], z["g_336"]]).astype(np.float32)
    for nm, arr in [("sig", sig), ("dino", dino), ("clip", clip), ("rn", rn),
                    ("d224", d224), ("d280", d280), ("d336", d336)]:
        assert arr.shape[0] == nq + len(vg), (nm, arr.shape)

    rr = {"k1": 8, "k2": 3, "lam": 0.5, "pool_size": args.pool}
    evaldir = os.path.join(out, "eval")
    rows = []

    def run(tag, banks, weights):
        q, g = build_multi(banks, nq, weights)
        r = eval_dba(q, g, ids, os.path.join(evaldir, tag), rr, tag)
        r["weights"] = weights
        rows.append(r)
        return r

    # reference champion
    run("ref_sig_dino08", [sig, dino], [1.0, 0.8])
    # separate dino scales instead of TTA
    run("scales_224_280", [sig, d224, d280], [1.0, 0.8, 0.8])
    run("scales_3", [sig, d224, d280, d336], [1.0, 0.7, 0.7, 0.7])
    # + clip
    for wc in (0.1, 0.2, 0.3, 0.4, 0.6):
        run(f"clip_w{wc}", [sig, dino, clip], [1.0, 0.8, wc])
    # + resnet34
    for wr in (0.1, 0.2, 0.3):
        run(f"rn_w{wr}", [sig, dino, rn], [1.0, 0.8, wr])
    # + clip + resnet
    for wc in (0.2, 0.3):
        for wr in (0.1, 0.2):
            run(f"clip{wc}_rn{wr}", [sig, dino, clip, rn], [1.0, 0.8, wc, wr])

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "multi-backbone fusion",
              "rr": rr, "results": rows, "best": best}
    with open(os.path.join(out, "multi_fusion_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
