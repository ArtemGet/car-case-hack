"""CPU postproc: does a higher-resolution DINOv2 component improve the champion?

Consumes DINOv2 val embeddings extracted at several sizes
(``reid.models.extract_scales`` -> ``features.npz``) plus the SigLIP2 component
(reused from ``runs/W2-5-val-siglip``), fuses them exactly like the champion
(``L2(concat[L2(sig), w*L2(dino)])``), applies the champion per-query
k-reciprocal ``k1=8,k2=3,lam=0.5,pool=200`` and reports OFFICIAL metrics.

Query-side TTA is scale averaging of ONE image (no hflip) -- allowed. The
gallery uses a single view per config; a gallery DBA/smoothing variant was
already tried separately (``dba_postproc``) and did not help.

    python -m reid.models.dino_res_postproc \
        --features runs/exp-0053-dinores/features.npz \
        --out runs/exp-0053-dinores --json reports/exp-0053-dinores.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.dba_postproc import eval_dba, l2
from reid.models.fusion_postproc import build_post

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fuse_q(*views):
    """L2-normalised mean of several (nq, D) views of the SAME query images."""
    v = [l2(np.asarray(x, dtype=np.float32)) for x in views]
    return l2(np.mean(np.stack(v, axis=0), axis=0))


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--ref-dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--features", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    ng = len(vg)
    assert sig.shape[0] == nq + ng

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)
    rr_cfg = {"k1": args.k1, "k2": args.k2, "lam": args.lam, "pool_size": args.pool}

    feats = np.load(args.features)
    sizes = sorted({int(k.split("_")[1]) for k in feats.files})
    print(f"[features] sizes={sizes} files={feats.files}", flush=True)
    qv = {s: feats[f"q_{s}"] for s in sizes}
    gv = {s: feats[f"g_{s}"] for s in sizes}

    # ---- dino variants (q, g) -> stacked (nq+ng, 512) ----
    variants = {}
    # reference: the deployed component (existing TTA 224+280 query, base 224 gallery)
    ref_dino = np.load(os.path.join(args.ref_dino_run, "embeddings.npy")).astype(np.float32)
    variants["ref existing 224/280 TTA"] = ref_dino
    # highest single scale for both q and g
    top = max(sizes)
    variants[f"base {top} q+g"] = np.vstack([qv[top], gv[top]])
    if 224 in sizes:
        variants["base 224 q+g"] = np.vstack([qv[224], gv[224]])
    # gallery at highest scale, query TTA over available scales
    q_tta_all = fuse_q(*[qv[s] for s in sizes])
    variants[f"g={top} q=TTA({sizes})"] = np.vstack([q_tta_all, gv[top]])
    # gallery 336, query TTA 336+280 (no 224)
    if top in sizes and 280 in sizes:
        variants[f"g={top} q=TTA(280,{top})"] = np.vstack([fuse_q(qv[280], qv[top]), gv[top]])
        variants[f"g=280 q=TTA(280,{top})"] = np.vstack([fuse_q(qv[280], qv[top]), gv[280]])

    rows = []
    for name, dino in variants.items():
        for w in (0.6, 0.8, 1.0):
            q, g = build_post(sig, dino, nq, {"w": w})
            tag = f"{name} w={w}"
            ed = os.path.join(out, "eval", name.replace(" ", "_").replace(",", "-") + f"_w{w}")
            rows.append(eval_dba(q, g, ids, ed, rr_cfg, tag))
            rows[-1]["variant"] = name; rows[-1]["w"] = w

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "DINOv2 resolution sweep",
              "rr": rr_cfg, "sizes": sizes, "results": rows, "best": best}
    with open(os.path.join(out, "dino_res_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
