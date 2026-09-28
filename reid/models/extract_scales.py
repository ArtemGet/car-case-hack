"""Extract DINOv2-B (exp-0007) val embeddings at several square input sizes.

GPU-only (asserts CUDA). This is a *preparation* job feeding a CPU postproc
grid (``reid.models.dino_res_postproc``): we re-extract the DINOv2 component of
the champion fusion at higher resolution (280 / 336) to test whether resolution
beats the deployed 224 base (+280 query TTA).

Run under the shared GPU lock:
    python docs\\_workspace\\tools\\gpu_lock.py run --wait 3600 -- \
        python -m reid.models.extract_scales --sizes 224,280,336 \
        --out runs/exp-0053-dinores/features.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_dataset_dir():
    cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
    if not cand:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "train.csv"))
    if not cand:
        raise SystemExit("dataset dir not found")
    return os.path.dirname(cand[0])


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--ckpt", default=os.path.join("runs", "exp-0007", "best.pt"))
    ap.add_argument("--sizes", default="224,280,336")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (project rule) but torch.cuda.is_available()=False")
    from service.infer.backends import DinoBackend, set_determinism
    set_determinism(42)

    ds = args.dataset or _find_dataset_dir()
    images_dir = os.path.join(ds, "images")
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"[extract] dataset={ds} sizes={sizes} ckpt={args.ckpt}", flush=True)

    df = read_csv(os.path.join(ds, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)

    backend = DinoBackend(args.ckpt, device="cuda", amp="bf16",
                          batch_size=args.batch_size, num_workers=0)
    # DinoBackend stores the raw string; its extract() compares self.device.type.
    backend.device = torch.device("cuda")
    out = {}
    for s in sizes:
        print(f"[extract] dino pytorch @{s}", flush=True)
        out[f"q_{s}"] = backend.extract(vq, images_dir, s)
        out[f"g_{s}"] = backend.extract(vg, images_dir, s)
        print(f"[extract]   q={out[f'q_{s}'].shape} g={out[f'g_{s}'].shape}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[extract] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
