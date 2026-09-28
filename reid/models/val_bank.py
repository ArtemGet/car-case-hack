"""Extract a DINOv2 ReID checkpoint's hold-out val embeddings (GPU-only).

Feeds the EX-4 fusion grid. Mirrors the deployed DINOv2 contract: aspect-
preserving square crop -> eval transform -> L2 embedding, with optional
query-side TTA over /14 scales (224 + 280) averaged as L2(mean_s L2(e_s)) --
the SAME convention as ``runs/W2-5-val-dino``.

    python docs\\_workspace\\tools\\gpu_lock.py run --wait 3600 -- \
        python -m reid.models.val_bank --ckpt runs/exp-0078-ex4-triplet/best.pt \
        --sizes 224,280 --out runs/exp-0078-ex4-triplet/val_features.npz
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
        raise SystemExit("dataset dir not found")
    return os.path.dirname(cand[0])


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def do_tta(banks):
    """L2(mean_s L2(e_s)) -- matches the deployed 224+280 mean-TTA."""
    stacked = np.stack([l2(b) for b in banks], axis=0)  # (S, N, D)
    return np.ascontiguousarray(l2(stacked.mean(axis=0)), dtype=np.float32)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--sizes", default="224,280")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (project rule) but unavailable")
    from service.infer.backends import DinoBackend, set_determinism
    set_determinism(42)

    ds = args.dataset or _find_dataset_dir()
    images_dir = os.path.join(ds, "images")
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"[val_bank] dataset={ds} sizes={sizes} ckpt={args.ckpt}", flush=True)

    df = read_csv(os.path.join(ds, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    print(f"[val_bank] q={len(vq)} g={len(vg)}", flush=True)

    backend = DinoBackend(args.ckpt, device="cuda", amp="bf16",
                          batch_size=args.batch_size, num_workers=0)
    # DinoBackend stores the raw string; its extract() compares self.device.type.
    backend.device = torch.device("cuda")
    out = {}
    q_banks, g_banks = [], []
    for s in sizes:
        print(f"[val_bank] dino @{s}", flush=True)
        q = backend.extract(vq, images_dir, s)
        g = backend.extract(vg, images_dir, s)
        out[f"q_{s}"] = q
        out[f"g_{s}"] = g
        q_banks.append(q)
        g_banks.append(g)

    if len(sizes) > 1:
        # Deployed contract (service/infer/run.py): TTA is QUERY-only --
        # gallery uses the base scale, query averages all scales.
        out["q_tta"] = do_tta(q_banks)
        out["g_tta"] = np.ascontiguousarray(g_banks[0], dtype=np.float32)
    else:
        out["q_tta"], out["g_tta"] = q_banks[0], g_banks[0]
    print(f"[val_bank] tta q={out['q_tta'].shape} g={out['g_tta'].shape}",
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    # W2-5-layout embeddings.npy (q then g) for drop-in fusion use
    emb_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "embeddings.npy")
    np.save(emb_path, np.vstack([out["q_tta"], out["g_tta"]]).astype(np.float32))
    vq.to_csv(os.path.join(os.path.dirname(emb_path), "query.csv"), index=False)
    vg.to_csv(os.path.join(os.path.dirname(emb_path), "gallery.csv"), index=False)
    print(f"[val_bank] saved {args.out} and {emb_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
