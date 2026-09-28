"""Measure the input pipeline vs GPU forward to locate the training bottleneck.

    python tools/profile_input.py --dataset "docs/Датасет/dataset" --n 512 --workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid.data.aug import build_eval_transform  # noqa: E402
from reid.data.io import read_csv, TRAIN_COLUMNS  # noqa: E402
from reid.train import VehicleDataset, loader_kwargs  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def bench_loader(name, df, dataset, tf, target, cache_dir, draft, workers, batch=64):
    ds = VehicleDataset(df, dataset, tf, target, cache_dir=cache_dir, draft_factor=draft)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        **loader_kwargs(workers))
    it = iter(loader)
    next(it)  # warmup
    t0, n = time.time(), 0
    for _ in range(2):  # two passes over the subset
        for _, _, idx in loader:
            n += len(idx)
    dt = time.time() - t0
    print(f"  {name:<22} {n/dt:7.1f} img/s", flush=True)
    return n / dt


def bench_gpu(backbone, target, batch=64, iters=40):
    from reid.models import build_model
    if not torch.cuda.is_available():
        print("  GPU: not available")
        return None
    model = build_model(backbone=backbone, num_classes=100, emb_dim=512,
                        pretrained=False, margin=0.3, scale=30.0, gem_p=3.0,
                        image_size=target).cuda().eval()
    x = torch.randn(batch, 3, target, target, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            model.embed(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model.embed(x)
        torch.cuda.synchronize()
    fps = batch * iters / (time.time() - t0)
    print(f"  {('GPU forward ' + backbone):<22} {fps:7.1f} img/s")
    return fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--target", type=int, default=224)
    ap.add_argument("--cache", default=os.path.join("artifacts", "cache", "train_320"))
    args = ap.parse_args()

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    df = df.head(args.n).reset_index(drop=True)
    tf = build_eval_transform(args.target)
    cache = os.path.abspath(args.cache)
    cache = cache if os.path.isdir(cache) else None

    print(f"input pipeline (n={len(df)}, workers={args.workers}):")
    bench_loader("full decode (old)", df, args.dataset, tf, args.target, None, 0,
                 args.workers)
    bench_loader("draft=2 (partial JPEG)", df, args.dataset, tf, args.target, None,
                 2.0, args.workers)
    if cache:
        bench_loader("crop cache 320", df, args.dataset, tf, args.target, cache, 2.0,
                     args.workers)
    else:
        print("  crop cache: MISSING")

    print("GPU:")
    bench_gpu("dinov2_b", args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
