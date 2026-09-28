"""Pre-crop train frames into a small JPEG cache (kills the JPEG-decode bottleneck).

The source frames are large; full-resolution decode per crop dominates data loading
and leaves the GPU idle. This builds a one-time cache of aspect-preserving square
crops (default 320 px) so each epoch only decodes tiny images.

    python tools/build_crop_cache.py --dataset "docs/Датасет/dataset" \
        --out artifacts/cache/train_320 --size 320 --workers 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DATASET = None
_OUT = None
_SIZE = 320
_QUALITY = 92


def _init(dataset, out, size, quality):
    global _DATASET, _OUT, _SIZE, _QUALITY
    _DATASET, _OUT, _SIZE, _QUALITY = dataset, out, size, quality


def _one(row):
    from reid.data.crop import open_cropped
    image_id, x, y, w, h = row
    dst = os.path.join(_OUT, image_id + ".jpg")
    if os.path.exists(dst):
        return image_id, "skip"
    img = open_cropped(_DATASET, image_id, x, y, w, h, target=_SIZE,
                       draft_factor=2.0)
    img.save(dst, "JPEG", quality=_QUALITY)
    return image_id, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(os.path.join(args.dataset, f"{args.split}.csv"))
    rows = [(r.image_id, r.x, r.y, r.w, r.h) for r in df.itertuples()]
    print(f"cache: {len(rows)} crops -> {args.out} ({args.size}px q{args.quality})",
          flush=True)

    t0 = time.time()
    done = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.dataset, args.out, args.size,
                                       args.quality)) as ex:
        futs = [ex.submit(_one, r) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            _, status = fut.result()
            if status == "skip":
                skipped += 1
            else:
                done += 1
            if i % 1000 == 0 or i == len(futs):
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i}/{len(futs)} ({rate:.0f}/s)", flush=True)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"done: {done} written, {skipped} skipped, {total/1e6:.1f} MB, "
          f"{time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
