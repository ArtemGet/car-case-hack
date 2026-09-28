"""Run several frozen-backbone val extractions under ONE gpu-lock hold.

Keep GPU-heavy work in a single lock acquisition so no other agent interleaves.
"""
from __future__ import annotations

import sys

from reid.models import frozen_backbone_val as fv

JOBS = [
    ["--model", "dinov2l", "--sizes", "336,518",
     "--out", "runs/exp-0067-dinov2l-val"],
    ["--model", "eva02l", "--sizes", "336",
     "--out", "runs/exp-0068-eva02l-val"],
    ["--model", "siglip2so", "--sizes", "384",
     "--out", "runs/exp-0069-siglip2so-val"],
]


def main() -> int:
    rc = 0
    for job in JOBS:
        print(f"\n===== frozen_backbone_val {' '.join(job)} =====", flush=True)
        r = fv.main(job)
        if r != 0:
            rc = r
            print(f"!! job failed rc={r}: {job}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
