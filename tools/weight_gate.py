#!/usr/bin/env python
"""Weight-size gate: total model-weight bytes under the solution must be < 2 GiB.

Scans the *submitted* weights for known model-weight extensions (.pt .pth .bin
.onnx .engine .plan .safetensors .ckpt .trt .pb .tflite .npz), sums their sizes and
exits non-zero when the total exceeds the 2 GiB limit (perf-block disqualification,
METRICS.md).

The gate measures *deliverables*, not the dev workspace. The organizers scan the
solution directory (the checked-out repository), so whatever is committed counts;
dev training checkpoints under ``runs/`` (``*.pt``, hundreds of MB each),
``dataset``/``cache`` and other unshipped artifacts are NOT part of the submission
and must not be counted. Three ways to get that:

    # 1. git-aware (most faithful): count only files git tracks in the submission
    #    (runs/ and other ignored dev weights are excluded automatically).
    python tools/weight_gate.py --git

    # 2. explicit directory: count only the baked deliverable weights
    python tools/weight_gate.py --include-dir artifacts

    # 3. whole tree with dev dirs pruned (default)
    python tools/weight_gate.py

Usage:
    python tools/weight_gate.py [--dir ] [--limit-gb 2.0]
                                [--git | --include-dir artifacts]...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

WEIGHT_EXTS = {
    ".pt", ".pth", ".bin", ".onnx", ".engine", ".plan",
    ".safetensors", ".ckpt", ".trt", ".pb", ".tflite", ".npz",
}

# Directories that are not part of the submission / not deliverable weights.
# NOTE: ``runs`` holds dev training checkpoints (*.pt, hundreds of MB each) and
# ``data``/``dataset``/``cache`` hold inputs; none of them are submitted. Only the
# baked runtime weights (``artifacts/*.onnx``) count toward the 2 GiB gate.
PRUNE_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "env",
              "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
              ".mypy_cache", "docs", "runs", "data", "dataset", "models",
              "out", "cache"}

MB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def iter_weights(root: str, include_dirs=None):
    roots = [root]
    if include_dirs:
        roots = [d if os.path.isabs(d) else os.path.join(root, d)
                 for d in include_dirs]
    for start in roots:
        if not os.path.isdir(start):
            continue
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext in WEIGHT_EXTS:
                    full = os.path.join(dirpath, name)
                    try:
                        size = os.path.getsize(full)
                    except OSError:
                        continue
                    yield full, size


def iter_git_weights(root: str):
    """Yield (path, size) for *tracked* weight files only — the submission set.

    Uses ``git ls-files`` so ignored dev checkpoints (``runs/*.pt`` etc.) are
    never counted: the gate reflects exactly what a clean clone / the organizers
    would see. Raises RuntimeError when ``root`` is not a git work tree or git is
    unavailable, so callers can fail loudly instead of silently over/under-counting.
    """
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z"],
            check=True, capture_output=True,
        ).stdout
    except FileNotFoundError as exc:  # git not installed
        raise RuntimeError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"not a git work tree ({detail})") from exc

    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        if os.path.splitext(rel)[1].lower() not in WEIGHT_EXTS:
            continue
        full = os.path.join(root, rel)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        yield full, size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Model-weight size gate (< 2 GiB).")
    ap.add_argument("--dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                    help="solution directory to scan (default: repository root)")
    ap.add_argument("--limit-gb", type=float, default=2.0,
                    help="size limit in GiB (default: 2.0)")
    ap.add_argument("--include-dir", action="append", default=None,
                    metavar="SUB",
                    help="scan only this subdirectory (relative to --dir, or "
                         "absolute); repeatable. Default: whole --dir with dev "
                         "dirs (runs/, data/, ...) pruned.")
    ap.add_argument("--git", action="store_true",
                    help="git-aware: count only files tracked by git (the actual "
                         "submission). Excludes ignored dev checkpoints under "
                         "runs/. Mutually exclusive with --include-dir.")
    args = ap.parse_args(argv)

    if args.git and args.include_dir:
        ap.error("--git and --include-dir are mutually exclusive")

    root = os.path.abspath(args.dir)
    limit_bytes = int(args.limit_gb * GIB)

    total = 0
    files = []
    if args.git:
        try:
            entries = list(iter_git_weights(root))
        except RuntimeError as exc:
            print(f"weight-gate: FAIL — {exc}", file=sys.stderr)
            return 2
        for full, size in entries:
            total += size
            files.append((full, size))
    else:
        for full, size in iter_weights(root, args.include_dir):
            total += size
            files.append((full, size))

    files.sort(key=lambda kv: kv[1], reverse=True)
    scope = " (git-tracked only)" if args.git \
        else (f" (include: {', '.join(args.include_dir)})" if args.include_dir else "")
    print(f"weight-gate: scanning {root}" + scope)
    for full, size in files:
        rel = os.path.relpath(full, root)
        print(f"  {size / MB:12.3f} MB  {rel}")

    print(f"weight-gate: {len(files)} file(s), total {total / MB:.3f} MB "
          f"({total / GIB:.4f} GiB), limit {args.limit_gb:.2f} GiB")

    if total > limit_bytes:
        print(f"weight-gate: FAIL — exceeds limit by {(total - limit_bytes) / MB:.3f} MB",
              file=sys.stderr)
        return 1
    print("weight-gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
