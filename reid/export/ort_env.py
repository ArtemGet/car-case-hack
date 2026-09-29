#!/usr/bin/env python
"""Optional ONNX Runtime CUDA provider options, driven by environment variables.

The deployed ORT CUDA EP grows its memory arena to fill the GPU by default, so
the frozen fusion session shows a ~15.6 GB arena peak even though the fp16
weights are only ~345 MB.  This module lets an operator cap that arena at
deploy/bench time **without changing the default behaviour**: every variable
is opt-in and, when none is set, callers get their provider list back byte-for-
byte unchanged (plain strings, exactly as before).

Environment variables
---------------------
``REID_ORT_GPU_MEM_LIMIT_MB``
    Integer MB.  Sets ``gpu_mem_limit`` (bytes) on the CUDA EP — the total
    device memory the ORT arena may allocate.  Unset / non-positive -> ignored.
``REID_ORT_ARENA``
    ``same``   -> ``arena_extend_strategy=kSameAsRequested`` (grow by exact
                 request instead of doubling).
    ``off``    -> ``enable_cuda_mem_arena=false`` where the ORT build supports
                 it; on builds without that knob (e.g. 1.26) it falls back to
                 ``kSameAsRequested`` with a warning (ORT would otherwise reject
                 the unknown option and silently drop to the CPU EP).
    ``default``/unset -> untouched.
``REID_ORT_CUDNN_HEURISTIC=1``
    -> ``cudnn_conv_algo_search=HEURISTIC`` (skip the EXHAUSTIVE/BENCHMARK
    workspace probing that can itself reserve a large cuDNN workspace).

Only the CUDA EP entry is modified; ``CPUExecutionProvider`` (the explicit
per-node shape-op fallback) and any explicit CPU-only list pass through as-is.

Nothing here touches the network.
"""
from __future__ import annotations

import os

__all__ = ["cuda_provider_options", "apply_cuda_options"]

_TRUE = {"1", "true", "yes", "on"}
_SUPPORT_CACHE = {}


def _ort_cuda_supports(key: str) -> bool:
    """Best-effort check that this ORT build's CUDA EP accepts provider ``key``.

    ORT 1.26 rejects unknown provider options by *falling back to the CPU EP*
    (later failing on CUDA tensors), so we must not pass a key it doesn't know.
    We scan the CUDA provider DLL for the option token once and cache the result.
    """
    if key in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[key]
    ok = True
    try:
        import glob

        import onnxruntime as ort
        base = os.path.dirname(ort.__file__)
        blob = b""
        for p in glob.glob(os.path.join(base, "capi",
                                        "onnxruntime_providers_cuda*")):
            try:
                with open(p, "rb") as f:
                    blob += f.read()
            except OSError:
                pass
        if blob:
            ok = key.encode() in blob
    except Exception:  # noqa: BLE001 - never break session creation
        ok = True
    _SUPPORT_CACHE[key] = ok
    return ok


def cuda_provider_options():
    """Return the CUDA EP option dict from the env, or ``None`` if none set."""
    opts = {}

    raw_limit = os.environ.get("REID_ORT_GPU_MEM_LIMIT_MB")
    if raw_limit:
        try:
            mb = int(float(raw_limit))
        except (TypeError, ValueError):
            mb = 0
        if mb > 0:
            opts["gpu_mem_limit"] = str(int(mb) * 1024 * 1024)

    arena = (os.environ.get("REID_ORT_ARENA") or "").strip().lower()
    if arena == "same":
        opts["arena_extend_strategy"] = "kSameAsRequested"
    elif arena == "off":
        if _ort_cuda_supports("enable_cuda_mem_arena"):
            opts["enable_cuda_mem_arena"] = "false"
        else:
            # No arena-disable knob in this ORT build; exact-growth is the
            # closest supported behaviour and still shrinks the peak sharply.
            import sys
            print("[ort_env] REID_ORT_ARENA=off unsupported by this ORT build "
                  "(no enable_cuda_mem_arena); using kSameAsRequested",
                  file=sys.stderr, flush=True)
            opts["arena_extend_strategy"] = "kSameAsRequested"

    if (os.environ.get("REID_ORT_CUDNN_HEURISTIC") or "").strip().lower() in _TRUE:
        opts["cudnn_conv_algo_search"] = "HEURISTIC"

    return opts or None


def apply_cuda_options(providers):
    """Merge env CUDA options into a provider list; unchanged when none set.

    Accepts entries as plain strings or ``(name, options)`` tuples (ORT's own
    accepted provider format) and preserves order — so ``CUDAExecutionProvider``
    stays first whenever it was first.
    """
    opts = cuda_provider_options()
    if not opts:
        return providers
    out = []
    for p in providers:
        if isinstance(p, (tuple, list)) and p:
            name, base = p[0], dict(p[1]) if len(p) > 1 and p[1] else {}
        else:
            name, base = p, {}
        if name == "CUDAExecutionProvider":
            base.update(opts)
            out.append(("CUDAExecutionProvider", base))
        else:
            out.append(p)
    return out
