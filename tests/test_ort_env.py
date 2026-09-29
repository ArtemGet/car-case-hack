"""Unit tests for reid/export/ort_env.py (perf-engineer).

Contract: with no ``REID_ORT_*`` env set the provider list is returned exactly
unchanged (default deploy behaviour must not drift); each env var maps onto the
documented CUDA EP option.
"""
from __future__ import annotations

import importlib

from reid.export import ort_env


def _fresh(monkeypatch, **env):
    for k in ("REID_ORT_GPU_MEM_LIMIT_MB", "REID_ORT_ARENA",
              "REID_ORT_CUDNN_HEURISTIC"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(ort_env)


def test_default_is_unchanged(monkeypatch):
    m = _fresh(monkeypatch)
    base = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert m.cuda_provider_options() is None
    assert m.apply_cuda_options(base) == base
    assert m.apply_cuda_options(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_mem_limit(monkeypatch):
    m = _fresh(monkeypatch, REID_ORT_GPU_MEM_LIMIT_MB="12288")
    assert m.cuda_provider_options() == {"gpu_mem_limit": str(12288 * 1024 * 1024)}


def test_arena_modes(monkeypatch):
    m = _fresh(monkeypatch, REID_ORT_ARENA="same")
    assert m.cuda_provider_options() == {"arena_extend_strategy": "kSameAsRequested"}
    # `off` uses enable_cuda_mem_arena where supported, else kSameAsRequested
    m = _fresh(monkeypatch, REID_ORT_ARENA="off")
    monkeypatch.setattr(m, "_ort_cuda_supports", lambda key: True)
    assert m.cuda_provider_options() == {"enable_cuda_mem_arena": "false"}
    monkeypatch.setattr(m, "_ort_cuda_supports", lambda key: False)
    assert m.cuda_provider_options() == {"arena_extend_strategy": "kSameAsRequested"}
    m = _fresh(monkeypatch, REID_ORT_ARENA="default")
    assert m.cuda_provider_options() is None


def test_cudnn_heuristic(monkeypatch):
    m = _fresh(monkeypatch, REID_ORT_CUDNN_HEURISTIC="1")
    assert m.cuda_provider_options() == {"cudnn_conv_algo_search": "HEURISTIC"}


def test_merge_preserves_order_and_existing_opts(monkeypatch):
    m = _fresh(monkeypatch, REID_ORT_GPU_MEM_LIMIT_MB="8192")
    out = m.apply_cuda_options([("CUDAExecutionProvider", {"device_id": 0}),
                                "CPUExecutionProvider"])
    assert out[0][0] == "CUDAExecutionProvider"
    assert out[0][1]["device_id"] == 0
    assert out[0][1]["gpu_mem_limit"] == str(8192 * 1024 * 1024)
    assert out[1] == "CPUExecutionProvider"
