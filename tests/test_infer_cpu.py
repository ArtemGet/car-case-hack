"""W2-5 CPU-prototype: tests for the torch-free SigLIP2 CPU inference path.

Covers :mod:`service.infer.cpu_backend` and the ``--device cpu`` wiring in
:mod:`service.infer.run`:

* the ORT session exposes exactly ``["CPUExecutionProvider"]`` and produces a
  finite ``(N, 512)`` float32 L2-normalised embedding (real ONNX CPU, skipped
  when the weight file is absent);
* the CPU branch is reachable only through an explicit ``device="cpu"`` and
  rejects the DINOv2/fusion variants;
* the production GPU path still hard-requires CUDA (assert not broken);
* importing the runner does not import torch.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGLIP_ONNX = os.path.join(REPO, "artifacts", "siglip2_fp16.onnx")

sys.path.insert(0, REPO)

from service.infer import run as runner  # noqa: E402


def _make_images(tmp_path, n=2):
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(n):
        iid = f"img{i:02d}"
        Image.new("RGB", (80 + 7 * i, 60 + 5 * i),
                  (20 * i % 255, 90, 200 - i)).save(images / f"{iid}.jpg",
                                                    quality=90)
        rows.append((iid, 0, 0, 70, 50))
    return str(images), rows


def _df(rows):
    import pandas as pd

    return pd.DataFrame(rows, columns=["image_id", "x", "y", "w", "h"])


@pytest.mark.skipif(not os.path.exists(SIGLIP_ONNX),
                    reason="artifacts/siglip2_fp16.onnx absent")
def test_cpu_backend_real_onnx(tmp_path):
    from service.infer.cpu_backend import SiglipCpuBackend, rss_mb

    images, rows = _make_images(tmp_path, n=2)
    rss0 = rss_mb()
    be = SiglipCpuBackend(SIGLIP_ONNX, threads=2)
    # exactly one provider, and it is the CPU EP
    assert list(be.sess.get_providers()) == ["CPUExecutionProvider"]

    df = _df(rows)
    emb = be.extract(df, images, batch_size=2)
    assert emb.shape == (2, 512)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    # L2-normalised
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)

    # determinism: byte-identical on a re-run
    emb2 = be.extract(df, images, batch_size=2)
    assert np.array_equal(emb, emb2)

    # memory stays bounded and is reported
    assert not np.isnan(rss0)
    assert rss_mb() < 2048.0


def test_cpu_variant_guard():
    # fusion/dino are CUDA-only; the CPU branch must refuse them loudly.
    import pandas as pd

    df = pd.DataFrame([("a", 0, 0, 1, 1)],
                      columns=["image_id", "x", "y", "w", "h"])
    with pytest.raises(SystemExit):
        runner._build_embeddings_cpu("fusion", ".", df, df)


def test_cpu_device_is_explicit_only():
    assert runner._is_cpu_device("cpu")
    assert runner._is_cpu_device("CPU")
    assert not runner._is_cpu_device("auto")
    assert not runner._is_cpu_device("cuda")
    assert not runner._is_cpu_device(None)


def test_cli_device_default_and_choices():
    args = runner.parse_args([
        "--images", "i", "--query", "q", "--gallery", "g", "--out", "o",
        "--device", "cpu", "--max-ram-mb", "1800",
    ])
    assert args.device == "cpu"
    assert args.max_ram_mb == 1800


def test_gpu_path_still_requires_cuda():
    """The production path must not silently fall back to CPU (assert intact)."""
    from service.infer import backends

    # require_cuda=False gives CPU-only providers (used by the CPU branch only)
    assert backends._cuda_ort_providers(require_cuda=False) == \
        ["CPUExecutionProvider"]
    # GpuExtractor with a non-cuda device is refused before any session loads
    with pytest.raises(RuntimeError):
        backends.GpuExtractor("fusion", dino_onnx=None, device="cpu")


def test_runner_import_does_not_pull_torch():
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "import service.infer.run; "
            "print('TORCH' if 'torch' in sys.modules else 'CLEAN')" % REPO)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
