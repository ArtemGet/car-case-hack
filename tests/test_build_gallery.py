"""tests/test_build_gallery.py — tools/build_gallery.py (CPU-галерея для API-демо).

Контрактные тесты (веса не нужны): резолв каталога/CSV, guard варианта,
отсутствие изображений. Реальная сборка (skip без ``artifacts/siglip2_fp16.onnx``
или onnxruntime): синтетический датасет -> gallery.npy ``(N,512)`` f32 +
gallery_ids.json, загрузка через ``service.api.gallery.load_gallery`` и
самопоиск top-1 == собственный id.

Без GPU и без сети.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

from conftest import weights_available

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_PATH = os.path.join(REPO, "tools", "build_gallery.py")
SIGLIP_ONNX = os.path.join(REPO, "artifacts", "siglip2_fp16.onnx")


def _load_tool():
    import sys

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("build_gallery", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bg = _load_tool()

try:
    import onnxruntime  # noqa: F401

    _HAS_ORT = True
except Exception:  # noqa: BLE001
    _HAS_ORT = False

# weights_available() == False и для отсутствующего файла, и для git-lfs-указателя.
_HAS_WEIGHTS = weights_available(SIGLIP_ONNX)
_needs_weights = pytest.mark.skipif(
    not (_HAS_WEIGHTS and _HAS_ORT),
    reason="artifacts/siglip2_fp16.onnx (нет или LFS-указатель) или onnxruntime отсутствуют")


def _make_dataset(root, n=5, missing=False):
    images = os.path.join(root, "images")
    os.makedirs(images, exist_ok=True)
    rows = ["image_id,x,y,w,h"]
    for i in range(n):
        iid = f"g{i:02d}"
        Image.new("RGB", (96 + 4 * i, 64 + 3 * i),
                  (30 * i % 255, 80, 190 - i)).save(
            os.path.join(images, f"{iid}.jpg"), quality=90)
        rows.append(f"{iid},5,5,70,50")
    if missing:
        rows.append("ghost,0,0,10,10")  # нет файла
    with open(os.path.join(root, "test_gallery.csv"), "w",
              encoding="utf-8", newline="") as f:
        f.write("\n".join(rows) + "\n")
    return root


# ---------------------------------------------------------------------------
# Контрактные (без весов)
# ---------------------------------------------------------------------------
def test_resolve_images_dir(tmp_path):
    ds = _make_dataset(str(tmp_path))
    assert bg.resolve_images_dir(ds) == os.path.join(ds, "images")
    # каталог из одних jpg тоже принимается
    assert bg.resolve_images_dir(os.path.join(ds, "images")) == \
        os.path.join(ds, "images")


def test_resolve_csv_relative_to_dataset(tmp_path):
    ds = _make_dataset(str(tmp_path))
    assert bg.resolve_csv(ds, "test_gallery.csv") == \
        os.path.join(ds, "test_gallery.csv")
    with pytest.raises(SystemExit):
        bg.resolve_csv(ds, "no_such.csv")


def test_variant_guard(tmp_path):
    ds = _make_dataset(str(tmp_path))
    with pytest.raises(SystemExit):
        bg.build_gallery(ds, "test_gallery.csv",
                         os.path.join(ds, "out", "gallery.npy"),
                         variant="fusion")


def test_missing_image_rejected(tmp_path):
    ds = _make_dataset(str(tmp_path), n=2, missing=True)
    with pytest.raises(SystemExit) as ei:
        bg.build_gallery(ds, "test_gallery.csv",
                         os.path.join(ds, "out", "gallery.npy"))
    assert "ghost" in str(ei.value)


def test_parse_args_defaults():
    args = bg.parse_args(["--dataset", "d", "--out", "o/gallery.npy"])
    assert args.csv == "test_gallery.csv"
    assert args.variant == "siglip"
    assert args.limit == 0
    assert args.threads == 8


# ---------------------------------------------------------------------------
# Реальная CPU-сборка (skip без весов)
# ---------------------------------------------------------------------------
@_needs_weights
def test_build_gallery_real_cpu(tmp_path):
    ds = _make_dataset(str(tmp_path), n=5)
    out = os.path.join(str(tmp_path), "out", "gallery.npy")
    report = bg.build_gallery(ds, "test_gallery.csv", out, threads=2,
                              batch_size=5)

    assert report["n_gallery"] == 5
    assert report["dim"] == 512
    assert report["providers"] == ["CPUExecutionProvider"]
    assert report["rss_after_mb"] < 2048.0
    assert report["self_search"]["top1_self_rate"] == 1.0

    emb = np.load(out)
    assert emb.shape == (5, 512)
    assert emb.dtype == np.float32
    assert np.isfinite(emb).all()
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)

    from service.api.gallery import load_gallery

    g = load_gallery(out)
    assert g.dim == 512
    assert len(g.ids) == 5
    ids, scores = g.search(emb[0], top_k=5)
    assert ids[0] == g.ids[0]                 # самопоиск
    assert scores[0] > 0.99


@_needs_weights
def test_build_gallery_limit(tmp_path):
    ds = _make_dataset(str(tmp_path), n=6)
    out = os.path.join(str(tmp_path), "out", "gallery.npy")
    report = bg.build_gallery(ds, "test_gallery.csv", out, limit=2,
                              threads=2, batch_size=2)
    assert report["n_gallery"] == 2
    assert np.load(out).shape == (2, 512)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
