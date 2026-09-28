"""tests/test_api_gallery_image.py — превью кадров галереи.

Проверяем ``GET /api/v1/gallery/{gallery_id}/image``:
  * 200 + ``image/jpeg`` для существующего id (фикстурный .jpg в tmp);
  * 404 для неизвестного id, для path traversal и когда каталог не задан;
  * ``/api/v1/search`` заполняет ``image_url`` у кандидатов.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.api.app import create_app
from service.api.config import AppConfig
from service.api.engine import StubEngine
from service.api.gallery import NumpyGalleryIndex

DIM = 16


def _jpeg_bytes(seed: int, size: tuple[int, int] = (48, 48)) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _make_app(images_dir, ids=("abc123", "def456"), dim: int = DIM):
    engine = StubEngine(dim=dim, seed=3, threshold=0.0)
    embs = []
    raw: dict[str, bytes] = {}
    for i, gid in enumerate(ids):
        raw[gid] = _jpeg_bytes(10 + i)
        img = Image.open(io.BytesIO(raw[gid])).convert("RGB")
        embs.append(engine.embed(img, (0, 0, 20, 20)))
    gallery = NumpyGalleryIndex(np.stack(embs), list(ids))
    cfg = AppConfig(engine="stub", dim=dim, images_dir=images_dir)
    return create_app(cfg=cfg, engine=engine, gallery=gallery), raw


@pytest.fixture()
def tmp_images(tmp_path):
    ids = ("abc123", "def456")
    raw = {}
    for i, gid in enumerate(ids):
        raw[gid] = _jpeg_bytes(10 + i)
        (tmp_path / f"{gid}.jpg").write_bytes(raw[gid])
    return tmp_path, raw


# ---------------------------------------------------------------------------
# image endpoint
# ---------------------------------------------------------------------------
def test_gallery_image_200(tmp_images):
    images_dir, raw = tmp_images
    app, _ = _make_app(str(images_dir))
    with TestClient(app) as client:
        r = client.get("/api/v1/gallery/abc123/image")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == raw["abc123"]


def test_gallery_image_unknown_id_404(tmp_images):
    images_dir, _ = tmp_images
    app, _ = _make_app(str(images_dir))
    with TestClient(app) as client:
        r = client.get("/api/v1/gallery/does-not-exist/image")
    assert r.status_code == 404
    assert "не найден" in r.json()["detail"]


def test_gallery_image_traversal_404(tmp_images):
    images_dir, _ = tmp_images
    app, _ = _make_app(str(images_dir))
    with TestClient(app) as client:
        for evil in ("a..b", "..", "%2E%2E"):
            r = client.get(f"/api/v1/gallery/{evil}/image")
            assert r.status_code == 404, (evil, r.status_code)


def test_gallery_image_dir_unset_404(tmp_images):
    # id валиден и есть в галерее, но каталог изображений не сконфигурирован
    images_dir, _ = tmp_images
    app, _ = _make_app(None)
    with TestClient(app) as client:
        r = client.get("/api/v1/gallery/abc123/image")
    assert r.status_code == 404
    assert "REID_API_IMAGES_DIR" in r.json()["detail"]


def test_gallery_image_missing_file_404(tmp_images):
    # каталог задан, id есть в галерее, но файла нет
    images_dir, _ = tmp_images
    os.remove(str(images_dir / "abc123.jpg"))
    app, _ = _make_app(str(images_dir))
    with TestClient(app) as client:
        r = client.get("/api/v1/gallery/abc123/image")
    assert r.status_code == 404
    assert "не найден" in r.json()["detail"]


# ---------------------------------------------------------------------------
# /search проставляет image_url
# ---------------------------------------------------------------------------
def test_search_sets_image_url(tmp_images):
    images_dir, _ = tmp_images
    app, _ = _make_app(str(images_dir))
    b64 = base64.b64encode(_jpeg_bytes(10)).decode("ascii")
    payload = {"image_base64": b64,
               "bbox": {"x": 0, "y": 0, "w": 20, "h": 20}, "top_k": 2}
    with TestClient(app) as client:
        r = client.post("/api/v1/search", json=payload)
    assert r.status_code == 200, r.text
    cands = r.json()["candidates"]
    assert len(cands) == 2
    for c in cands:
        assert c["image_url"] == f"/api/v1/gallery/{c['gallery_id']}/image"
