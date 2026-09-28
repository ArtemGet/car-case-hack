"""tests/test_api.py — контракт FastAPI-фасада (W1-7) на заглушке.

Проверяем без GPU и без весов:
  * /health, /api/v1/models, генерацию /openapi.json;
  * /api/v1/search: top-N + confidence + отказ (флаг);
  * валидацию: bbox вне кадра -> 422, битый base64/формат -> 400;
  * /api/v1/explain: карта + наложение (data URL PNG);
  * /api/v1/jobs: создание и получение результата.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service.api.app import create_app
from service.api.config import AppConfig
from service.api.engine import StubEngine
from service.api.gallery import NumpyGalleryIndex

DIM = 32  # маленькая размерность заглушки — быстрые тесты


# ---------------------------------------------------------------------------
# Синтетика
# ---------------------------------------------------------------------------
def _make_image(seed: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    # крупное цветное пятно, чтобы кропы различались стабильно
    arr[10:40, 10:40] = np.array([(seed * 37) % 256, (seed * 91) % 256,
                                  (seed * 53) % 256], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _jpeg_b64(img: Image.Image) -> tuple[str, Image.Image]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), Image.open(io.BytesIO(raw)).convert("RGB")


def _fixture_app(threshold: float = 0.0, n_gallery: int = 6):
    engine = StubEngine(dim=DIM, seed=7, threshold=threshold)
    ids, embs = [], []
    for i in range(n_gallery):
        b64, decoded = _jpeg_b64(_make_image(100 + i))
        ids.append(f"g{i:02d}")
        embs.append(engine.embed(decoded, (10, 10, 30, 30)))
    gallery = NumpyGalleryIndex(np.stack(embs), ids)
    cfg = AppConfig(engine="stub", dim=DIM)
    return create_app(cfg=cfg, engine=engine, gallery=gallery), engine


@pytest.fixture()
def app_client():
    app, _ = _fixture_app()
    with TestClient(app) as client:
        yield client, app


def _query_payload(seed: int = 102, bbox=None) -> dict:
    b64, _ = _jpeg_b64(_make_image(seed))
    return {"image_base64": b64, "bbox": bbox or {"x": 10, "y": 10, "w": 30, "h": 30}}


# ---------------------------------------------------------------------------
# Служебное
# ---------------------------------------------------------------------------
def test_health(app_client):
    client, app = app_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "stub"
    assert body["gallery_size"] == 6
    # stub-режим: доступна только сама заглушка
    assert body["models_available"] == 1


def test_openapi_generates(app_client):
    client, _ = app_client
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["openapi"].startswith("3.")
    for path in ("/health", "/api/v1/models", "/api/v1/search",
                 "/api/v1/explain", "/api/v1/jobs"):
        assert path in spec["paths"], path


def test_models_lists_variants(app_client):
    client, _ = app_client
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    names = {m["name"] for m in body["models"]}
    assert {"stub", "dino", "siglip", "fusion"} <= names
    assert body["default"] == "stub"
    stub = next(m for m in body["models"] if m["name"] == "stub")
    assert stub["available"] is True
    # В stub-режиме реальные веса недоступны, даже если файлы существуют.
    assert {m["name"] for m in body["models"] if m["available"]} == {"stub"}
    assert {m["name"] for m in body["models"] if m["is_default"]} == {"stub"}


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------
def test_search_returns_topn_and_confidence(app_client):
    client, _ = app_client
    payload = _query_payload(seed=102)
    payload["top_k"] = 3
    r = client.post("/api/v1/search", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "stub"
    assert len(body["candidates"]) == 3
    assert [c["rank"] for c in body["candidates"]] == [1, 2, 3]
    # self-match: запрос — это gallery g02, ожидаем top-1 == g02
    assert body["candidates"][0]["gallery_id"] == "g02"
    assert body["confidence"] > 0.99
    assert body["accepted"] is True and body["refused"] is False
    assert body["latency_ms"] >= 0.0
    # монотонность score
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_search_refusal_flag():
    # порог 1.01 недостижим для cosine -> гарантированный отказ
    engine = StubEngine(dim=DIM, seed=7, threshold=1.01)
    ids, embs = [], []
    for i in range(6):
        b64, decoded = _jpeg_b64(_make_image(200 + i))
        ids.append(f"g{i:02d}")
        embs.append(engine.embed(decoded, (0, 0, 20, 20)))
    gallery = NumpyGalleryIndex(np.stack(embs), ids)
    app = create_app(cfg=AppConfig(engine="stub", dim=DIM), engine=engine,
                     gallery=gallery)
    with TestClient(app) as client:
        r = client.post("/api/v1/search", json=_query_payload(seed=5))
        assert r.status_code == 200
        body = r.json()
        assert body["refused"] is True and body["accepted"] is False
        assert len(body["candidates"]) >= 1  # top-N всё равно возвращаем


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------
def test_bbox_out_of_bounds_422(app_client):
    client, _ = app_client
    payload = _query_payload(bbox={"x": 60, "y": 60, "w": 20, "h": 20})
    r = client.post("/api/v1/search", json=payload)
    assert r.status_code == 422
    assert "границ" in r.json()["detail"]


def test_bbox_nonpositive_422(app_client):
    client, _ = app_client
    payload = _query_payload(bbox={"x": 0, "y": 0, "w": 0, "h": 10})
    r = client.post("/api/v1/search", json=payload)
    assert r.status_code == 422


def test_bad_base64_400(app_client):
    client, _ = app_client
    r = client.post("/api/v1/search",
                    json={"image_base64": "not*base64!!",
                          "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}})
    assert r.status_code == 400


def test_unsupported_format_400(app_client):
    client, _ = app_client
    buf = io.BytesIO()
    _make_image(1).save(buf, format="GIF")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    r = client.post("/api/v1/search",
                    json={"image_base64": b64,
                          "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------
def test_explain_returns_heatmap_and_overlay(app_client):
    client, _ = app_client
    r = client.post("/api/v1/explain", json=_query_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"]
    assert body["heatmap_png_base64"].startswith("data:image/png;base64,")
    assert body["overlay_png_base64"].startswith("data:image/png;base64,")
    assert body["bbox"] == {"x": 10, "y": 10, "w": 30, "h": 30}


def test_explain_validates_bbox(app_client):
    client, _ = app_client
    r = client.post("/api/v1/explain",
                    json=_query_payload(bbox={"x": 100, "y": 0, "w": 10, "h": 10}))
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def test_jobs_create_and_fetch(app_client):
    client, _ = app_client
    r = client.post("/api/v1/jobs", json=_query_payload(seed=101))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    r2 = client.get(f"/api/v1/jobs/{job_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "done"
    assert body["result"]["candidates"][0]["gallery_id"] == "g01"


def test_jobs_unknown_404(app_client):
    client, _ = app_client
    assert client.get("/api/v1/jobs/deadbeef").status_code == 404
