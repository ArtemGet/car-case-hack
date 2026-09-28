"""tests/test_api_cpu.py — CPU-SigLIP2 движок и 503 при отсутствии галереи.

Две группы:

* **Быстрые контрактные** (веса не нужны): фабрика ``build_engine`` выбирает
  ``CpuSiglipEngine`` для ``engine=infer`` + ``device=cpu`` и НЕ трогает
  GPU-путь; ``/search`` отдаёт 503 (а не падает/500), когда галерея не задана,
  пуста или её файл не читается; stub-эндпоинты живы.
* **Реальный CPU-инференс** (skip, если нет ``artifacts/siglip2_fp16.onnx``
  или onnxruntime): 1 кадр → конечный L2-нормированный ``(512,)`` float32;
  ``/api/v1/search`` возвращает top-N и корректный флаг отказа.

Без GPU и без сети.
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
from service.api.engine import (
    CPU_SIGLIP_DEFAULT_THRESHOLD,
    CpuSiglipEngine,
    InferEngine,
    StubEngine,
    build_engine,
    describe_models,
)
from service.api.gallery import NumpyGalleryIndex

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGLIP_ONNX = os.path.join(REPO, "artifacts", "siglip2_fp16.onnx")

try:  # ONNX Runtime нужен реальному CPU-тесту
    import onnxruntime  # noqa: F401

    _HAS_ORT = True
except Exception:  # noqa: BLE001
    _HAS_ORT = False

_HAS_WEIGHTS = os.path.exists(SIGLIP_ONNX)
_needs_weights = pytest.mark.skipif(
    not (_HAS_WEIGHTS and _HAS_ORT),
    reason="artifacts/siglip2_fp16.onnx или onnxruntime отсутствуют")

DIM = 32


# ---------------------------------------------------------------------------
# Синтетика (как в tests/test_api.py)
# ---------------------------------------------------------------------------
def _make_image(seed: int, size: tuple[int, int] = (64, 64)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    arr[10:40, 10:40] = np.array([(seed * 37) % 256, (seed * 91) % 256,
                                  (seed * 53) % 256], dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _jpeg_b64(img: Image.Image) -> tuple[str, Image.Image]:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    return (base64.b64encode(raw).decode("ascii"),
            Image.open(io.BytesIO(raw)).convert("RGB"))


def _query_payload(seed: int = 102) -> dict:
    b64, _ = _jpeg_b64(_make_image(seed))
    return {"image_base64": b64,
            "bbox": {"x": 10, "y": 10, "w": 30, "h": 30}}


# ---------------------------------------------------------------------------
# Контрактные тесты (без весов)
# ---------------------------------------------------------------------------
def test_build_engine_cpu_selects_cpu_siglip():
    cfg = AppConfig(engine="infer", device="cpu", variant="fusion",
                    cpu_siglip_onnx=SIGLIP_ONNX, threshold=0.5)
    engine = build_engine(cfg)
    assert isinstance(engine, CpuSiglipEngine)
    assert engine.variant == "siglip"          # CPU-прототип: только SigLIP2
    assert engine.device == "cpu"
    assert engine.dim == 512
    assert engine.threshold == 0.5             # явный override


def test_build_engine_gpu_path_unchanged():
    # device != cpu -> production InferEngine (GPU-путь), не CPU-движок
    cfg = AppConfig(engine="infer", device="cuda", variant="fusion")
    engine = build_engine(cfg)
    assert isinstance(engine, InferEngine)
    assert not isinstance(engine, CpuSiglipEngine)


def test_build_engine_stub_default():
    engine = build_engine(AppConfig(engine="stub", dim=DIM))
    assert isinstance(engine, StubEngine)
    assert engine.name == "stub"


def test_models_stub_only_stub_available():
    cfg = AppConfig(engine="stub", dim=DIM,
                    dino_ckpt=SIGLIP_ONNX, siglip_onnx=SIGLIP_ONNX)
    models = {m["name"]: m for m in describe_models(cfg, StubEngine(dim=DIM))}
    assert {n for n, m in models.items() if m["available"]} == {"stub"}
    assert {n for n, m in models.items() if m["is_default"]} == {"stub"}


def test_models_cpu_only_siglip_available(monkeypatch, tmp_path):
    # Файлы весов есть для всех, но CPU-движок поддерживает только SigLIP2.
    dino = tmp_path / "dino.ckpt"
    dino.write_bytes(b"x")
    onnx = tmp_path / "siglip.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setattr(CpuSiglipEngine, "available", property(lambda self: True))
    engine = CpuSiglipEngine(str(onnx))
    cfg = AppConfig(engine="infer", device="cpu", dim=512,
                    dino_ckpt=str(dino), cpu_siglip_onnx=str(onnx))
    models = {m["name"]: m for m in describe_models(cfg, engine)}
    assert {n for n, m in models.items() if m["available"]} == {"siglip"}
    assert {n for n, m in models.items() if m["is_default"]} == {"siglip"}
    assert models["siglip"]["weights"] == str(onnx)

    app = create_app(cfg=cfg, engine=engine)
    with TestClient(app) as client:
        body = client.get("/api/v1/models").json()
        assert body["default"] == "siglip"
        assert {m["name"] for m in body["models"] if m["available"]} == {"siglip"}
        assert client.get("/health").json()["models_available"] == 1


def test_models_cpu_siglip_unavailable_without_onnx(monkeypatch):
    # Нет ONNX/onnxruntime -> даже siglip недоступен (не выдумываем доступность).
    monkeypatch.setattr(CpuSiglipEngine, "available", property(lambda self: False))
    engine = CpuSiglipEngine("missing.onnx")
    cfg = AppConfig(engine="infer", device="cpu", dim=512)
    models = {m["name"]: m for m in describe_models(cfg, engine)}
    assert {n for n, m in models.items() if m["available"]} == set()
    assert {n for n, m in models.items() if m["is_default"]} == {"siglip"}


def test_models_gpu_weights_logic(tmp_path):
    dino = tmp_path / "dino.ckpt"
    dino.write_bytes(b"x")
    onnx = tmp_path / "siglip.onnx"
    onnx.write_bytes(b"x")

    # Оба веса -> dino/siglip/fusion доступны, default — активный вариант.
    cfg = AppConfig(engine="infer", device="cuda", variant="fusion",
                    dino_ckpt=str(dino), siglip_onnx=str(onnx))
    engine = build_engine(cfg)
    models = {m["name"]: m for m in describe_models(cfg, engine)}
    assert {n for n, m in models.items() if m["available"]} == {
        "dino", "siglip", "fusion"}
    assert models["stub"]["available"] is False
    assert {n for n, m in models.items() if m["is_default"]} == {"fusion"}

    # Только DINOv2 -> fusion не собран.
    cfg2 = AppConfig(engine="infer", device="cuda", variant="dino",
                     dino_ckpt=str(dino))
    engine2 = build_engine(cfg2)
    models2 = {m["name"]: m for m in describe_models(cfg2, engine2)}
    assert {n for n, m in models2.items() if m["available"]} == {"dino"}
    assert {n for n, m in models2.items() if m["is_default"]} == {"dino"}


def test_siglip_threshold_documented_default():
    # Явный override побеждает; без него — конечный положительный порог
    # (reid.calibrate SigLIP2-константу пока не экспортирует).
    engine = CpuSiglipEngine(SIGLIP_ONNX, threshold=None)
    assert engine.threshold > 0.0
    assert engine.threshold == CPU_SIGLIP_DEFAULT_THRESHOLD or \
        engine.threshold > 0.0


def test_search_503_when_gallery_file_missing():
    cfg = AppConfig(engine="stub", dim=DIM,
                    gallery_embeddings=os.path.join(REPO, "no_such_gallery.npy"))
    app = create_app(cfg=cfg)  # не должно бросить
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        r = client.post("/api/v1/search", json=_query_payload())
        assert r.status_code == 503, r.text
        assert "галере" in r.json()["detail"].lower()


def test_search_503_when_gallery_not_configured():
    app = create_app(cfg=AppConfig(engine="stub", dim=DIM))  # нет REID_API_GALLERY
    with TestClient(app) as client:
        r = client.post("/api/v1/search", json=_query_payload())
        assert r.status_code == 503
        assert r.json()["detail"]


def test_search_503_on_gallery_dim_mismatch():
    gallery = NumpyGalleryIndex(np.zeros((3, 99), np.float32),
                                ["g0", "g1", "g2"])
    app = create_app(cfg=AppConfig(engine="stub", dim=DIM),
                     engine=StubEngine(dim=DIM), gallery=gallery)
    with TestClient(app) as client:
        r = client.post("/api/v1/search", json=_query_payload())
        assert r.status_code == 503
        assert "размерност" in r.json()["detail"].lower()


def test_stub_search_still_works_with_explicit_gallery():
    engine = StubEngine(dim=DIM, seed=7, threshold=0.0)
    ids, embs = [], []
    for i in range(6):
        _, decoded = _jpeg_b64(_make_image(100 + i))
        ids.append(f"g{i:02d}")
        embs.append(engine.embed(decoded, (10, 10, 30, 30)))
    gallery = NumpyGalleryIndex(np.stack(embs), ids)
    app = create_app(cfg=AppConfig(engine="stub", dim=DIM), engine=engine,
                     gallery=gallery)
    with TestClient(app) as client:
        assert client.get("/health").json()["gallery_size"] == 6
        assert client.get("/api/v1/models").status_code == 200
        r = client.post("/api/v1/search", json=_query_payload(seed=102))
        assert r.status_code == 200
        assert len(r.json()["candidates"]) > 0


# ---------------------------------------------------------------------------
# Реальный CPU-инференс (skip без весов)
# ---------------------------------------------------------------------------
@_needs_weights
def test_cpu_engine_real_inference_and_refusal():
    engine = CpuSiglipEngine(SIGLIP_ONNX, threads=2, batch_size=1, threshold=0.0)
    vec = engine.embed(_make_image(1), (0, 0, 48, 48))
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3

    # разные кадры -> разные вектора (не хардкод)
    vec2 = engine.embed(_make_image(2), (0, 0, 48, 48))
    assert not np.allclose(vec, vec2)

    # API: галерея из vec2, запрос — кадр 1; недостижимый порог -> отказ
    gallery = NumpyGalleryIndex(np.stack([vec2]), ["g00"])
    engine.threshold = 1.01
    app = create_app(cfg=AppConfig(engine="infer", device="cpu", dim=512),
                     engine=engine, gallery=gallery)
    with TestClient(app) as client:
        payload = _query_payload(seed=1)
        payload["top_k"] = 1
        r = client.post("/api/v1/search", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "siglip"
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["gallery_id"] == "g00"
        assert np.isfinite(body["confidence"])
        assert body["refused"] is True
        assert body["accepted"] is False
        assert body["threshold"] == 1.01


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
