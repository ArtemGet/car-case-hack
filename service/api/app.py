"""FastAPI-приложение (авто-OpenAPI/Swagger).

Создаётся фабрикой :func:`create_app` (для тестов удобно подменить движок и
галерею). Модуль экспортирует ``app`` для ``uvicorn service.api.app:app``.

Эндпоинты:
    GET  /health
    GET  /api/v1/models
    POST /api/v1/search
    POST /api/v1/explain
    POST /api/v1/jobs
    GET  /api/v1/jobs/{job_id}
    GET  /api/v1/gallery/{gallery_id}/image

Фасад ничего не считает про модель: декодирует/валидирует вход, зовёт движок
(заглушку, CPU-SigLIP2 или ``service.infer``) и статическую галерею, применяет
порог отказа. Если готовая галерея отсутствует, ``/search`` возвращает 503.
"""
from __future__ import annotations

import base64
import binascii
import io
import os
import re
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from .config import AppConfig, REPO_ROOT
from .engine import InferenceEngine, build_engine, describe_models
from .gallery import GalleryIndex, NumpyGalleryIndex, load_gallery
from .schemas import (
    BBox,
    Candidate,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    JobCreated,
    JobStatus,
    ModelsResponse,
    SearchRequest,
    SearchResponse,
)

_ACCEPTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}

# gallery_id — basename без разделителей пути: буквы/цифры/._- и без "..".
_GALLERY_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


# ---------------------------------------------------------------------------
# Валидация входа
# ---------------------------------------------------------------------------
def decode_image_b64(data: str) -> Image.Image:
    """base64 (опц. data URL) -> RGB PIL.Image. 400 при любой проблеме."""
    if not isinstance(data, str) or not data.strip():
        raise HTTPException(status_code=400, detail="пустое изображение")
    if data.startswith("data:"):
        try:
            data = data.split(",", 1)[1]
        except IndexError:
            raise HTTPException(status_code=400, detail="битый data URL изображения")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="некорректная base64-строка")
    if not raw:
        raise HTTPException(status_code=400, detail="пустое изображение")
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format
        img.load()
    except Exception:  # noqa: BLE001 — PIL бросает разные исключения
        raise HTTPException(status_code=400,
                            detail="не удалось декодировать изображение")
    if fmt is not None and fmt.upper() not in _ACCEPTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"формат {fmt!r} не поддержан (нужен JPEG/PNG/BMP/WEBP)")
    return img.convert("RGB")


def validate_bbox(bbox: BBox, image_size: tuple[int, int]) -> None:
    """bbox внутри кадра (иначе 422). w,h>=1 и x,y>=0 уже проверил pydantic."""
    width, height = image_size
    if bbox.x + bbox.w > width or bbox.y + bbox.h > height:
        raise HTTPException(
            status_code=422,
            detail=(f"bbox выходит за границы кадра {width}x{height}: "
                    f"x={bbox.x},y={bbox.y},w={bbox.w},h={bbox.h}"))


# ---------------------------------------------------------------------------
# Визуализация explain
# ---------------------------------------------------------------------------
_ANCHORS = [(0.0, (0, 0, 128)), (0.25, (0, 180, 255)), (0.5, (0, 255, 0)),
            (0.75, (255, 255, 0)), (1.0, (255, 0, 0))]


def colorize(heat: np.ndarray) -> np.ndarray:
    """0..1 -> RGB uint8 по сине-красной шкале (без matplotlib)."""
    h = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    xs = np.array([a[0] for a in _ANCHORS], dtype=np.float32)
    rgb = np.stack([np.interp(h, xs, [a[1][c] for a in _ANCHORS])
                    for c in range(3)], axis=-1)
    return rgb.astype(np.uint8)


def _png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_explain_images(image: Image.Image, bbox: BBox, heat: np.ndarray
                        ) -> tuple[str, str]:
    x, y, w, h = bbox.as_tuple()
    crop = image.crop((x, y, x + w, y + h)).convert("RGB")
    hmap = np.asarray(heat, dtype=np.float32)
    if hmap.shape != (crop.height, crop.width):
        hmap = np.asarray(Image.fromarray(hmap).resize(crop.size, Image.BILINEAR))
    color = colorize(hmap)
    base = np.asarray(crop, dtype=np.float32)
    overlay = (0.55 * base + 0.45 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return _png_data_url(color), _png_data_url(overlay)


# ---------------------------------------------------------------------------
# Оркестрация поиска
# ---------------------------------------------------------------------------
def run_search(engine: InferenceEngine, gallery: GalleryIndex,
               req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    image = decode_image_b64(req.image_base64)
    validate_bbox(req.bbox, image.size)
    vec = engine.embed(image, req.bbox.as_tuple())
    ids, scores = gallery.search(vec, top_k=req.top_k)

    if len(scores):
        conf = float(scores[0])
    else:
        conf = 0.0
    # Решение отказа — домен reid.calibrate (единый источник правила).
    try:
        from reid.calibrate import is_accepted

        accepted = bool(is_accepted(conf, engine.threshold)) and len(scores) > 0
    except Exception:  # noqa: BLE001 — модуль недоступен: то же правило явно
        accepted = bool(len(scores) > 0 and conf >= engine.threshold)

    candidates = [
        Candidate(rank=i + 1, gallery_id=gid, score=float(sc), confidence=float(sc),
                  image_url=f"/api/v1/gallery/{gid}/image")
        for i, (gid, sc) in enumerate(zip(ids, scores))
    ]
    return SearchResponse(
        query_id=None,
        model=engine.name,
        bbox=req.bbox,
        top_k=int(req.top_k),
        candidates=candidates,
        confidence=conf,
        threshold=float(engine.threshold),
        accepted=accepted,
        refused=not accepted,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Доступность галереи
# ---------------------------------------------------------------------------
def gallery_unavailable(app: FastAPI) -> str | None:
    """Причина, по которой ``/search`` не может работать (HTTP 503).

    ``None`` — галерея готова. Иначе — понятное сообщение: файл эмбеддингов не
    загружен/не задан, галерея пуста или её размерность не совпадает с движком.
    """
    err = getattr(app.state, "gallery_error", None)
    if err:
        return str(err)
    if not len(app.state.gallery.ids):
        return ("галерея пуста: нет готовых эмбеддингов "
                "(REID_API_GALLERY / REID_API_GALLERY_IDS)")
    return None


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
def create_app(cfg: AppConfig | None = None, engine: InferenceEngine | None = None,
               gallery: GalleryIndex | None = None) -> FastAPI:
    cfg = cfg or AppConfig.from_env()

    if engine is None:
        engine = build_engine(cfg)

    # Галерея статична и строится заранее. Отсутствие/битость файлов НЕ должно
    # валить приложение: /health и /models остаются доступны, а /search вернёт
    # понятную 503 (см. gallery_unavailable). Явно переданная галерея (тесты/UI)
    # файлы не читает, но её размерность всё равно должна совпасть с движком.
    gallery_error: str | None = None
    if gallery is None:
        if cfg.gallery_embeddings and cfg.gallery_embeddings.lower() != "none":
            try:
                gallery = load_gallery(cfg.gallery_embeddings, cfg.gallery_ids,
                                       backend=cfg.gallery_backend)
            except Exception as exc:  # noqa: BLE001 — файл отсутствует/битый
                gallery = NumpyGalleryIndex(
                    np.zeros((0, engine.dim), np.float32), [])
                gallery_error = (f"не удалось загрузить галерею "
                                 f"{cfg.gallery_embeddings!r}: "
                                 f"{type(exc).__name__}: {exc}")
        else:
            gallery = NumpyGalleryIndex(np.zeros((0, engine.dim), np.float32), [])
            gallery_error = ("галерея не сконфигурирована: задайте "
                             "REID_API_GALLERY=<embeddings.npy> и "
                             "REID_API_GALLERY_IDS=<ids.json>")
    if (gallery_error is None and len(gallery.ids)
            and gallery.dim != engine.dim):
        gallery_error = (f"размерность галереи {gallery.dim} != размерности "
                         f"движка {engine.dim} ({engine.name})")

    app = FastAPI(
        title=cfg.title,
        version=cfg.version,
        description=("Тонкий фасад Vehicle ReID: поиск по кропу (BBox), "
                     "confidence и режим отказа, объяснение значимости. "
                     "Логика модели — в service.infer."),
        openapi_tags=[
            {"name": "system", "description": "Служебные эндпоинты."},
            {"name": "reid", "description": "Поиск, модели, объяснение."},
            {"name": "jobs", "description": "Асинхронные задачи (скелет)."},
        ],
    )
    app.state.cfg = cfg
    app.state.engine = engine
    app.state.gallery = gallery
    app.state.gallery_error = gallery_error
    app.state.jobs: dict[str, dict] = {}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        models = describe_models(app.state.cfg, app.state.engine)
        return HealthResponse(
            version=app.state.cfg.version,
            engine=app.state.engine.name,
            gallery_size=len(app.state.gallery.ids),
            gallery_dim=(app.state.gallery.dim if len(app.state.gallery.ids) else None),
            models_available=int(sum(1 for m in models if m["available"])),
        )

    @app.get("/api/v1/models", response_model=ModelsResponse, tags=["reid"])
    def models() -> ModelsResponse:
        return ModelsResponse(default=app.state.engine.name,
                              models=describe_models(app.state.cfg, app.state.engine))

    @app.get(
        "/api/v1/gallery/{gallery_id}/image",
        tags=["reid"],
        responses={
            200: {"content": {"image/jpeg": {}},
                  "description": "JPEG-кадр галереи."},
            404: {"description": "Неизвестный gallery_id / нет файла / "
                                 "каталог изображений не задан."},
        },
    )
    def gallery_image(gallery_id: str) -> FileResponse:
        """Отдать JPEG-кадр ``<gallery_id>.jpg`` для превью кандидата.

        Отдаём только для id, реально присутствующих в статической галерее;
        любой небезопасный id (разделители пути, ``..``) — 404. Каталог
        изображений задаётся ``REID_API_IMAGES_DIR`` (см. ``config.py``).
        """
        if (not _GALLERY_ID_RE.fullmatch(gallery_id)
                or ".." in gallery_id):
            raise HTTPException(status_code=404, detail="изображение не найдено")
        if gallery_id not in app.state.gallery.ids:
            raise HTTPException(status_code=404,
                                detail=f"gallery_id {gallery_id!r} не найден")
        images_dir = app.state.cfg.images_dir
        if not images_dir:
            raise HTTPException(
                status_code=404,
                detail="каталог изображений галереи не задан "
                       "(REID_API_IMAGES_DIR)")
        path = os.path.join(images_dir, f"{gallery_id}.jpg")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404,
                                detail=f"файл изображения не найден: "
                                       f"{gallery_id}.jpg")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/v1/search", response_model=SearchResponse, tags=["reid"])
    def search(req: SearchRequest) -> SearchResponse:
        missing = gallery_unavailable(app)
        if missing:
            raise HTTPException(status_code=503, detail=missing)
        try:
            return run_search(app.state.engine, app.state.gallery, req)
        except HTTPException:
            raise
        except RuntimeError as exc:  # движок недоступен (нет весов и т.п.)
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"inference error: {exc}")

    @app.post("/api/v1/explain", response_model=ExplainResponse, tags=["reid"])
    def explain(req: ExplainRequest) -> ExplainResponse:
        image = decode_image_b64(req.image_base64)
        validate_bbox(req.bbox, image.size)
        try:
            heat, method = app.state.engine.explain(image, req.bbox.as_tuple())
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"explain error: {exc}")
        heat_png, overlay_png = make_explain_images(image, req.bbox, heat)
        return ExplainResponse(model=app.state.engine.name, bbox=req.bbox,
                               method=method, heatmap_png_base64=heat_png,
                               overlay_png_base64=overlay_png)

    @app.post("/api/v1/jobs", response_model=JobCreated, status_code=202,
              tags=["jobs"])
    def create_job(req: SearchRequest) -> JobCreated:
        missing = gallery_unavailable(app)
        if missing:
            raise HTTPException(status_code=503, detail=missing)
        job_id = uuid.uuid4().hex
        app.state.jobs[job_id] = {"status": "running", "result": None,
                                  "error": None}
        try:
            app.state.jobs[job_id]["result"] = run_search(
                app.state.engine, app.state.gallery, req)
            app.state.jobs[job_id]["status"] = "done"
        except HTTPException as exc:
            app.state.jobs[job_id].update(status="error", error=str(exc.detail))
        except Exception as exc:  # noqa: BLE001
            app.state.jobs[job_id].update(status="error", error=str(exc))
        return JobCreated(job_id=job_id)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
    def job_status(job_id: str) -> JobStatus:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatus(job_id=job_id, status=job["status"],
                         result=job["result"], error=job["error"])

    # Serve the built demo UI (web/dist) from the SAME origin, so the browser
    # talks to /api and /health directly — the prototype needs no separate
    # proxy (vite/nginx) and stays fully offline. Mounted last so the explicit
    # API routes above win.
    static_dir = os.path.join(REPO_ROOT, "web", "dist")
    if os.path.isdir(static_dir):
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")

    return app


# ASGI-вход для uvicorn (движок/галерея -- из env; по умолчанию stub)
app = create_app()


__all__ = ["app", "create_app", "decode_image_b64", "validate_bbox",
           "make_explain_images", "colorize", "run_search",
           "gallery_unavailable"]
