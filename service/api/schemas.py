"""Pydantic-схемы HTTP-контрактов API (авто-OpenAPI/Swagger).

Все запросы/ответы описаны моделями pydantic v2, поэтому ``/openapi.json`` и
``/docs`` генерируются без ручной разметки. Изображение передаётся строкой
base64 (опционально с ``data:image/...;base64,`` префиксом) — это не требует
``python-multipart`` и удобно тонкому браузерному клиенту.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Общие типы
# ---------------------------------------------------------------------------
ImageB64 = Annotated[
    str,
    Field(
        min_length=1,
        description="Изображение кадра в base64 (JPEG/PNG/BMP/WEBP); "
        "допускается префикс `data:image/png;base64,`.",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    ),
]


class BBox(BaseModel):
    """BBox в пикселях исходного кадра: левый верх (x, y) + размер (w, h)."""

    x: int = Field(ge=0, description="Левый край, пиксели исходного кадра.")
    y: int = Field(ge=0, description="Верхний край, пиксели исходного кадра.")
    w: int = Field(ge=1, description="Ширина, пиксели (>=1).")
    h: int = Field(ge=1, description="Высота, пиксели (>=1).")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    engine: str
    gallery_size: int
    gallery_dim: int | None = None
    models_available: int


# ---------------------------------------------------------------------------
# /api/v1/models
# ---------------------------------------------------------------------------
class ModelInfo(BaseModel):
    name: str
    variant: Literal["stub", "dino", "siglip", "fusion"]
    dim: int
    description: str
    available: bool
    is_default: bool
    weights: str | None = None


class ModelsResponse(BaseModel):
    default: str
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# /api/v1/search
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    image_base64: ImageB64
    bbox: BBox
    top_k: int = Field(default=10, ge=1, le=100,
                       description="Сколько ближайших gallery-объектов вернуть.")
    model: str | None = Field(default=None,
                              description="Имя модели (по умолчанию — модели "
                              "деплоя). Демо-стенд: `stub`.")


class Candidate(BaseModel):
    rank: int = Field(ge=1)
    gallery_id: str
    score: float = Field(description="Косинусная близость к запросу.")
    confidence: float = Field(
        description="Кросс-запросно сравнимая уверенность (cosine top-1).")
    image_url: str | None = Field(
        default=None,
        description="URL превью кадра галереи "
        "(`GET /api/v1/gallery/{gallery_id}/image`).")


class SearchResponse(BaseModel):
    query_id: str | None = None
    model: str
    bbox: BBox
    top_k: int
    candidates: list[Candidate]
    confidence: float = Field(description="Уверенность top-1.")
    threshold: float = Field(description="Замороженный порог отказа.")
    accepted: bool = Field(description="True — кандидат принят; False — отказ.")
    refused: bool = Field(description="Флаг отказа = not accepted.")
    latency_ms: float


# ---------------------------------------------------------------------------
# /api/v1/explain
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    image_base64: ImageB64
    bbox: BBox
    model: str | None = None


class ExplainResponse(BaseModel):
    model: str
    bbox: BBox
    method: str = Field(description="Как построена карта (метод/заглушка).")
    heatmap_png_base64: str = Field(description="Карта значимости (data URL PNG).")
    overlay_png_base64: str = Field(
        description="Карта, наложенная на кроп запроса (data URL PNG).")


# ---------------------------------------------------------------------------
# /api/v1/jobs
# ---------------------------------------------------------------------------
class JobCreated(BaseModel):
    job_id: str
    status: Literal["pending"] = "pending"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    result: SearchResponse | None = None
    error: str | None = None


__all__ = [
    "BBox",
    "Candidate",
    "ExplainRequest",
    "ExplainResponse",
    "HealthResponse",
    "JobCreated",
    "JobStatus",
    "ModelInfo",
    "ModelsResponse",
    "SearchRequest",
    "SearchResponse",
]
