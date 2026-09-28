"""Конфигурация API из переменных окружения (все опциональны, дефолты — демо).

Демо-стенд по умолчанию поднимается на заглушке (``stub``) без весов и без GPU —
эндпоинты и Swagger работают сразу.

Реальные движки
---------------
* ``REID_API_ENGINE=infer`` + ``REID_API_DEVICE=cpu`` → ``CpuSiglipEngine``:
  SigLIP2 ONNX на ``CPUExecutionProvider`` через ``service.infer.cpu_backend``
  (PIL-препроцесс, без torch, RSS ~0.4 ГБ). CPU-прототип умеет только SigLIP2,
  поэтому вариант принудительно ``siglip``.
* ``REID_API_ENGINE=infer`` + ``REID_API_DEVICE=cuda|auto`` → ``InferEngine``
  (production GPU-путь: torch DINOv2 и/или ONNX SigLIP2/CUDA) — не изменён.

Готовая статическая галерея (строится заранее, эмбеддинги SigLIP2 CPU)
----------------------------------------------------------------------
* ``REID_API_GALLERY``     — путь к ``*.npy`` ``(Ng, D)`` float32;
* ``REID_API_GALLERY_IDS`` — путь к ``*_ids.json`` (по умолчанию ``<gallery>_ids.json``;
  JSON-список id или ``{"ids": [...], "dim": D}``);
* ``REID_API_GALLERY_BACKEND`` — ``auto|numpy|faiss``;
* ``REID_API_IMAGES_DIR``   — каталог JPEG-кадров ``<gallery_id>.jpg`` для
  превью (``GET /api/v1/gallery/{gallery_id}/image``). Дефолт —
  ``<repo>/docs/Датасет/dataset/images``, **только если каталог существует**,
  иначе превью отключены (404).

Если файл галереи отсутствует/не читается, приложение стартует, ``/health`` и
``/api/v1/models`` доступны, а ``/api/v1/search`` возвращает понятную 503.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Каталог с JPEG-кадрами датасета (для превью галереи). Дефолт — только если
# он реально существует в этом чекауте; иначе None (эндпоинт отдаёт 404).
DEFAULT_IMAGES_DIR = os.path.join(REPO_ROOT, "docs", "Датасет", "dataset", "images")


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _abs(path: str | None) -> str | None:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _resolve_images_dir() -> str | None:
    """Каталог кадров галереи: ``REID_API_IMAGES_DIR``, иначе дефолт, если есть.

    Если переменная задана явно (даже несуществующим путём) — возвращаем её
    (эндпоинт честно вернёт 404 на отсутствующий файл). Иначе — только
    существующий дефолтный каталог, иначе ``None``.
    """
    env = os.environ.get("REID_API_IMAGES_DIR")
    if env not in (None, ""):
        return _abs(env)
    if os.path.isdir(DEFAULT_IMAGES_DIR):
        return DEFAULT_IMAGES_DIR
    return None


@dataclass
class AppConfig:
    """Настройки фасада. ``engine``: ``stub`` | ``infer``."""

    title: str = "Vehicle ReID API"
    version: str = "0.1.0"
    engine: str = "stub"
    variant: str = "fusion"
    device: str = "cpu"  # GPU не занят фасадом по умолчанию
    dim: int = 512

    # заглушка
    stub_seed: int = 42
    stub_threshold: float = 0.0

    # реальный инференс (локальные веса, GPU-путь)
    dino_ckpt: str | None = field(default=None)
    siglip_onnx: str | None = field(default=None)
    batch_size: int = 64
    siglip_threads: int = 8

    # CPU-прототип SigLIP2 (service.infer.cpu_backend, ONNX CPU EP, без torch)
    cpu_siglip_onnx: str | None = field(default=None)
    cpu_max_patches: int = 256
    cpu_max_ram_mb: int = 0        # 0 = не проверять RSS
    cpu_batch_size: int = 1        # фасад эмбеддит по одному кадру

    # порог отказа: явный override (иначе — resolve_siglip_threshold)
    threshold: float | None = None

    # статическая галерея (готовая, построена заранее)
    gallery_embeddings: str | None = None
    gallery_ids: str | None = None
    gallery_backend: str = "auto"
    # каталог JPEG-кадров галереи для превью (GET /api/v1/gallery/{id}/image)
    images_dir: str | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            version=os.environ.get("REID_API_VERSION", "0.1.0"),
            engine=os.environ.get("REID_API_ENGINE", "stub").strip().lower(),
            variant=os.environ.get("REID_API_VARIANT", "fusion").strip().lower(),
            device=os.environ.get("REID_API_DEVICE", "cpu").strip().lower(),
            dim=_env_int("REID_API_DIM", 512),
            stub_seed=_env_int("REID_API_STUB_SEED", 42),
            stub_threshold=_env_float("REID_API_STUB_THRESHOLD", 0.0),
            dino_ckpt=_abs(os.environ.get("REID_API_DINO_CKPT", "runs/exp-0007/best.pt")),
            siglip_onnx=_abs(os.environ.get(
                "REID_API_SIGLIP_ONNX",
                "runs/external/vehicle_reid_siglip2_naflex_512d.onnx")),
            batch_size=_env_int("REID_API_BATCH_SIZE", 64),
            siglip_threads=_env_int("REID_API_SIGLIP_THREADS", 8),
            cpu_siglip_onnx=_abs(os.environ.get(
                "REID_API_CPU_SIGLIP_ONNX",
                os.path.join("artifacts", "siglip2_fp16.onnx"))),
            cpu_max_patches=_env_int("REID_API_CPU_MAX_PATCHES", 256),
            cpu_max_ram_mb=_env_int("REID_API_CPU_MAX_RAM_MB", 0),
            cpu_batch_size=_env_int("REID_API_CPU_BATCH_SIZE", 1),
            threshold=(float(os.environ["REID_API_THRESHOLD"])
                       if os.environ.get("REID_API_THRESHOLD") not in (None, "")
                       else None),
            gallery_embeddings=_abs(os.environ.get("REID_API_GALLERY")),
            gallery_ids=_abs(os.environ.get("REID_API_GALLERY_IDS")),
            gallery_backend=os.environ.get("REID_API_GALLERY_BACKEND", "auto").strip().lower(),
            images_dir=_resolve_images_dir(),
        )


__all__ = ["AppConfig", "REPO_ROOT"]
