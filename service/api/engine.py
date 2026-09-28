"""Движки инференса фасада — тонкая оркестрация, без собственной логики модели.

Два движка за одним протоколом :class:`InferenceEngine`:

* :class:`StubEngine` — детерминированная заглушка (без весов/GPU) для демо,
  тестов и UI-разработки. Эмбеддинг вычисляется из пикселей кропа; результаты
  НЕ хардкожены.
* :class:`InferEngine` — реальный пайплайн. Ничего не считает сам: сохраняет
  присланный кадр во временный каталог, строит однострочный DataFrame и
  вызывает ``service.infer.backends`` (``DinoBackend``/``SiglipBackend``),
  фьюжн — через ``reid.rerank``, порог — из ``service.infer.run``. Логика
  эмбеддинга не дублируется.
* :class:`CpuSiglipEngine` — реальный CPU-путь: тот же приём оркестрации, но
  бэкенд — ``service.infer.cpu_backend.SiglipCpuBackend`` (SigLIP2 ONNX на
  ``CPUExecutionProvider``, PIL-препроцесс, **без torch**, RSS ~0.4 ГБ). Только
  SigLIP2; DINOv2/фьюжн на CPU не грузятся.

GPU по умолчанию не используется (``device="cpu"``); реальные движки грузят
веса лениво, при первом запросе.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image

from .config import REPO_ROOT, AppConfig

_GRAY = 64


# ---------------------------------------------------------------------------
# Общий детерминированный explain-фолбэк (W2-6 заменит на настоящий Grad-CAM)
# ---------------------------------------------------------------------------
def gradient_heatmap(image: Image.Image, bbox: tuple[int, int, int, int]
                     ) -> tuple[np.ndarray, str]:
    """Карта значимости из градиента яркости кропа (реальный расчёт, не хардкод).

    Это честная заглушка: полноценный Grad-CAM по активациям модели — задача
    W2-6; пока endpoint возвращает карту, зависящую от пикселей запроса.
    """
    x, y, w, h = bbox
    crop = image.crop((x, y, x + w, y + h)).convert("L")
    arr = np.asarray(crop, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.float32), "input-gradient"
    gy, gx = np.gradient(arr)
    mag = np.hypot(gx, gy)
    mx = float(mag.max())
    if mx > 1e-9:
        mag = mag / mx
    return mag.astype(np.float32), "input-gradient"


# ---------------------------------------------------------------------------
# Протокол
# ---------------------------------------------------------------------------
@runtime_checkable
class InferenceEngine(Protocol):
    name: str
    variant: str
    dim: int
    threshold: float

    @property
    def available(self) -> bool:
        ...

    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Один L2-нормированный вектор ``(dim,)`` float32 для кропа."""
        ...

    def explain(self, image: Image.Image, bbox: tuple[int, int, int, int]
                ) -> tuple[np.ndarray, str]:
        ...


# ---------------------------------------------------------------------------
# Заглушка (демо/тесты)
# ---------------------------------------------------------------------------
class StubEngine:
    """Детерминированный не-модельный эмбеддинг + gradient-explain.

    Вектор = фиксированная псевдослучайная проекция 8x8 grayscale-подписи
    кропа, L2-нормированная. Одинаковый кроп -> одинаковый вектор; разные
    кадры -> разные вектора. Никаких выученных весов.
    """

    variant = "stub"

    def __init__(self, dim: int = 512, seed: int = 42, threshold: float = 0.0,
                 name: str = "stub"):
        self.dim = int(dim)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.name = name
        rng = np.random.default_rng(self.seed)
        self._proj = rng.standard_normal((self.dim, _GRAY * _GRAY)).astype(np.float32)

    @property
    def available(self) -> bool:
        return True

    def _signature(self, image: Image.Image, bbox: tuple[int, int, int, int]
                   ) -> np.ndarray:
        x, y, w, h = bbox
        crop = image.crop((x, y, x + w, y + h)).convert("L").resize((_GRAY, _GRAY))
        feat = np.asarray(crop, dtype=np.float32).reshape(-1) / 255.0
        feat = feat - float(feat.mean())
        return feat

    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        feat = self._signature(image, bbox)
        vec = self._proj @ feat
        n = float(np.linalg.norm(vec))
        if n < 1e-12:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec
        return (vec / n).astype(np.float32)

    def explain(self, image: Image.Image, bbox: tuple[int, int, int, int]
                ) -> tuple[np.ndarray, str]:
        return gradient_heatmap(image, bbox)


# ---------------------------------------------------------------------------
# Реальный движок: тонкая обёртка над service.infer.backends
# ---------------------------------------------------------------------------
FUSION_W = 0.6  # вес DINOv2 в fusion-конкатенации (как в service.infer.run)
_DIMS = {"dino": 512, "siglip": 512, "fusion": 1024}


class InferEngine:
    """Оркестрация реальных бэкендов. Веса грузятся лениво при первом embed."""

    def __init__(self, variant: str = "fusion", device: str = "cpu",
                 dino_ckpt: str | None = None, siglip_onnx: str | None = None,
                 batch_size: int = 64, siglip_threads: int = 8):
        if variant not in _DIMS:
            raise ValueError(f"unknown variant: {variant!r}")
        self.variant = variant
        self.name = variant
        self.device = device
        self.dino_ckpt = dino_ckpt
        self.siglip_onnx = siglip_onnx
        self.batch_size = int(batch_size)
        self.siglip_threads = int(siglip_threads)
        self.dim = _DIMS[variant]
        self.threshold = self._resolve_threshold()
        self._dino = None
        self._sig = None

    # -- availability / config ------------------------------------------------
    def _needs(self) -> list[str]:
        need = []
        if self.variant in ("dino", "fusion"):
            need.append(self.dino_ckpt or "")
        if self.variant in ("siglip", "fusion"):
            need.append(self.siglip_onnx or "")
        return [p for p in need]

    @property
    def available(self) -> bool:
        return all(p and os.path.exists(p) for p in self._needs())

    def _resolve_threshold(self) -> float:
        try:
            from service.infer.run import DEFAULT_THRESHOLDS

            return float(DEFAULT_THRESHOLDS[self.variant])
        except Exception:  # noqa: BLE001 — веса/конфиг недоступны: безопасный дефолт
            return 1.0  # всё отвергается, пока порог не задан явно

    # -- lazy backends --------------------------------------------------------
    def _ensure(self) -> None:
        if not self.available:
            raise RuntimeError(
                f"движок {self.variant!r} недоступен: нет весов {self._needs()}")
        from service.infer.backends import DinoBackend, SiglipBackend

        if self.variant in ("dino", "fusion") and self._dino is None:
            self._dino = DinoBackend(self.dino_ckpt, self._resolve_device(),
                                     amp="bf16", batch_size=self.batch_size,
                                     num_workers=0)
        if self.variant in ("siglip", "fusion") and self._sig is None:
            self._sig = SiglipBackend(self.siglip_onnx, threads=self.siglip_threads)

    def _resolve_device(self):
        from service.infer.backends import resolve_device

        return resolve_device(self.device)

    # -- public API -----------------------------------------------------------
    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        self._ensure()
        import pandas as pd

        from reid import rerank

        x, y, w, h = bbox
        with tempfile.TemporaryDirectory(prefix="reid_api_") as tmp:
            image.convert("RGB").save(os.path.join(tmp, "query.jpg"),
                                      "JPEG", quality=95)
            df = pd.DataFrame([{"image_id": "query", "x": int(x), "y": int(y),
                                "w": int(w), "h": int(h)}])
            if self.variant == "dino":
                vec = self._dino.extract(df, tmp, 224)[0]
            elif self.variant == "siglip":
                vec = self._sig.extract(df, tmp)[0]
            else:
                q_sig = self._sig.extract(df, tmp)[0]
                q_dino = self._dino.extract(df, tmp, 224)[0]
                vec = rerank.l2norm(np.concatenate(
                    [q_sig, FUSION_W * q_dino]).reshape(1, -1))[0]
        return np.asarray(vec, dtype=np.float32)

    def explain(self, image: Image.Image, bbox: tuple[int, int, int, int]
                ) -> tuple[np.ndarray, str]:
        # Полноценный Grad-CAM по активациям — W2-6. Пока — честный фолбэк.
        heat, _ = gradient_heatmap(image, bbox)
        return heat, "input-gradient (fallback; Grad-CAM — W2-6)"


# ---------------------------------------------------------------------------
# CPU-прототип: SigLIP2 ONNX (CPU EP) — реальный инференс без torch
# ---------------------------------------------------------------------------
CPU_SIGLIP_DIM = 512

# Задокументированный дефолт порога отказа для SigLIP2, если ни
# ``reid.calibrate``, ни ``service.infer.run`` не отдают значение для варианта
# ``siglip``. Равен service.infer.run.DEFAULT_THRESHOLDS["siglip"] — порог,
# снятый на замороженном val-сплите (seed=42) как точка max(0.7*F1+0.3*TNR).
#
# ВАЖНО: это шкала GPU-препроцесса SigLIP2 (letterbox/patch-бюджет
# backends.SiglipBackend). CPU-прототип использует тот же ONNX-граф, но другой
# PIL-препроцесс (service.infer.cpu_backend, patch-budget max_patches), поэтому
# сама cosine-шкала может слегка сместиться; отдельной CPU-калибровки пока нет.
# Пока её нет, порог берётся здесь и ЯВНО документирован; переопределяется
# REID_API_THRESHOLD / аргументом конструктора без правки кода.
CPU_SIGLIP_DEFAULT_THRESHOLD = 0.8943735957145691


def resolve_siglip_threshold(override: float | None = None) -> float:
    """Порог отказа для SigLIP2. Порядок (первое попадание):

    1. явный ``override`` (конструктор/``REID_API_THRESHOLD``);
    2. env ``REID_SIGLIP_THRESHOLD`` (или ``REID_API_THRESHOLD``);
    3. ``reports/calibration_siglip.json`` → ``chosen.threshold`` (хэндофф
       ``reid.calibrate``);
    4. константа ``reid.calibrate.DEFAULT_THRESHOLD_SIGLIP`` (или ``_CPU``),
       если calibration-agent её заведёт;
    5. ``service.infer.run.DEFAULT_THRESHOLDS["siglip"]``;
    6. :data:`CPU_SIGLIP_DEFAULT_THRESHOLD` (задокументированный дефолт).

    ``reid.calibrate`` сегодня не экспортирует SigLIP2-константу, поэтому
    фактически используется п.5/п.6 — обе величины совпадают.
    """
    if override is not None:
        return float(override)
    env = (os.environ.get("REID_SIGLIP_THRESHOLD")
           or os.environ.get("REID_API_THRESHOLD"))
    if env:
        return float(env)
    cal = os.path.join(REPO_ROOT, "reports", "calibration_siglip.json")
    if os.path.exists(cal):
        try:
            with open(cal, "r", encoding="utf-8") as f:
                thr = json.load(f).get("chosen", {}).get("threshold")
            if thr is not None:
                return float(thr)
        except Exception:  # noqa: BLE001 — битый отчёт: идём к дефолту
            pass
    try:  # единый источник калибровки (если добавят SigLIP2-константу)
        from reid import calibrate

        for attr in ("DEFAULT_THRESHOLD_SIGLIP", "DEFAULT_THRESHOLD_SIGLIP_CPU"):
            v = getattr(calibrate, attr, None)
            if v is not None:
                return float(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        from service.infer.run import DEFAULT_THRESHOLDS

        v = DEFAULT_THRESHOLDS.get("siglip")
        if v is not None:
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    return float(CPU_SIGLIP_DEFAULT_THRESHOLD)


class CpuSiglipEngine:
    """SigLIP2 ONNX на CPU (``service.infer.cpu_backend``), без torch.

    Это оркестрация, а не логика модели: присланный кадр сохраняется во
    временный JPEG, из него строится однострочный DataFrame (image_id + bbox) и
    вызывается ``SiglipCpuBackend.extract``. Собственного эмбеддинга здесь нет.
    Один L2-нормированный вектор ``(512,)`` float32 на кроп.
    """

    variant = "siglip"
    device = "cpu"

    def __init__(self, onnx_path: str | None = None, *, threads: int = 8,
                 max_patches: int = 256, max_ram_mb: int = 0,
                 batch_size: int = 1, threshold: float | None = None,
                 name: str = "siglip"):
        self.onnx_path = onnx_path
        self.threads = int(threads)
        self.max_patches = int(max_patches)
        self.max_ram_mb = int(max_ram_mb)
        self.batch_size = int(batch_size)
        self.name = name
        self.dim = CPU_SIGLIP_DIM
        self.threshold = resolve_siglip_threshold(threshold)
        self._backend = None

    @property
    def available(self) -> bool:
        if not (self.onnx_path and os.path.exists(self.onnx_path)):
            return False
        try:
            import onnxruntime  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    def _ensure(self) -> None:
        if not self.available:
            raise RuntimeError(
                "CPU-движок SigLIP2 недоступен: нет ONNX "
                f"{self.onnx_path!r} или onnxruntime (задайте "
                "REID_API_CPU_SIGLIP_ONNX)")
        if self._backend is None:
            from service.infer.cpu_backend import SiglipCpuBackend

            self._backend = SiglipCpuBackend(
                self.onnx_path, max_patches=self.max_patches,
                threads=self.threads, max_ram_mb=self.max_ram_mb)

    def embed(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        self._ensure()
        import pandas as pd

        x, y, w, h = bbox
        with tempfile.TemporaryDirectory(prefix="reid_api_cpu_") as tmp:
            image.convert("RGB").save(os.path.join(tmp, "query.jpg"),
                                      "JPEG", quality=95)
            df = pd.DataFrame([{"image_id": "query", "x": int(x), "y": int(y),
                                "w": int(w), "h": int(h)}])
            vec = self._backend.extract(df, tmp, batch_size=self.batch_size)[0]
        return np.asarray(vec, dtype=np.float32)

    def explain(self, image: Image.Image, bbox: tuple[int, int, int, int]
                ) -> tuple[np.ndarray, str]:
        # Полноценный Grad-CAM по активациям — W2-6. Пока — честный фолбэк.
        heat, _ = gradient_heatmap(image, bbox)
        return heat, "input-gradient (fallback; Grad-CAM — W2-6)"


# ---------------------------------------------------------------------------
# Сборка / описание моделей
# ---------------------------------------------------------------------------
def build_engine(cfg: AppConfig) -> InferenceEngine:
    if cfg.engine == "infer":
        # CPU-прототип: SigLIP2 ONNX на CPU EP (device="cpu" — явный выбор).
        # SigLIP2 — единственный поддерживаемый CPU-вариант, поэтому cfg.variant
        # здесь игнорируется (DINOv2/fusion требуют CUDA). GPU-путь (cuda/auto)
        # не изменён.
        if str(cfg.device).strip().lower() == "cpu":
            onnx = cfg.cpu_siglip_onnx or cfg.siglip_onnx
            return CpuSiglipEngine(
                onnx, threads=cfg.siglip_threads,
                max_patches=cfg.cpu_max_patches,
                max_ram_mb=cfg.cpu_max_ram_mb,
                batch_size=cfg.cpu_batch_size,
                threshold=cfg.threshold)
        return InferEngine(variant=cfg.variant, device=cfg.device,
                           dino_ckpt=cfg.dino_ckpt, siglip_onnx=cfg.siglip_onnx,
                           batch_size=cfg.batch_size,
                           siglip_threads=cfg.siglip_threads)
    return StubEngine(dim=cfg.dim, seed=cfg.stub_seed,
                      threshold=cfg.stub_threshold)


def describe_models(cfg: AppConfig, engine: InferenceEngine) -> list[dict]:
    """Список моделей/вариантов для ``/api/v1/models``.

    Доступность определяется ТИПОМ активного движка, а не только наличием
    файлов весов:

    * :class:`StubEngine` — доступен только ``stub``;
    * :class:`CpuSiglipEngine` — доступен только ``siglip`` (CPU не умеет
      DINOv2/fusion — наличие файла весов ничего не меняет);
    * :class:`InferEngine` (GPU) — ``dino``/``siglip``/``fusion`` по наличию
      соответствующих весов.

    ``default`` — имя активного движка, ``is_default`` — только у него.
    """
    default = engine.name
    is_stub = isinstance(engine, StubEngine)
    is_cpu = isinstance(engine, CpuSiglipEngine)

    # Наличие файлов весов — релевантно только GPU-пути (InferEngine).
    dino_ok = bool(cfg.dino_ckpt and os.path.exists(cfg.dino_ckpt))
    # SigLIP2 может быть доступен как GPU-ONNX (cfg.siglip_onnx) и/или как
    # CPU-ONNX (cfg.cpu_siglip_onnx) — оба валидны для варианта siglip.
    sig_path = None
    for cand in (cfg.siglip_onnx, cfg.cpu_siglip_onnx):
        if cand and os.path.exists(cand):
            sig_path = cand
            break

    if is_stub:
        # Заглушка — единственный доступный вариант в stub-режиме.
        stub_available = True
        dino_available = siglip_available = fusion_available = False
        sig_weights = None
    elif is_cpu:
        # CPU-прототип умеет только SigLIP2; DINOv2/fusion требуют CUDA и
        # недоступны, даже если файлы весов присутствуют. Реальная доступность
        # SigLIP2 — по движку (ONNX-файл + onnxruntime).
        stub_available = False
        dino_available = fusion_available = False
        siglip_available = bool(engine.available)
        sig_path = engine.onnx_path or sig_path
        sig_weights = sig_path if siglip_available else None
    else:
        # GPU-путь (InferEngine): доступность — по наличии весов.
        stub_available = False
        dino_available = dino_ok
        siglip_available = sig_path is not None
        fusion_available = dino_ok and sig_path is not None
        sig_weights = sig_path if siglip_available else None

    out = [{
        "name": "stub", "variant": "stub", "dim": int(cfg.dim),
        "description": "Детерминированная заглушка без весов (демо/тесты).",
        "available": stub_available, "is_default": default == "stub",
        "weights": None,
    }]
    out.append({
        "name": "dino", "variant": "dino", "dim": 512,
        "description": "DINOv2-B + GeM + BNNeck (champion exp-0007).",
        "available": dino_available, "is_default": default == "dino",
        "weights": cfg.dino_ckpt if dino_available else None,
    })
    out.append({
        "name": "siglip", "variant": "siglip", "dim": 512,
        "description": ("SigLIP2 NaFlex vehicle-ReID (внешние публичные веса, "
                        "ONNX; на CPU — CPUExecutionProvider без torch)."),
        "available": siglip_available, "is_default": default == "siglip",
        "weights": sig_weights,
    })
    out.append({
        "name": "fusion", "variant": "fusion", "dim": 1024,
        "description": "fusion = L2([SigLIP2, 0.6*DINOv2]) — деплой-чемпион.",
        "available": fusion_available, "is_default": default == "fusion",
        "weights": None,
    })
    return out


__all__ = [
    "InferenceEngine",
    "StubEngine",
    "InferEngine",
    "CpuSiglipEngine",
    "resolve_siglip_threshold",
    "CPU_SIGLIP_DIM",
    "CPU_SIGLIP_DEFAULT_THRESHOLD",
    "build_engine",
    "describe_models",
    "gradient_heatmap",
]
