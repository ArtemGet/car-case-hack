"""service.infer.run — офлайн batch-раннер трёх файлов сдачи (W2-5).

Одна команда, полностью офлайн (все веса в образе), детерминированно выдаёт
ровно три артефакта в ``--out`` по контракту ``docs/_workspace/INTERFACES.md §2``:

    submission.csv   без заголовка; query_id + ровно 10 gallery_id по убыванию
    embeddings.npy   float32 (1110+750, D), порядок query(в порядке файла) → gallery
    candidates.csv   заголовок query_id,gallery_id,confidence; только top-1 при
                     score >= threshold; иначе строки НЕТ (отказ)

Пайплайн-чемпион (``--variant fusion``, вариант A финального деплоя):
    fusion = L2( concat[ L2(SigLIP2-naflex 512d ONNX fp16), 0.8 * L2(DINOv2-B@224 512d ONNX fp16) ] )
    + query-side TTA ВЫКЛЮЧЕНА (224-only) — deploy-конфиг variant A
    + per-query k-reciprocal (k1=8, k2=3, lam=0.5, pool=300; ``reid.rerank``)
    + порог отказа из ``reid.calibrate`` (или из ``reports/calibration_fusion.json``,
      пере-калиброванного на этой же 224-only GPU-шкале).

Препроцессинг по умолчанию — **GPU** (``--preproc gpu``): CPU draft-decode + bbox
ROI, затем CUDA letterbox/normalize и ONNX IOBinding
(``reid.export.gpu_preproc`` через проверенные бэкенды ``tools.bench_perf``);
``--preproc cpu`` возвращает старый PIL-путь для абляции.

КРИТИЧНО: train_320 crop-cache — **train-only** (он завышал val на ~+0.014) и в
инференсе/вале запрещён: ``--crop-cache`` и ``--stage-size>0`` отклоняются, пока
не передан явный ``--allow-train-preproc`` (для воспроизведения старых отчётов).

DINOv2 и SigLIP2 запускаются через ONNX Runtime fp16; CUDA-провайдер обязателен
(``assert providers[0] == "CUDAExecutionProvider"``), тихий CPU-откат запрещён.
TTA@280 (абляция) использует отдельный статический граф
(``artifacts/dinov2_b_280_fp16.onnx``), т.к. канонический fp16-экспорт DINOv2
зафиксирован на 224.

Доступны варианты ``--variant {fusion,siglip,dino}`` — ``fusion`` по умолчанию.

CPU-ПРОТОТИП (демо, 2 CPU / 2 ГБ, без GPU): ``--device cpu`` (или env
``REID_DEVICE=cpu``) включает единственную CPU-ветку — SigLIP2 ONNX на
ONNX Runtime **CPUExecutionProvider** (провайдер ровно один, assert), PIL/NumPy
препроцесс без torch, ``--variant siglip``. DINOv2 не грузится, память ~0.4 ГБ,
``--max-ram-mb`` следит за RSS. Боевой GPU-путь (``--device cuda``, default)
по-прежнему жёстко требует CUDA и НЕ откатывается на CPU.

Пример (GPU):
    python -m service.infer.run --images /in/images \
        --query /in/test_query.csv --gallery /in/test_gallery.csv \
        --out /out --variant fusion --preproc gpu

Пример (CPU-демо):
    REID_DEVICE=cpu python -m service.infer.run --images /in/images \
        --query /in/test_query.csv --gallery /in/test_gallery.csv \
        --out /out --variant siglip --device cpu --max-ram-mb 1800
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

from reid import rerank
from reid.calibrate import (DEFAULT_THRESHOLD, score_confidences,
                            write_candidates)

try:  # W2-4 redo: champion fusion w=0.8 config lives in reid.calibrate
    from reid.calibrate import (DEFAULT_FUSION_W,
                                DEFAULT_RERANK_FUSION_W08,
                                DEFAULT_THRESHOLD_FUSION_FINAL,
                                DEFAULT_THRESHOLD_FUSION_W08,
                                DEFAULT_THRESHOLD_FUSION_W08_FP16)
    # Variant-A deploy scale (224-only, TTA OFF, GPU-preproc) recalibrated on the
    # val split: reports/calibration_fusion.json → chosen 0.6970806 (F1 0.9702 /
    # TNR 1.0 vs 0.97553 at the TTA fp16 constant). calibration-agent owns the
    # canonical constant DEFAULT_THRESHOLD_FUSION_FINAL in reid.calibrate; the
    # runner reads it directly.
    DEFAULT_THRESHOLD_FUSION_224 = float(DEFAULT_THRESHOLD_FUSION_FINAL)
except ImportError:  # pragma: no cover - older reid.calibrate
    DEFAULT_FUSION_W = 0.8
    DEFAULT_RERANK_FUSION_W08 = {"k1": 8, "k2": 3, "lam": 0.5, "pool_size": 200}
    DEFAULT_THRESHOLD_FUSION_W08 = 0.690625786781311
    DEFAULT_THRESHOLD_FUSION_W08_FP16 = 0.6897585988044739
    DEFAULT_THRESHOLD_FUSION_FINAL = 0.6970806121826172
    DEFAULT_THRESHOLD_FUSION_224 = 0.6970806121826172
from reid.data.io import QUERY_COLUMNS, read_csv

# NOTE: ``service.infer.backends`` (torch/torchvision, CUDA) is imported lazily
# inside the GPU branch so the CPU prototype never pays for the torch import.
# The CPU path is served by ``service.infer.cpu_backend`` (PIL + NumPy + ORT CPU
# EP only); see ``--device cpu`` below.

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Frozen per-variant configuration (best-from-val; see reports/exp-*).
# ---------------------------------------------------------------------------
VARIANTS = ("fusion", "siglip", "dino")
PREPROC = ("gpu", "cpu")
DEFAULT_PREPROC = "gpu"

RERANK = {
    # Champion fusion postproc. Base params sourced from reid.calibrate (W2-4
    # redo, exp-0050) so the calibration-agent owns the frozen pair in one place,
    # but pool_size is bumped 200 -> 300 per exp-0057 (W2-11 rr_pool_grid:
    # pool=300, k1=8,k2=3,lam=0.5 -> val mAP@10 0.72340, robust across 300/500/750).
    # The refusal threshold is STILL read from reid.calibrate (it is being
    # recalibrated on the fp16 scale by calibration-agent).
    "fusion": {**dict(DEFAULT_RERANK_FUSION_W08), "pool_size": 300},
    # Non-deployed fallbacks (kept for ablation); retuned for their own scores.
    "dino":   {"k1": 5, "k2": 2, "lam": 0.7, "pool_size": 100},
    "siglip": {"k1": 8, "k2": 4, "lam": 0.5, "pool_size": 200},
}
# Query-side TTA scales. Vehicles are not left/right symmetric -> no hflip.
# DEPLOY config (variant A): TTA is OFF — fusion runs 224-only. A 224-only DINOv2
# static graph is the canonical fp16 export; the 280 ablation needs a second graph.
# Pass --dino-tta '224,280' to re-enable (ablation only; it regressed deploy
# latency and is not the frozen config).
# DINOv2 is a /14 ViT; SigLIP2 uses a patch budget, not a /14 pixel scale.
TTA = {"dino": (224,), "siglip": (), "fusion": (224,)}
# Champion fusion weight (W2-8, exp-0046): L2(concat[L2(SigLIP2), 0.8*L2(DINOv2)]).
FUSION_W = float(DEFAULT_FUSION_W)

# Refusal thresholds (cosine top-1 after TTA + k-reciprocal); see
# ``resolve_threshold`` for the resolution order. The fusion value comes from
# ``reid.calibrate`` (calibration-agent) — never tuned on test. Deploy is ONNX
# fp16, so the fusion threshold is the fp16-scale one (exp-0061/exp-0066).
DEFAULT_THRESHOLDS = {
    "dino": float(DEFAULT_THRESHOLD),
    # calibrated on the frozen val split (seed=42, 375 q / 81 open-set).
    "siglip": 0.8943735957145691,
    # variant-A deploy (ONNX fp16, GPU-preproc, 224-only / TTA OFF, w=0.8,
    # k-rr 8/3/0.5/300). Recalibrated on the 224-only scale via
    # reports/calibration_fusion.json (F1 0.9702 / TNR 1.0 / score 0.97916);
    # the TTA fp16 constant 0.6897586 is weaker on this scale (score 0.97553).
    # Kept as a module default so the offline runtime image (which excludes
    # reports/) still uses the right value; calibration-agent owns the canonical
    # constant in reid.calibrate.
    "fusion": float(DEFAULT_THRESHOLD_FUSION_224),
}

# Offline weight files (all inside the image; no network at run time).
# DINOv2: the canonical fp16 export is a STATIC 224 graph, so TTA@280 needs a
# second static graph exported at 280 (same exp-0007 weights).
DEFAULT_DINO_ONNX = os.path.join("artifacts", "dinov2_b_fp16.onnx")
DEFAULT_DINO_ONNX_280 = os.path.join("artifacts", "dinov2_b_280_fp16.onnx")
DEFAULT_SIGLIP_ONNX = os.path.join("artifacts", "siglip2_fp16.onnx")

# Kept for the (non-deployed) PyTorch DINOv2 fallback / historical reports.
DEFAULT_DINO_CKPT = os.path.join("runs", "exp-0007", "best.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_images_dir(path: str) -> str:
    """Accept either the JPEG folder itself or a dataset root with ``images/``."""
    p = os.path.abspath(path)
    if not os.path.isdir(p):
        raise SystemExit(f"--images: каталог не найден: {p}")
    has_jpg = any(f.lower().endswith(".jpg") for f in os.listdir(p))
    if has_jpg:
        return p
    nested = os.path.join(p, "images")
    if os.path.isdir(nested):
        return nested
    return p


def _resolve_siglip_weights(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    p = os.path.join(REPO, DEFAULT_SIGLIP_ONNX)
    if os.path.exists(p):
        return p
    raise SystemExit("не найден SigLIP2 ONNX: укажите --siglip-weights")


def _dino_onnx_for(size: int, explicit_224: str | None = None,
                   explicit_280: str | None = None) -> str:
    """Map a DINOv2 TTA scale to its static ONNX graph (224 / 280)."""
    size = int(size)
    if size == 224:
        cand = explicit_224 or os.path.join(REPO, DEFAULT_DINO_ONNX)
    elif size == 280:
        cand = explicit_280 or os.path.join(REPO, DEFAULT_DINO_ONNX_280)
    else:
        raise SystemExit(
            f"DINOv2 ONNX TTA scale {size} не поддержан (только 224/280); "
            "передайте --dino-tta '224' или добавьте граф для этого скейла")
    if not os.path.exists(cand):
        raise SystemExit(f"не найден DINOv2 ONNX: {cand}")
    return os.path.abspath(cand)


def resolve_threshold(variant: str, override=None) -> float:
    """Resolve the refusal threshold for ``variant``.

    Order (first hit wins):
      1. explicit ``override`` (CLI ``--threshold``);
      2. env ``REID_<VARIANT>_THRESHOLD``;
      3. ``reports/calibration_<variant>.json`` → ``chosen.threshold`` (written
         by ``reid.calibrate`` — the calibration-agent's hand-off);
      4. ``DEFAULT_THRESHOLDS[variant]``.
    """
    if override is not None:
        return float(override)
    env = os.environ.get(f"REID_{variant.upper()}_THRESHOLD")
    if env:
        return float(env)
    cal = os.path.join(REPO, "reports", f"calibration_{variant}.json")
    if os.path.exists(cal):
        try:
            with open(cal, "r", encoding="utf-8") as f:
                data = json.load(f)
            thr = data.get("chosen", {}).get("threshold")
            if thr is not None:
                return float(thr)
        except Exception:  # noqa: BLE001 - fall through to the frozen default
            pass
    thr = DEFAULT_THRESHOLDS.get(variant)
    if thr is None:
        raise SystemExit(
            f"для variant={variant} порог не задан: укажите --threshold "
            f"(или откалибруйте его на val, см. service.infer.val_eval)")
    return float(thr)


def _check_coverage(df, images_dir, tag) -> dict:
    ids = df["image_id"].tolist()
    missing = [i for i in ids if not os.path.exists(
        os.path.join(images_dir, f"{i}.jpg"))]
    cov = {
        "split": tag,
        "n_ids": len(ids),
        "n_unique": len(set(ids)),
        "present": len(ids) - len(missing),
        "missing": len(missing),
        "coverage": (len(ids) - len(missing)) / max(1, len(ids)),
    }
    if missing:
        raise SystemExit(
            f"[{tag}] отсутствуют изображения ({len(missing)}): {missing[:5]}")
    if cov["n_unique"] != len(ids):
        raise SystemExit(f"[{tag}] дубликаты image_id в CSV")
    return cov


def _fuse_query_tta(base: np.ndarray, extras: list[np.ndarray]) -> np.ndarray:
    return np.stack([
        rerank.fuse_embeddings([base[i]] + [e[i] for e in extras])
        for i in range(base.shape[0])
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline: embeddings
# ---------------------------------------------------------------------------
def _guard_train_preproc(crop_cache, stage_size, preproc, allow):
    """Reject train-only preprocessing (the train_320 crop cache) unless forced.

    The train_320 cache is built from the training split and inflated val mAP by
    ~+0.014; it must never be used for inference or val. ``stage_size>0`` likewise
    reproduces that 2-stage train pipeline on the GPU. Both are refused unless the
    explicit escape hatch ``allow`` is set (for reproducing old reports only).
    """
    if crop_cache:
        norm = os.path.normpath(str(crop_cache)).replace("\\", "/").lower()
        if "train_320" in norm or "/train/" in norm or norm.endswith("/train"):
            raise SystemExit(
                "ЗАЩИТА: crop-cache train_320 — train-only (завышает val ~+0.014); "
                "инференс/вал не должны его использовать. Уберите --crop-cache.")
        if preproc == "gpu":
            raise SystemExit(
                "GPU-preproc не использует crop-cache; уберите --crop-cache.")
        if not allow:
            raise SystemExit(
                "crop-cache отключён по умолчанию (W2-5-final); передайте "
                "--allow-train-preproc только для воспроизведения старых отчётов.")
    if int(stage_size or 0) > 0 and not allow:
        raise SystemExit(
            "ЗАЩИТА: --stage-size>0 воспроизводит train_320-пайплайн (train-only); "
            "для инференса/вала запрещено (--allow-train-preproc для репродукции).")
    return crop_cache


def _is_cpu_device(device) -> bool:
    """True only for an explicit ``device="cpu"`` (never "auto")."""
    return str(device).strip().lower() == "cpu"


def _build_embeddings_cpu(variant, images_dir, query_df, gallery_df, *,
                          siglip_weights=None, batch_size=32,
                          siglip_threads=8, max_ram_mb=0, preproc="cpu"):
    """CPU-prototype embeddings: SigLIP2-only, ORT CPU EP, PIL/NumPy pre-process.

    The demo box (2 CPU / 2 GB / no GPU) loads **only** the SigLIP2 graph — the
    DINOv2 backbone is never constructed and the fusion path is unreachable
    here — so peak RSS stays ~0.4 GB (vs the GPU fusion's two backbones).
    """
    if variant != "siglip":
        raise SystemExit(
            "CPU-прототип поддерживает только --variant siglip (SigLIP2). "
            f"Получено variant={variant!r}; fusion/dino требуют CUDA "
            "(--device cuda, без CPU-отката).")
    from .cpu_backend import SiglipCpuBackend, rss_mb

    q_ids = query_df["image_id"].astype(str).tolist()
    g_ids = gallery_df["image_id"].astype(str).tolist()
    wi = _resolve_siglip_weights(siglip_weights)
    if preproc == "gpu":
        _log("[extract] --device cpu: --preproc gpu несовместим; "
             "использую PIL-препроцесс (aspect-preserving, preproc=cpu)")
    info = {"variant": variant, "device": "cpu", "preproc": "cpu",
            "tta_scales": [], "fusion_w": None, "timings_s": {},
            "max_ram_mb": int(max_ram_mb or 0),
            "providers": ["CPUExecutionProvider"]}
    _log(f"[extract] CPU-прототип SigLIP2 ONNX (ORT CPU EP), "
         f"threads={siglip_threads} batch={batch_size}")
    _log(f"[extract]   weights={wi} rss={rss_mb():.0f}MB")
    t0 = time.time()
    be = SiglipCpuBackend(wi, threads=siglip_threads, max_ram_mb=max_ram_mb)
    q = be.extract(query_df, images_dir, batch_size=batch_size)
    g = be.extract(gallery_df, images_dir, batch_size=batch_size)
    info["timings_s"]["siglip"] = time.time() - t0
    info["rss_after_mb"] = rss_mb()
    q = np.ascontiguousarray(q, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    assert q.shape[0] == len(q_ids), (q.shape, len(q_ids))
    assert g.shape[0] == len(g_ids), (g.shape, len(g_ids))
    info["dim"] = int(q.shape[1]) if q.ndim == 2 else 0
    _log(f"[extract] done in {info['timings_s']['siglip']:.1f}s "
         f"dim={info['dim']} rss={info['rss_after_mb']:.0f}MB")
    return q, g, q_ids, g_ids, info


def build_embeddings(variant, images_dir, query_df, gallery_df, *, device=None,
                     dino_ckpt=None, dino_onnx=None, dino_onnx_280=None,
                     siglip_weights=None, dino_tta=None,
                     w=FUSION_W, batch_size=64, num_workers=0,
                     siglip_threads=8, crop_cache=None, seed=42,
                     preproc=DEFAULT_PREPROC, draft_factor=1.0, stage_size=0,
                     allow_train_preproc=False, max_ram_mb=0):
    """Return (q_emb, g_emb, q_ids, g_ids, info).

    ``preproc='gpu'`` (default) runs crop/letterbox/normalise on CUDA and the
    DINOv2/SigLIP2 forwards through ONNX Runtime CUDA EP with IOBinding
    (``reid.export.gpu_preproc``). ``preproc='cpu'`` is the legacy PIL/ORT path.
    DINOv2 runs end-to-end as fp16 ONNX; TTA (now 224-only by default) uses one
    static graph per scale. ``dino_ckpt`` is accepted for backwards-compatible
    callers but the deployed path never loads ``.pt``.
    """
    crop_cache = _guard_train_preproc(crop_cache, stage_size, preproc,
                                      allow_train_preproc)
    # Explicit CPU prototype only (never "auto"): SigLIP2-only, ORT CPU EP.
    # Everything below this point is the CUDA path and hard-requires CUDA.
    if _is_cpu_device(device):
        return _build_embeddings_cpu(
            variant, images_dir, query_df, gallery_df,
            siglip_weights=siglip_weights, batch_size=batch_size,
            siglip_threads=siglip_threads, max_ram_mb=max_ram_mb,
            preproc=preproc)
    from .backends import (DinoOnnxBackend, GpuExtractor, SiglipBackend,
                           resolve_device)
    device = resolve_device(device or "auto")
    q_ids = query_df["image_id"].astype(str).tolist()
    g_ids = gallery_df["image_id"].astype(str).tolist()
    raw_tta = TTA[variant] if dino_tta is None else dino_tta
    if isinstance(raw_tta, str):  # tolerate "224,280" from a CLI
        raw_tta = [s for s in raw_tta.split(",") if s.strip()]
    scales = tuple(int(s) for s in raw_tta)
    if variant in ("fusion", "dino") and not scales:
        scales = (224,)
    info = {"variant": variant, "device": str(device), "preproc": preproc,
            "tta_scales": list(scales),
            "fusion_w": float(w) if variant == "fusion" else None,
            "timings_s": {}}

    # ---- GPU-preprocess path (variant A deploy) ---------------------------
    if preproc == "gpu":
        if not str(device).startswith("cuda"):
            raise SystemExit(
                "preproc=gpu требует CUDA; передайте --device cuda "
                "(или --preproc cpu для CPU-прогона)")
        _log(f"[extract] GPU-preproc {variant} scales={list(scales)} w={w}")
        t0 = time.time()
        # resolve the frozen default weight paths (same as the CPU path)
        dino_onnx = dino_onnx or os.path.join(REPO, DEFAULT_DINO_ONNX)
        if any(int(s) == 280 for s in scales):
            dino_onnx_280 = dino_onnx_280 or os.path.join(
                REPO, DEFAULT_DINO_ONNX_280)
        siglip_weights = (_resolve_siglip_weights(siglip_weights)
                          if variant in ("fusion", "siglip") else siglip_weights)
        for p in (dino_onnx, dino_onnx_280, siglip_weights):
            if p and not os.path.exists(p):
                raise SystemExit(f"не найден ONNX-вес: {p}")
        ex = GpuExtractor(
            variant, dino_onnx=dino_onnx, dino_onnx_280=dino_onnx_280,
            siglip_weights=siglip_weights, scales=scales, w=w,
            draft_factor=draft_factor, stage_size=stage_size, device="cuda")
        _log(f"[extract]   gpu backend={ex.desc}")
        info["gpu_desc"] = ex.desc
        info["gpu_weight_files"] = list(ex.weight_files)
        g = ex.extract_df(gallery_df, images_dir, batch_size=batch_size)
        q = ex.extract_df(query_df, images_dir, batch_size=batch_size)
        info["timings_s"]["gpu"] = time.time() - t0
        q = np.ascontiguousarray(q, dtype=np.float32)
        g = np.ascontiguousarray(g, dtype=np.float32)
        assert q.shape[0] == len(q_ids), (q.shape, len(q_ids))
        assert g.shape[0] == len(g_ids), (g.shape, len(g_ids))
        if variant == "fusion":
            info["fusion_dim"] = int(q.shape[1])
        return q, g, q_ids, g_ids, info

    # ---- legacy CPU-preprocess path ---------------------------------------
    q_dino = g_dino = q_sig = g_sig = None

    if variant in ("dino", "fusion"):
        if not scales:
            scales = (224,)
        base = scales[0]
        _log(f"[extract] DINOv2 ONNX fp16 (CUDA EP), scales={list(scales)}")
        t0 = time.time()
        sessions = {}
        for s in scales:
            wi = _dino_onnx_for(s, dino_onnx, dino_onnx_280)
            _log(f"[extract]   dino@{s} onnx={wi}")
            sessions[s] = DinoOnnxBackend(wi, size=s, threads=siglip_threads,
                                          draft_factor=2.0)
        info["dino_onnx"] = {int(s): sessions[s].weights for s in scales}
        g_dino = sessions[base].extract(gallery_df, images_dir, base,
                                        batch_size=batch_size,
                                        cache_dir=crop_cache)
        q_dino = sessions[base].extract(query_df, images_dir, base,
                                        batch_size=batch_size,
                                        cache_dir=crop_cache)
        extras = [sessions[s].extract(query_df, images_dir, s,
                                      batch_size=batch_size,
                                      cache_dir=crop_cache)
                  for s in scales[1:]]
        if extras:
            q_dino = _fuse_query_tta(q_dino, extras)
        info["timings_s"]["dino"] = time.time() - t0
        del sessions

    if variant in ("siglip", "fusion"):
        wi = _resolve_siglip_weights(siglip_weights)
        _log(f"[extract] SigLIP2 onnx={wi}")
        t0 = time.time()
        sig = SiglipBackend(wi, threads=siglip_threads)
        q_sig = sig.extract(query_df, images_dir)
        g_sig = sig.extract(gallery_df, images_dir)
        info["timings_s"]["siglip"] = time.time() - t0

    if variant == "dino":
        q, g = q_dino, g_dino
    elif variant == "siglip":
        q, g = q_sig, g_sig
    else:
        q = rerank.l2norm(np.concatenate([q_sig, w * q_dino], axis=1))
        g = rerank.l2norm(np.concatenate([g_sig, w * g_dino], axis=1))
        info["fusion_dim"] = int(q.shape[1])

    q = np.ascontiguousarray(q, dtype=np.float32)
    g = np.ascontiguousarray(g, dtype=np.float32)
    assert q.shape[0] == len(q_ids), (q.shape, len(q_ids))
    assert g.shape[0] == len(g_ids), (g.shape, len(g_ids))
    return q, g, q_ids, g_ids, info


# ---------------------------------------------------------------------------
# Pipeline: rank + write the three artefacts
# ---------------------------------------------------------------------------
def _score_stats(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return {}
    qs = np.percentile(s, [1, 5, 25, 50, 75, 95, 99]).round(6).tolist()
    return {
        "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
        "std": float(s.std()), "p1": qs[0], "p5": qs[1], "p25": qs[2],
        "p50": qs[3], "p75": qs[4], "p95": qs[5], "p99": qs[6],
    }


def rank_embeddings(q, g, variant, rerank_cfg=None):
    """Per-query k-reciprocal ordering + cosine top-1 confidence.

    Returns ``(orders, rr_scores, confidences)``. ``orders`` is (Nq, Ng),
    best-first; ``confidences`` is cross-query comparable cosine top-1.
    """
    rr = dict(RERANK[variant] if rerank_cfg is None else rerank_cfg)
    n_q, n_g = q.shape[0], g.shape[0]
    _log(f"[rank] per-query k-reciprocal {rr} on {n_q} queries / {n_g} gallery")
    orders = np.empty((n_q, n_g), dtype=np.int64)
    scores_rr = np.empty((n_q, n_g), dtype=np.float32)
    for i in range(n_q):
        prep = rerank.prepare_query(q[i], g, pool_size=rr["pool_size"])
        orders[i], scores_rr[i] = rerank.rank_prepared(
            prep, k1=rr["k1"], k2=rr["k2"], lam=rr["lam"])
    conf = score_confidences(q, g, orders, mode="cosine")
    return orders, scores_rr, conf


def rank_and_write(out_dir, q, g, q_ids, g_ids, *, variant, threshold,
                   top_k=10, rerank_cfg=None):
    """Write submission.csv / embeddings.npy / candidates.csv. Returns info."""
    os.makedirs(out_dir, exist_ok=True)
    rr = dict(RERANK[variant] if rerank_cfg is None else rerank_cfg)
    n_q, n_g = q.shape[0], g.shape[0]
    if n_g < top_k:
        raise SystemExit(f"галерея меньше {top_k}: {n_g}")

    t0 = time.time()
    orders, _scores_rr, conf = rank_embeddings(q, g, variant, rerank_cfg=rr)
    rank_time = time.time() - t0

    # submission.csv — no header, exactly top_k gids, best first, no dups
    sub_path = os.path.join(out_dir, "submission.csv")
    with open(sub_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            row = [qid] + [g_ids[j] for j in orders[i, :top_k]]
            assert len(set(row[1:])) == top_k
            w.writerow(row)

    # embeddings.npy — float32, query file order then gallery
    emb_path = os.path.join(out_dir, "embeddings.npy")
    np.save(emb_path, np.vstack([q, g]).astype(np.float32))

    # candidates.csv — top-1 only where score >= threshold (refusal = no row)
    cand_path = os.path.join(out_dir, "candidates.csv")
    n_written = write_candidates(cand_path, q_ids, g_ids, orders, conf,
                                 float(threshold))

    info = {
        "n_query": n_q, "n_gallery": n_g, "dim": int(q.shape[1]),
        "rerank": rr, "rank_time_s": rank_time,
        "threshold": float(threshold),
        "accepted": n_written, "refused": n_q - n_written,
        "refusal_rate": (n_q - n_written) / max(1, n_q),
        "score_cosine_top1": _score_stats(conf),
        "artifacts": {
            "submission.csv": {"bytes": os.path.getsize(sub_path),
                               "sha256": _sha256(sub_path)},
            "embeddings.npy": {"bytes": os.path.getsize(emb_path),
                               "sha256": _sha256(emb_path)},
            "candidates.csv": {"bytes": os.path.getsize(cand_path),
                               "sha256": _sha256(cand_path)},
        },
    }
    return info


def run(images_dir, query_csv, gallery_csv, out_dir, *, variant="fusion",
        threshold=None, **kw):
    """Full pipeline. Returns a JSON-serialisable run report."""
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}")
    images_dir = _resolve_images_dir(images_dir)
    query_df = read_csv(query_csv, required=QUERY_COLUMNS)
    gallery_df = read_csv(gallery_csv, required=QUERY_COLUMNS)

    _log(f"[coverage] images_dir={images_dir}")
    cov_q = _check_coverage(query_df, images_dir, "query")
    cov_g = _check_coverage(gallery_df, images_dir, "gallery")
    _log(f"[coverage] query={cov_q['present']}/{cov_q['n_ids']} "
         f"gallery={cov_g['present']}/{cov_g['n_ids']}")

    th = resolve_threshold(variant, threshold)

    t0 = time.time()
    q, g, q_ids, g_ids, info = build_embeddings(
        variant, images_dir, query_df, gallery_df, **kw)
    info["coverage"] = {"query": cov_q, "gallery": cov_g}
    info["extract_total_s"] = time.time() - t0

    winfo = rank_and_write(out_dir, q, g, q_ids, g_ids, variant=variant,
                           threshold=th, top_k=10)
    info.update(winfo)
    info["total_s"] = time.time() - t0
    info["out"] = os.path.abspath(out_dir)

    _log(f"[scores] cosine top-1: {json.dumps(winfo['score_cosine_top1'])}")
    _log(f"[refusal] accepted={winfo['accepted']} refused={winfo['refused']} "
         f"({winfo['refusal_rate'] * 100:.1f}%) threshold={th:.4f}")
    _log(f"[done] {info['total_s']:.1f}s -> {info['out']} "
         f"(3 files, dim={info['dim']})")
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Офлайн batch-инференс: 3 файла сдачи (W2-5)")
    ap.add_argument("--images", required=True,
                    help="каталог изображений <image_id>.jpg (или dataset root)")
    ap.add_argument("--query", required=True, help="test_query.csv")
    ap.add_argument("--gallery", required=True, help="test_gallery.csv")
    ap.add_argument("--out", required=True, help="каталог вывода (3 файла)")
    ap.add_argument("--variant", default="fusion", choices=list(VARIANTS),
                    help="fusion (по умолчанию, лучший mAP) | siglip | dino")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог отказа; по умолчанию — калиброванный на val")
    # optional overrides / knobs
    ap.add_argument("--model", default=None,
                    help="алиас основного ONNX: fusion/dino -> DINOv2-224, "
                         "siglip -> SigLIP2")
    ap.add_argument("--dino-ckpt", default=None,
                    help="[legacy] PyTorch DINOv2 checkpoint (не используется "
                         "задеплоенным ONNX-путём)")
    ap.add_argument("--dino-onnx", default=None,
                    help="DINOv2 fp16 ONNX @224 (по умолчанию artifacts/dinov2_b_fp16.onnx)")
    ap.add_argument("--dino-onnx-280", default=None,
                    help="DINOv2 fp16 ONNX @280 для TTA (по умолчанию "
                         "artifacts/dinov2_b_280_fp16.onnx)")
    ap.add_argument("--siglip-weights", default=None,
                    help="SigLIP2 fp16 ONNX (по умолчанию artifacts/siglip2_fp16.onnx)")
    ap.add_argument("--dino-tta", default=None,
                    help="DINOv2 TTA-скейлы, напр. '224,280' (абляция); "
                         "по умолчанию TTA ВЫКЛЮЧЕНА — 224-only (variant A)")
    ap.add_argument("--w", type=float, default=FUSION_W,
                    help="вес DINOv2 в fusion-конкатенации")
    ap.add_argument("--preproc", default=DEFAULT_PREPROC, choices=list(PREPROC),
                    help="препроцессинг: gpu (по умолчанию, CUDA letterbox+"
                         "normalize+IOBinding) | cpu (legacy PIL)")
    ap.add_argument("--draft-factor", type=float, default=1.0,
                    help="JPEG partial-decode margin (Image.draft); GPU-путь")
    ap.add_argument("--stage-size", type=int, default=0,
                    help="[только репродукция] 2-stage letterbox (напр. 320), "
                         "воспроизводит train_320-пайплайн; запрещён без "
                         "--allow-train-preproc")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--siglip-threads", type=int, default=8)
    ap.add_argument("--crop-cache", default=None,
                    help="[train-only, запрещён] кэш кропов train_320; "
                         "инференс/вал его не используют")
    ap.add_argument("--allow-train-preproc", action="store_true",
                    help="аварийный обход защиты train_320 (только для "
                         "воспроизведения старых отчётов; НЕ для сдачи)")
    ap.add_argument("--device", default=os.environ.get("REID_DEVICE", "cuda"),
                    choices=["cuda", "cpu"],
                    help="cuda (боевой, default) | cpu (демо-прототип: SigLIP2 "
                         "на CPUExecutionProvider, только --variant siglip; "
                         "env REID_DEVICE=cpu). CPU — только при явном выборе.")
    ap.add_argument("--max-ram-mb", type=int, default=0,
                    help="лимит RSS демо-прототипа в МБ (0 = без лимита); "
                         "превышение на CPU-ветке → ошибка")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-json", default=None,
                    help="путь для JSON-лога прогона (вне --out)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)

    if _is_cpu_device(args.device):
        # CPU prototype: seed without pulling torch.
        from .cpu_backend import set_cpu_determinism
        set_cpu_determinism(args.seed)
    else:
        from .backends import set_determinism
        set_determinism(args.seed)

    dino_ckpt = args.dino_ckpt
    dino_onnx = args.dino_onnx
    dino_onnx_280 = args.dino_onnx_280
    siglip_weights = args.siglip_weights
    if args.model:
        # Primary --model: DINOv2 ONNX for fusion/dino, SigLIP2 for siglip.
        if args.variant == "siglip":
            siglip_weights = siglip_weights or args.model
        else:
            dino_onnx = dino_onnx or args.model

    tta = None
    if args.dino_tta is not None:
        tta = tuple(int(s) for s in args.dino_tta.split(",") if s.strip())

    report = run(
        args.images, args.query, args.gallery, args.out,
        variant=args.variant, threshold=args.threshold,
        device=args.device, dino_ckpt=dino_ckpt, dino_onnx=dino_onnx,
        dino_onnx_280=dino_onnx_280, siglip_weights=siglip_weights,
        dino_tta=tta, w=args.w, batch_size=args.batch_size,
        num_workers=args.num_workers, siglip_threads=args.siglip_threads,
        crop_cache=args.crop_cache, seed=args.seed,
        preproc=args.preproc, draft_factor=args.draft_factor,
        stage_size=args.stage_size,
        allow_train_preproc=args.allow_train_preproc,
        max_ram_mb=args.max_ram_mb,
    )
    if args.log_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_json)), exist_ok=True)
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _log(f"[log] {args.log_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
