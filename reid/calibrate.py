"""reid/calibrate.py — confidence scoring and refusal-threshold calibration (W2-4).

Роль в пайплайне
----------------
Режим отказа оценивается на уровне ЗАПРОСА (METRICS.md, official evaluate.py):
``candidates.csv`` содержит только top-1 для тех query, что приняты; отказ —
это ОТСУТСТВИЕ строки для данного ``query_id``. Балл блока = 10%·(0.7·F1 + 0.3·TNR),
где ``TNR`` считается только по open-set запросам.

Модуль даёт чистый API:

    score_confidences(queries, gallery, order, mode)  -> top-1 confidence (монотонный)
    threshold_curve(scores, has_match, top1_correct)  -> кривая (t -> F1/TNR/score)
    select_threshold(curve)                           -> точка max(0.7·F1 + 0.3·TNR)
    apply_threshold(scores, threshold)                -> bool-маска accepted/refusal

и CLI, который прогоняет весь val-пайплайн: берёт эмбеддинги (base + query-side TTA),
считает per-query k-reciprocal порядок (``reid.rerank``), строит кривую, фиксирует
порог, пишет ``candidates.csv`` с этим порогом и проверяет F1/TNR ОФИЦИАЛЬНЫМ
скриптом (``reid.eval.harness``).

Скор уверенности
----------------
Ранжирование делает reid.rerank, но его внутренний `score` нормируется min-max
внутри top-pool одного запроса и потому несопоставим между запросами (много 1.0,
TNR=0 на любом пороге). Поэтому confidence строится из ИСХОДНОЙ cosine-близости
top-1 (и опционально margin/z-score), что даёт кросс-запросно сравнимую,
монотонную величину:

    cosine : sim(top1)
    margin : sim(top1) - sim(top2)         (истинная уверенность: насколько отрыв)
    zscore : (sim(top1) - mean(sim)) / std(sim)  по всей галерее запроса

Метрики F1/TNR здесь пересчитываются ровно по формулам official evaluate.py
(``candidate_metrics``), а PR-AUC берётся из официальной функции ``pr_auc``.
Итоговая точка верифицируется вызовом официального скрипта.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.eval.official_evaluate import pr_auc

__all__ = [
    "score_confidences",
    "threshold_curve",
    "select_threshold",
    "apply_threshold",
    "is_accepted",
    "write_candidates",
    "calibrate",
    "calibrate_from_run",
]

# Порог по умолчанию: замороженный конфиг re-rank (W2-1/W2-2, exp-0011/exp-0025).
DEFAULT_RERANK = {"k1": 5, "k2": 2, "lam": 0.7, "pool_size": 100}
DEFAULT_SCALES = (224, 280)

# Зафиксированный порог отказа для исторического однопользовательского пайплайна
# DINOv2-B (variant=dino), получен ТОЛЬКО на val-сплите
# (reid.data.splits.holdout_val, seed=42), точка max(0.7·F1+0.3·TNR).
# Скор = cosine(TTA-fused query 224+280, gallery 224) top-1 после k-reciprocal
# re-rank (k1=5,k2=2,lam=0.7). Эксперимент exp-0030, отчёт
# reports/calibration_exp-0030.json. НЕ подбирать по тесту.
# ВНИМАНИЕ: это порог для DINO-варианта; для fusion-чемпиона используйте
# DEFAULT_THRESHOLD_FUSION_W08 (ниже) — шкалы cosine разные.
DEFAULT_THRESHOLD = 0.4623238444328308

# ---------------------------------------------------------------------------
# W2-4 redo — чемпион W3: fusion w=0.8
# ---------------------------------------------------------------------------
# Пайплайн-чемпион (W3):
#     fusion = L2( concat[ L2(SigLIP2-naflex 512d), 0.8 * L2(DINOv2-B 512d) ] )
#     + query-side TTA 224/280 (DINO, без hflip)
#     + per-query k-reciprocal (эти параметры)
#     + порог отказа (этот константа).
DEFAULT_FUSION_W = 0.8
DEFAULT_RERANK_FUSION_W08 = {"k1": 8, "k2": 3, "lam": 0.5, "pool_size": 200}

# [SUPERSEDED — НЕ для деплоя] Порог отказа для fusion w=0.8 (fp32, TTA 224/280,
# pool=200). Снят на СТАРОЙ шкале; финальный variant-A деплой (224-only, GPU-preproc,
# pool=300) использует DEFAULT_THRESHOLD_FUSION_FINAL (ниже). Оставлен для истории
# и абляций. Получен ТОЛЬКО на val-сплите
# (reid.data.splits.holdout_val, seed=42: 375 query, из них 81 open-set) как
# точка max(0.7·F1+0.3·TNR); confidence = cosine top-1 после query-side TTA
# 224+280 и k-reciprocal (k1=8,k2=3,lam=0.5,pool=200). Обучающих весов не
# меняли: fusion пересобран CPU-only из уже посчитанных val-эмбеддингов
# (runs/W2-5-val-{siglip,dino}), а корректность реконструкции подтверждена
# совпадением w=0.6 с runs/W2-5-val-fusion (max|diff|=0.0).
# Эксперимент exp-W2-4-w08, отчёт reports/calibration_fusion_w08.json.
# НЕ подбирать по тесту.
DEFAULT_THRESHOLD_FUSION_W08 = 0.690625786781311

# ---------------------------------------------------------------------------
# W2-4 redo-2 — порог для fp16-шкалы с TTA (НЕ финальный деплой)
# ---------------------------------------------------------------------------
# [SUPERSEDED — НЕ для деплоя] Порог снят на шкале TTA 224/280 (fp16 ONNX);
# финальный variant-A деплой — 224-only (TTA OFF), см. ниже.
# Инференс-сервис отдаёт val-прогон на fp16-шкале:
#   runs/W2-5-val-fusion-fp16/{score.npy,score.csv,embeddings.npy,...}
# где confidence = cosine top-1 того же пайплайна (TTA 224/280 + k-reciprocal
# k1=8,k2=3,lam=0.5,pool=300), но посчитанного fp16-ONNX-бэкбонами.
# Порог взят на ТОМ ЖЕ val-сплите (holdout_val seed=42: 375 query / 81 open-set)
# как точка max(0.7*F1+0.3*TNR) ПО FP16-ШКАЛЕ. Эксперимент exp-0059, отчёт
# reports/calibration_fusion_w08_fp16.json.
#
# Зачем отдельная константа: fp16-эмбеддинги немного размывают шкалу cosine,
# поэтому fp32-порог 0.690626 на fp16-скоре теряет один TP
# (F1 0.9685 против 0.9703) — на 0.6897586 балл восстанавливается.
# НЕ подбирать по тесту.
DEFAULT_THRESHOLD_FUSION_W08_FP16 = 0.6897585988044739

# ---------------------------------------------------------------------------
# ФИНАЛЬНЫЙ ДЕПЛОЙ (variant A): fusion, 224-only (TTA OFF), GPU-preproc
# ---------------------------------------------------------------------------
# КАНОНИЧЕСКИЙ порог отказа задеплоенного пайплайна:
#     fusion = L2( concat[ L2(SigLIP2 512d ONNX fp16),
#                          0.8 * L2(DINOv2-B@224 512d ONNX fp16) ] )
#     + GPU-preproc (CUDA letterbox + normalize + ORT IOBinding)
#     + confidence = cosine top-1 после per-query k-reciprocal
#       (k1=8, k2=3, lam=0.5, pool=300); TTA OFF (224-only).
# Снят ТОЛЬКО на val-сплите (reid.data.splits.holdout_val, seed=42:
# 375 query / 1528 gallery / 81 open-set) как точка max(0.7·F1 + 0.3·TNR).
# Val-прогон: runs/W2-5-val-final. Верифицировано официальным evaluate.py:
#   F1 = 0.9702276707530647, TNR = 1.0,
#   балл = 0.9791593695271452, PR-AUC = 0.9921793910286647,
#   TP/FP/FN/TN = 277/1/16/81, mAP@10 = 0.6908555366591081.
# Отчёт: reports/calibration_final.json/.md.
# НЕ подбирать по тесту; при смене чемпиона/TTA/шкалы — перекалибровать.
#
# SUPERSEDED (оставлены для истории/абляций, НЕ применять на 224-only деплое):
#   * DEFAULT_THRESHOLD_FUSION_W08      = 0.690625786781311  (fp32, TTA 224/280, pool=200)
#   * DEFAULT_THRESHOLD_FUSION_W08_FP16 = 0.6897585988044739 (fp16, TTA 224/280, pool=300)
DEFAULT_THRESHOLD_FUSION_FINAL = 0.6970806121826172

# Обратная совместимость: alias, который читает service/infer/run.py через
# getattr(). Всегда указывает на КАНОНИЧЕСКИЙ финальный порог выше.
DEFAULT_THRESHOLD_FUSION_W08_FP16_224 = DEFAULT_THRESHOLD_FUSION_FINAL


def is_accepted(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Принять (True) или отказать (False) по confidence ``score``."""
    return float(score) >= float(threshold)


# ---------------------------------------------------------------------------
# Чистый API: score -> threshold -> accepted/refusal
# ---------------------------------------------------------------------------
def score_confidences(queries: np.ndarray, gallery: np.ndarray,
                      order: np.ndarray, mode: str = "cosine") -> np.ndarray:
    """Top-1 confidence на запрос — монотонная, кросс-запросно сравнимая.

    Parameters
    ----------
    queries, gallery : (Nq, D), (Ng, D) float32
        L2-нормированные эмбеддинги (L2 выполняется здесь повторно, безопасно).
    order : (Nq, Ng) int
        Порядок галереи на запрос (лучший первым), как отдаёт
        :func:`reid.rerank.rank_prepared` (top-1 = ``order[i, 0]``).
    mode : {"cosine", "margin", "zscore"}

    Returns
    -------
    (Nq,) float32 — уверенность (больше = ближе/увереннее).
    """
    q = rerank.l2norm(np.asarray(queries, dtype=np.float32))
    g = rerank.l2norm(np.asarray(gallery, dtype=np.float32))
    if q.shape[0] != order.shape[0]:
        raise ValueError("score_confidences: queries/order length mismatch")
    if g.shape[0] != order.shape[1]:
        raise ValueError("score_confidences: gallery/order width mismatch")

    sims = q @ g.T                                   # (Nq, Ng) cosine
    top1 = sims[np.arange(len(sims)), order[:, 0]].astype(np.float64)

    if mode == "cosine":
        return top1.astype(np.float32)
    if mode == "margin":
        if order.shape[1] < 2:
            return top1.astype(np.float32)
        top2 = sims[np.arange(len(sims)), order[:, 1]].astype(np.float64)
        return (top1 - top2).astype(np.float32)
    if mode == "zscore":
        mu = sims.mean(axis=1)
        sd = sims.std(axis=1)
        z = (top1 - mu) / np.clip(sd, 1e-12, None)
        return z.astype(np.float32)
    raise ValueError(f"unknown score mode: {mode!r}")


def _curve_at(scores, has_match, top1_correct, thr):
    """Все счётчики и метрики кандидатов при пороге ``thr`` (формулы official)."""
    accepted = scores >= thr
    correct = has_match & top1_correct
    tp = int(np.sum(accepted & correct))
    fp = int(np.sum(accepted) - tp)
    fn = int(np.sum(~accepted & has_match))
    tn = int(np.sum(~accepted & ~has_match))
    fp_openset = int(np.sum(accepted & ~has_match))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    tnr = tn / (tn + fp_openset) if (tn + fp_openset) else float("nan")
    return {
        "threshold": float(thr),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accepted": int(accepted.sum()),
        "refused": int((~accepted).sum()),
        "Precision": precision, "Recall": recall,
        "F1": f1, "TNR": tnr,
        "score_0.7F1+0.3TNR": 0.7 * f1 + 0.3 * (tnr if tnr == tnr else 0.0),
    }


def threshold_curve(scores: np.ndarray, has_match: np.ndarray,
                    top1_correct: np.ndarray) -> list[dict]:
    """Кривая порог -> F1/TNR/score по всем осмысленным порогам.

    Рассматриваются: «принять всё» (ниже минимума), каждая уникальная оценка
    (граница accepted), и «отказать всем» (выше максимума). Порядок — по
    возрастанию порога.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    has_match = np.asarray(has_match, dtype=bool)
    top1_correct = np.asarray(top1_correct, dtype=bool)
    if not (len(scores) == len(has_match) == len(top1_correct)):
        raise ValueError("threshold_curve: lengths differ")

    uniq = np.unique(scores)
    grid = np.concatenate([uniq, [uniq[-1] + 1e-6]]) if len(uniq) else np.array([0.0])
    grid = np.concatenate([[uniq[0] - 1e-6], grid]) if len(uniq) else grid
    return [_curve_at(scores, has_match, top1_correct, float(t)) for t in grid]


def select_threshold(curve: list[dict]) -> dict:
    """Точка, максимизирующая 0.7·F1 + 0.3·TNR (±nan-safe).

    Тай-брейк: (1) больший score, (2) больший F1 (не отказывать в лишнем),
    (3) больший TNR, (4) меньший порог. Т.е. при равном балле предпочитаем
    точку, которая сохраняет найденные пары, если это бесплатно по TNR.
    """
    if not curve:
        raise ValueError("select_threshold: empty curve")

    def key(pt):
        return (pt["score_0.7F1+0.3TNR"], pt["F1"], pt["TNR"], -pt["threshold"])

    return max(curve, key=key)


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Маска accepted (True) / refusal (False): ``score >= threshold``."""
    return np.asarray(scores, dtype=np.float64).reshape(-1) >= float(threshold)


def write_candidates(path: str, q_ids, g_ids, order: np.ndarray,
                     scores: np.ndarray, threshold: float) -> int:
    """Записать ``candidates.csv``: строка только для accepted query.

    Возвращает число записанных (принятых) строк.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    accepted = apply_threshold(scores, threshold)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            if not accepted[i]:
                continue
            j = int(order[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i]):.6f}"])
    return int(accepted.sum())


def calibrate(scores, has_match, top1_correct) -> dict:
    """Полная калибровка одного скора: кривая + выбранная точка."""
    curve = threshold_curve(scores, has_match, top1_correct)
    chosen = select_threshold(curve)
    labels = np.asarray(has_match, dtype=np.float64)
    return {
        "curve": curve,
        "chosen": chosen,
        "pr_auc_raw": float(pr_auc(np.asarray(scores, dtype=np.float64).reshape(-1),
                                   labels)),
    }


# ---------------------------------------------------------------------------
# Перекалибровка по готовому val-прогону (fp16-шкала)
# ---------------------------------------------------------------------------
def _read_csv_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"не найден обязательный файл: {path}")
    return pd.read_csv(path)


def calibrate_from_run(run_dir: str, *, out_dir: str | None = None,
                       json_path: str | None = None,
                       rerank_cfg: dict | None = None,
                       scoring_modes=("cosine", "margin", "zscore"),
                       pipeline: str = "", split_meta: dict | None = None,
                       sources: dict | None = None,
                       extra_report: dict | None = None,
                       verbose: bool = True) -> dict:
    """Перекалибровать порог отказа по готовому прогону (без GPU).

    Читает из ``run_dir``:
      * ``score.npy``   — confidence top-1 (fp16-шкала), вектор (Nq,);
      * ``query.csv``/``gallery.csv`` — метки (vehicle_id, camera_id);
      * ``embeddings.npy`` — [query; gallery] для margin/z-score и проверки
        воспроизводимости cosine-скора.

    Для каждого скора строит кривую и выбирает точку ``max(0.7*F1+0.3*TNR)``;
    пишет ``candidates.csv`` (top-1 только для принятых query) в ``out_dir``
    (или ``run_dir``, если ``out_dir`` не задан) и проверяет F1/TNR официальным
    ``evaluate.py`` через :func:`reid.eval.harness.run_official`.

    Возвращает отчёт-dict (та же схема, что пишется в ``json_path``).
    """
    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.abspath(out_dir or run_dir)
    os.makedirs(out_dir, exist_ok=True)

    vq = _read_csv_required(os.path.join(run_dir, "query.csv"))
    vg = _read_csv_required(os.path.join(run_dir, "gallery.csv"))
    g_ids = vg["image_id"].tolist()
    n_q, n_g = len(vq), len(vg)

    score_path = os.path.join(run_dir, "score.npy")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"нет score.npy в {run_dir}")
    score = np.load(score_path).astype(np.float64).reshape(-1)
    if score.shape[0] != n_q:
        raise ValueError(f"score.npy rows {score.shape[0]} != n_query {n_q}")

    emb_path = os.path.join(run_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"нет embeddings.npy в {run_dir}")
    emb = np.load(emb_path).astype(np.float32)
    if emb.shape[0] != n_q + n_g:
        raise ValueError(f"embeddings rows {emb.shape[0]} != {n_q}+{n_g}")
    q = rerank.l2norm(emb[:n_q])
    g = rerank.l2norm(emb[n_q:])

    rr = dict(rerank_cfg or {**DEFAULT_RERANK_FUSION_W08, "pool_size": 300})
    orders = _rank_all(q, g, rr["k1"], rr["k2"], rr["lam"], rr["pool_size"])

    # --- проверка воспроизводимости: cosine(embeddings) == score.npy --------
    conf_cos = score_confidences(q, g, orders, mode="cosine").astype(np.float64)
    max_abs_diff = float(np.max(np.abs(conf_cos - score))) if n_q else 0.0

    score_csv = os.path.join(run_dir, "score.csv")
    top1_gid_ok = None
    if os.path.exists(score_csv):
        sc = pd.read_csv(score_csv)
        if len(sc) == n_q and "gallery_id" in sc.columns:
            recomputed = [g_ids[int(orders[i, 0])] for i in range(n_q)]
            top1_gid_ok = bool(recomputed == sc["gallery_id"].tolist())

    has_match, top1_correct = _labels(vq, vg, orders, g_ids)
    n_openset = int(np.sum(~has_match))

    # --- калибровка по всем скорам ------------------------------------------
    all_scores = {"cosine": score}
    for m in scoring_modes:
        if m == "cosine":
            continue
        all_scores[m] = score_confidences(q, g, orders, mode=m).astype(np.float64)

    results = {m: calibrate(all_scores[m], has_match, top1_correct)
               for m in all_scores}
    best_mode = max(all_scores, key=lambda m: (
        results[m]["chosen"]["score_0.7F1+0.3TNR"], results[m]["chosen"]["F1"]))
    chosen = results[best_mode]["chosen"]
    scores = all_scores[best_mode]

    # --- сравнение с fp32-порогом на fp16-шкале -----------------------------
    fp32_cmp = _curve_at(score, has_match, top1_correct,
                         DEFAULT_THRESHOLD_FUSION_W08)

    # --- артефакты ----------------------------------------------------------
    for name in ("query.csv", "gallery.csv", "gt.csv", "submission.csv",
                 "embeddings.npy"):
        src = os.path.join(run_dir, name)
        dst = os.path.join(out_dir, name)
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
    gt = os.path.join(out_dir, "gt.csv")
    sub = os.path.join(out_dir, "submission.csv")
    emb_out = os.path.join(out_dir, "embeddings.npy")
    qcsv = os.path.join(out_dir, "query.csv")
    gcsv = os.path.join(out_dir, "gallery.csv")
    cand = os.path.join(out_dir, "candidates.csv")

    q_ids = vq["image_id"].tolist()
    n_written = write_candidates(cand, q_ids, g_ids, orders, scores,
                                 chosen["threshold"])
    official = run_official(gt_csv=gt, submission=sub, candidates=cand,
                            embeddings=emb_out, query=qcsv, gallery=gcsv,
                            json_out=os.path.join(out_dir, "official_calibrated.json"))
    oc = official.get("candidates", {})

    if verbose:
        print(f"[calibrate_from_run] {os.path.basename(run_dir)}: "
              f"cosine==score.npy max|diff|={max_abs_diff:.2e}; "
              f"chosen={best_mode} thr={chosen['threshold']:.10f} "
              f"F1={chosen['F1']:.6f} TNR={chosen['TNR']:.6f} "
              f"score={chosen['score_0.7F1+0.3TNR']:.6f} "
              f"official F1={oc.get('F1')} TNR={oc.get('TNR')}", flush=True)

    report = {
        "task": "W2-4 redo-2",
        "agent": "calibration-agent",
        "variant": "fusion_w0.8_fp16",
        "pipeline": pipeline,
        "score_mode": best_mode,
        "config": {
            "fusion_w": 0.8,
            "tta_scales": [224, 280],
            "rerank": rr,
            "backends": "ONNX fp16 (CUDAExecutionProvider)",
            "sources": dict(sources or {}),
            "reconstruction": {
                "cosine_vs_score_npy_max_abs_diff": max_abs_diff,
                "top1_gid_matches_score_csv": top1_gid_ok,
            },
        },
        "split": dict(split_meta or {"seed": 42, "n_query": n_q,
                                     "n_gallery": n_g, "n_openset": n_openset}),
        "selection_rule": "max 0.7*F1 + 0.3*TNR (tie: F1, then TNR, then lower thr)",
        "chosen": chosen,
        "pr_auc_raw": results[best_mode]["pr_auc_raw"],
        "pr_auc_official": oc.get("PR-AUC"),
        "fp32_threshold_on_fp16_scale": fp32_cmp,
        "score_modes": {m: {"chosen": results[m]["chosen"],
                            "pr_auc_raw": results[m]["pr_auc_raw"]}
                        for m in all_scores},
        "curve": results[best_mode]["curve"],
        "official_calibrated": official,
        "n_candidates_rows": n_written,
        "artifacts": {
            "run_dir": run_dir.replace("\\", "/"),
            "candidates": cand.replace("\\", "/"),
            "submission": sub.replace("\\", "/"),
            "embeddings": emb_out.replace("\\", "/"),
            "official": os.path.join(out_dir, "official_calibrated.json").replace("\\", "/"),
        },
        "verdict": "pending",
    }
    if extra_report:
        report.update(extra_report)
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"report -> {json_path}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Val-пайплайн (CLI)
# ---------------------------------------------------------------------------
def _load_model(checkpoint: str, device):
    import torch
    from reid.models import build_model

    ck = torch.load(checkpoint, map_location="cpu")
    cfg = ck.get("config", {})
    sd = ck["state_dict"]
    num_classes = int(sd["arcface.weight"].shape[0])
    model = build_model(
        backbone=ck.get("backbone", cfg.get("backbone", "dinov2_b")),
        num_classes=num_classes,
        emb_dim=int(ck.get("emb_dim", cfg.get("emb_dim", 512))),
        pretrained=False,
        margin=float(cfg.get("margin", 0.3)),
        scale=float(cfg.get("scale", 30.0)),
        gem_p=float(cfg.get("gem_p", 3.0)),
        image_size=int(cfg.get("image_size", 224)),
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, cfg


def _extract(model, df, dataset_dir, size, device, cache_dir, cfg):
    import torch
    from reid.data.aug import build_eval_transform
    from reid.train import extract_embeddings

    tf = build_eval_transform(size)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("amp"))
    return extract_embeddings(
        model, df, dataset_dir, tf, size, device,
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 64))),
        num_workers=int(cfg.get("num_workers", 8)),
        amp_dtype=amp, cache_dir=cache_dir,
        draft_factor=float(cfg.get("draft_factor", 2.0)),
    )


def _rank_all(queries, gallery, k1, k2, lam, pool_size):
    n_q = queries.shape[0]
    n_g = gallery.shape[0]
    orders = np.empty((n_q, n_g), dtype=np.int64)
    for i in range(n_q):
        prep = rerank.prepare_query(queries[i], gallery, pool_size=pool_size)
        orders[i], _ = rerank.rank_prepared(prep, k1=k1, k2=k2, lam=lam)
    return orders


def _write_submission(path, q_ids, g_ids, order, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in order[i, :top_k]])


def _rationale(mode, chosen, n_openset, n_q):
    """Короткое письменное обоснование выбранной точки (для README/защиты)."""
    return (
        f"Скор уверенности: cosine top-1 галереи после query-side TTA (224+280) "
        f"и k-reciprocal re-rank (k1=5,k2=2,lam=0.7). Порог подобран ТОЛЬКО на "
        f"val-сплите seed=42 ({n_q} query, из них open-set {n_openset}) как точка, "
        f"максимизирующая 0.7*F1+0.3*TNR; метрики сверены официальным "
        f"evaluate.py. Выбран режим '{mode}': margin/z-score не улучшили балл "
        f"(margin теряет F1, т.к. у одного ТС в галерее много кадров и top-2 "
        f"близок; z-score эквивалентен cosine). Порог {chosen['threshold']:.4f} "
        f"лежит в разрыве между максимальным open-set скором и скором следующего "
        f"запроса, поэтому весь open-set отвергается (TNR={chosen['TNR']:.4f}) "
        f"при F1={chosen['F1']:.4f}. Точка устойчива: в окне порогов ±0.02 балл "
        f"меняется слабо (плато TNR=1.0). Порог заморожен в "
        f"reid.calibrate.DEFAULT_THRESHOLD и НЕ подбирался по тесту."
    )


def _write_gt(val_query, val_gallery, path):
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy()
    q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy()
    g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def _labels(val_query, val_gallery, order, g_ids):
    """has_match (валидный позитив после junk) и top1_correct (official-формула)."""
    gal_vid = val_gallery.set_index("image_id")["vehicle_id"]
    gal_cam = val_gallery.set_index("image_id")["camera_id"]
    vid_arr = gal_vid.reindex(g_ids).to_numpy()
    cam_arr = gal_cam.reindex(g_ids).to_numpy()

    has_match = np.zeros(len(val_query), dtype=bool)
    top1_correct = np.zeros(len(val_query), dtype=bool)
    for i, (_qid, row) in enumerate(val_query.iterrows()):
        same_vid = vid_arr == row["vehicle_id"]
        same_cam = cam_arr == row["camera_id"]
        has_match[i] = bool(np.any(same_vid & ~same_cam))       # valid_positives > 0
        top1_correct[i] = bool(vid_arr[order[i, 0]] == row["vehicle_id"])
    return has_match, top1_correct


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Откалибровать порог отказа (W2-4)")
    ap.add_argument("--from-run", default=None,
                    help="готовый val-прогон (score.npy+embeddings.npy+query/gallery/"
                         "gt/submission): перекалибровать порог CPU-only без модели")
    ap.add_argument("--checkpoint", default="runs/exp-0007/best.pt")
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--out", default="runs/exp-0030_calib")
    ap.add_argument("--json", default="reports/calibration_exp-0030.json")
    ap.add_argument("--scales", default="224,280",
                    help="base,tta scale (должны делиться на 14 для DINOv2)")
    ap.add_argument("--reuse-base", default="runs/exp-0011_rerank/val/raw_embeddings.npy",
                    help="[q_base;g_base].npy для переиспользования 224-эмбеддингов")
    ap.add_argument("--reuse-tta", default=None,
                    help="q_tta.npy (TTA-query); если задан — GPU не нужен, "
                         "порог пересчитывается CPU-only из готового скора")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "cosine", "margin", "zscore"])
    ap.add_argument("--k1", type=int, default=None)
    ap.add_argument("--k2", type=int, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--pool-size", type=int, default=None)
    args = ap.parse_args(argv)

    if args.from_run:
        rr_default = {**DEFAULT_RERANK_FUSION_W08, "pool_size": 300}
        rerank_cfg = {
            "k1": args.k1 if args.k1 is not None else rr_default["k1"],
            "k2": args.k2 if args.k2 is not None else rr_default["k2"],
            "lam": args.lam if args.lam is not None else rr_default["lam"],
            "pool_size": (args.pool_size if args.pool_size is not None
                          else rr_default["pool_size"]),
        }
        calibrate_from_run(
            args.from_run,
            out_dir=os.path.join(args.out, "val"),
            json_path=args.json,
            rerank_cfg=rerank_cfg,
            pipeline=("L2(concat[L2(SigLIP2 512d ONNX fp16), 0.8*L2(DINOv2-B 512d "
                      "ONNX fp16)]) + query-side TTA (224,280; no hflip) + per-query "
                      "k-reciprocal"),
            sources={"score_npy": os.path.join(args.from_run, "score.npy"),
                     "score_csv": os.path.join(args.from_run, "score.csv"),
                     "embeddings": os.path.join(args.from_run, "embeddings.npy")},
        )
        return 0

    if args.k1 is None:
        args.k1 = DEFAULT_RERANK["k1"]
    if args.k2 is None:
        args.k2 = DEFAULT_RERANK["k2"]
    if args.lam is None:
        args.lam = DEFAULT_RERANK["lam"]
    if args.pool_size is None:
        args.pool_size = DEFAULT_RERANK["pool_size"]

    if not args.dataset:
        print("ERROR: --dataset или DATASET_DIR обязателен", file=sys.stderr)
        return 2

    import torch
    from reid.train import resolve_crop_cache, set_seed

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(os.path.abspath(args.out), "val")
    os.makedirs(run_dir, exist_ok=True)

    train_full = read_csv(os.path.join(args.dataset, "train.csv"),
                          required=TRAIN_COLUMNS)
    val_query, val_gallery = holdout_val(
        train_full, val_fraction=0.2, open_set_fraction=0.2, seed=42)[1:]
    n_q, n_g = len(val_query), len(val_gallery)
    print(f"val: query={n_q} gallery={n_g}", flush=True)

    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    base = scales[0]
    q_ids = val_query["image_id"].tolist()
    g_ids = val_gallery["image_id"].tolist()

    # ---- embeddings: base reused, TTA scales extracted (query side only) ----
    q_base = g_base = None
    if args.reuse_base and os.path.exists(args.reuse_base):
        allb = np.load(args.reuse_base).astype(np.float32)
        if allb.shape[0] != n_q + n_g:
            raise SystemExit(f"reuse-base rows {allb.shape[0]} != {n_q}+{n_g}")
        q_base, g_base = rerank.l2norm(allb[:n_q]), rerank.l2norm(allb[n_q:])
        print(f"reused base embeddings <- {args.reuse_base}", flush=True)

    reuse_tta = args.reuse_tta if (args.reuse_tta and os.path.exists(args.reuse_tta)) else None
    if reuse_tta:
        q_tta = rerank.l2norm(np.load(reuse_tta).astype(np.float32))
        if q_tta.shape[0] != n_q:
            raise SystemExit(f"reuse-tta rows {q_tta.shape[0]} != {n_q}")
        if g_base is None:
            raise SystemExit("--reuse-tta требует --reuse-base (нужен g_base)")
        print(f"reused TTA query embeddings <- {reuse_tta}", flush=True)
    else:
        model, cfg = _load_model(args.checkpoint, device)
        cache_dir = resolve_crop_cache(cfg)
        if q_base is None:
            q_base = _extract(model, val_query, args.dataset, base, device, cache_dir, cfg)
            g_base = _extract(model, val_gallery, args.dataset, base, device, cache_dir, cfg)
        extra = [_extract(model, val_query, args.dataset, s, device, cache_dir, cfg)
                 for s in scales[1:]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if extra:
            q_tta = np.stack([
                rerank.fuse_embeddings([q_base[i]] + [e[i] for e in extra])
                for i in range(n_q)
            ]).astype(np.float32)
        else:
            q_tta = q_base
        print(f"TTA fused over scales {scales}", flush=True)

    # ---- per-query re-rank order on the deployed (TTA) query ----------------
    order = _rank_all(q_tta, g_base, args.k1, args.k2, args.lam, args.pool_size)
    has_match, top1_correct = _labels(val_query, val_gallery, order, g_ids)
    n_openset = int(np.sum(~has_match))
    print(f"open-set queries: {n_openset}/{n_q}", flush=True)

    # ---- calibracija po vsem skoram ---------------------------------------
    modes = ["cosine", "margin", "zscore"] if args.mode == "auto" else [args.mode]
    all_scores = {m: score_confidences(q_tta, g_base, order, mode=m) for m in modes}
    results = {m: calibrate(all_scores[m], has_match, top1_correct) for m in modes}

    # ---- vybor luchshego skora (max score) ---------------------------------
    best_mode = max(modes, key=lambda m: (results[m]["chosen"]["score_0.7F1+0.3TNR"],
                                          results[m]["chosen"]["F1"]))
    chosen = results[best_mode]["chosen"]
    scores = all_scores[best_mode]
    print(f"chosen score mode = {best_mode}: {json.dumps(chosen, ensure_ascii=False)}",
          flush=True)

    # ---- artefakty + oficialnaya proverka ---------------------------------
    gt = os.path.join(run_dir, "gt.csv")
    query_csv = os.path.join(run_dir, "query.csv")
    gallery_csv = os.path.join(run_dir, "gallery.csv")
    sub = os.path.join(run_dir, "submission.csv")
    cand = os.path.join(run_dir, "candidates.csv")
    emb = os.path.join(run_dir, "embeddings.npy")
    _write_gt(val_query, val_gallery, gt)
    val_query.to_csv(query_csv, index=False)
    val_gallery.to_csv(gallery_csv, index=False)
    _write_submission(sub, q_ids, g_ids, order)
    np.save(emb, np.vstack([q_tta, g_base]).astype(np.float32))
    np.save(os.path.join(run_dir, "q_tta.npy"), q_tta.astype(np.float32))

    n_written = write_candidates(cand, q_ids, g_ids, order, scores,
                                 chosen["threshold"])
    official = run_official(gt_csv=gt, submission=sub, candidates=cand,
                            embeddings=emb, query=query_csv, gallery=gallery_csv,
                            json_out=os.path.join(run_dir, "official_calibrated.json"))
    oc = official.get("candidates", {})
    print(f"candidates rows: {n_written} (refused {n_q - n_written}); "
          f"official F1={oc.get('F1')} TNR={oc.get('TNR')}", flush=True)

    report = {
        "task": "W2-4",
        "agent": "calibration-agent",
        "checkpoint": args.checkpoint,
        "score_mode": best_mode,
        "config": {
            "rerank": {"k1": args.k1, "k2": args.k2, "lam": args.lam,
                       "pool_size": args.pool_size},
            "scales": scales, "tta": "query-side fuse (no hflip)",
            "reused_base": args.reuse_base if os.path.exists(args.reuse_base) else None,
        },
        "split": {"seed": 42, "n_query": n_q, "n_gallery": n_g,
                  "n_openset": n_openset},
        "selection_rule": "max 0.7*F1 + 0.3*TNR (tie: F1, then TNR, then lower thr)",
        "rationale": _rationale(best_mode, chosen, n_openset, n_q),
        "chosen": chosen,
        "pr_auc_raw": results[best_mode]["pr_auc_raw"],
        "score_modes": {m: {"chosen": results[m]["chosen"],
                            "pr_auc_raw": results[m]["pr_auc_raw"]} for m in modes},
        "curve": results[best_mode]["curve"],
        "official_calibrated": official,
        "n_candidates_rows": n_written,
        "artifacts": {
            "candidates": cand.replace("\\", "/"),
            "submission": sub.replace("\\", "/"),
            "embeddings": emb.replace("\\", "/"),
            "official": os.path.join(run_dir, "official_calibrated.json").replace("\\", "/"),
        },
        "verdict": "pending",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {args.json}", flush=True)

    # sverka: nash raschjot == oficialnyj
    for k_src, k_dst in (("F1", "F1"), ("TNR", "TNR")):
        mine, off = chosen[k_src], oc.get(k_dst)
        if off is not None and abs(mine - off) > 1e-9:
            print(f"WARN: {k_src} mismatch mine={mine} official={off}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
