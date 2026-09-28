"""Metric harness — обёртка над официальным `evaluate.py` организаторов.

ВАЖНО: метрики считаются ТОЛЬКО официальным скриптом
(`reid/eval/official_evaluate.py`). Этот модуль не переопределяет и не
пересчитывает ни одну метрику — он лишь вызывает официальный скрипт через
subprocess и отдаёт его отчёт (JSON или stdout) в виде dict.

Публичный API:
    run_official(gt_csv, submission, candidates=None, embeddings=None,
                 query=None, gallery=None, json_out=None,
                 top_k=None, timeout=None) -> dict
    load_report(path) -> dict

Пример:
    from reid.eval.harness import run_official
    rep = run_official(
        gt_csv="runs/val_gt.csv",
        submission="out/submission.csv",
        candidates="out/candidates.csv",
        embeddings="out/embeddings.npy",
        query="in/test_query.csv",
        gallery="in/test_gallery.csv",
        json_out="reports/eval.json",
    )
    print(rep["ranking"]["mAP@10"])
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(HERE, "official_evaluate.py")


def _build_cmd(gt_csv, submission, candidates, embeddings, query, gallery,
               json_out, top_k):
    cmd = [sys.executable, OFFICIAL, "--gt", str(gt_csv),
           "--submission", str(submission)]
    if candidates:
        cmd += ["--candidates", str(candidates)]
    if embeddings:
        cmd += ["--embeddings", str(embeddings)]
    if query:
        cmd += ["--query", str(query)]
    if gallery:
        cmd += ["--gallery", str(gallery)]
    if top_k is not None:
        cmd += ["--top-k", str(int(top_k))]
    if json_out:
        cmd += ["--json", str(json_out)]
    return cmd


def _parse_stdout(text: str) -> dict:
    """Достаёт метрики из stdout официального скрипта, если --json не задан.

    Возвращает ту же структуру, что и `evaluate.py --json`:
    {"ranking": {...}, "full_ranking": {...}, "candidates": {...}}.
    """
    ranking: dict = {}
    full: dict = {}
    cands: dict = {}

    def f(pat, s=text):
        m = re.search(pat, s)
        return float(m.group(1)) if m else None

    m = re.search(r"mAP@(\d+)\s*:\s*([0-9.]+)", text)
    if m:
        ranking[f"mAP@{m.group(1)}"] = float(m.group(2))
    r1 = f(r"Rank-1\s*:\s*([0-9.]+)")
    r5 = f(r"Rank-5\s*:\s*([0-9.]+)")
    if r1 is not None:
        ranking["Rank-1"] = r1
    if r5 is not None:
        ranking["Rank-5"] = r5
    nsc = f(r"запросов в зачёте\s*:\s*(\d+)")
    if nsc is not None:
        ranking["n_scored"] = int(nsc)
    nos = f(r"open-set вне mAP\s*:\s*(\d+)")
    if nos is not None:
        ranking["n_openset_excluded"] = int(nos)

    mf = f(r"mAP \(полный\)\s*:\s*([0-9.]+)")
    mi = f(r"mINP\s*:\s*([0-9.]+)")
    if mf is not None:
        full["mAP_full"] = mf
    if mi is not None:
        full["mINP"] = mi

    counts = re.search(r"TP/FP/FN/TN\s*:\s*(\d+)/(\d+)/(\d+)/(\d+)", text)
    if counts:
        cands["TP"], cands["FP"], cands["FN"], cands["TN"] = (int(x) for x in counts.groups())
    for key, pat in (("Precision", r"Precision\s*:\s*([0-9.]+)"),
                     ("Recall", r"Recall\s*:\s*([0-9.]+)"),
                     ("F1", r"F1\s*:\s*([0-9.]+)"),
                     ("TNR", r"TNR\s*:\s*([0-9.]+)"),
                     ("PR-AUC", r"PR-AUC\s*:\s*([0-9.]+)")):
        v = f(pat)
        if v is not None:
            cands[key] = v

    out: dict = {}
    if ranking:
        out["ranking"] = ranking
    if full:
        out["full_ranking"] = full
    if cands:
        out["candidates"] = cands
    return out


def run_official(gt_csv, submission, candidates=None, embeddings=None,
                 query=None, gallery=None, json_out=None,
                 top_k=None, timeout=None) -> dict:
    """Запустить официальный `evaluate.py` и вернуть его отчёт как dict.

    Метрики НЕ пересчитываются здесь — источник истины только официальный
    скрипт. При заданном `json_out` результат читается из JSON-файла (полный
    отчёт); иначе парсится stdout.

    Raises:
        FileNotFoundError: если не найден официальный скрипт.
        RuntimeError: если официальный скрипт завершился с ненулевым кодом.
    """
    if not os.path.exists(OFFICIAL):
        raise FileNotFoundError(f"официальный evaluate.py не найден: {OFFICIAL}")
    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"ground truth не найден: {gt_csv}")
    if not os.path.exists(submission):
        raise FileNotFoundError(f"submission не найден: {submission}")
    if embeddings and not (query and gallery):
        raise ValueError("embeddings требует query и gallery")

    cmd = _build_cmd(gt_csv, submission, candidates, embeddings, query,
                     gallery, json_out, top_k)
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            "официальный evaluate.py завершился с ошибкой "
            f"(код {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    if json_out and os.path.exists(json_out):
        return load_report(json_out)
    return _parse_stdout(proc.stdout or "")


def load_report(path) -> dict:
    """Прочитать JSON-отчёт, сохранённый официальным `evaluate.py --json`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
