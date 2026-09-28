"""Unit-тесты чистого API калибровки порога отказа (W2-4).

Метрики F1/TNR пересчитываются ровно по формулам official evaluate.py; здесь
проверяются score → threshold → accepted/refusal и выбор точки.
"""
from __future__ import annotations

import numpy as np

from reid.calibrate import (
    DEFAULT_THRESHOLD,
    apply_threshold,
    calibrate,
    is_accepted,
    score_confidences,
    threshold_curve,
    write_candidates,
)


def _toy_embeddings():
    # 3 запроса: q0/q1 закрытые (top-1 свой), q2 open-set (нет валидного позитива)
    q = np.eye(3, dtype=np.float32)
    g = np.eye(3, dtype=np.float32)
    order = np.tile(np.arange(3), (3, 1))
    return q, g, order


def test_score_modes_shapes_and_monotone():
    q, g, order = _toy_embeddings()
    cos = score_confidences(q, g, order, "cosine")
    assert cos.shape == (3,)
    assert cos[0] > cos[1]  # q0 ближе к gallery[0], чем q1
    assert score_confidences(q, g, order, "margin").shape == (3,)
    assert score_confidences(q, g, order, "zscore").shape == (3,)


def test_curve_matches_manual_counts():
    scores = np.array([0.9, 0.4, 0.1])
    has_match = np.array([True, True, False])
    top1_correct = np.array([True, False, False])
    curve = threshold_curve(scores, has_match, top1_correct)

    # порог 0.9 (первый >= 0.5): принят только q0 -> TP=1, FP=0, FN=1, TN=1
    pt = min((c for c in curve if c["threshold"] >= 0.5),
             key=lambda c: c["threshold"])
    assert (pt["TP"], pt["FP"], pt["FN"], pt["TN"]) == (1, 0, 1, 1)
    assert abs(pt["F1"] - 2 / 3) < 1e-9 and pt["TNR"] == 1.0

    # порог выше максимума: отказ всем -> F1=0, TNR=1
    top = max(curve, key=lambda c: c["threshold"])
    assert top["F1"] == 0.0 and top["TNR"] == 1.0


def test_select_threshold_and_apply():
    # уникальный максимум при 0.7: приняты q0,q1 (оба верны), open-set q2 отвергнут
    scores = np.array([0.9, 0.7, 0.1])
    has_match = np.array([True, True, False])
    top1_correct = np.array([True, True, False])
    chosen = calibrate(scores, has_match, top1_correct)["chosen"]
    assert abs(chosen["threshold"] - 0.7) < 1e-9
    assert chosen["F1"] == 1.0 and chosen["TNR"] == 1.0
    acc = apply_threshold(scores, chosen["threshold"])
    assert bool(acc[0]) and bool(acc[1]) and not bool(acc[2])


def test_write_candidates_refusal_omits_rows(tmp_path):
    q, g, order = _toy_embeddings()
    path = tmp_path / "candidates.csv"
    n = write_candidates(str(path), ["q0", "q1", "q2"], ["x", "y", "z"],
                         order, np.array([0.9, 0.4, 0.1]), threshold=0.5)
    assert n == 1
    rows = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert rows[0] == "query_id,gallery_id,confidence"
    assert len(rows) == 2  # заголовок + единственная принятая строка q0
    assert rows[1].startswith("q0,")
    assert all(not r.startswith(("q1,", "q2,")) for r in rows[1:])


def test_default_threshold_is_frozen():
    assert 0.0 < DEFAULT_THRESHOLD < 1.0
    assert is_accepted(DEFAULT_THRESHOLD)
    assert not is_accepted(DEFAULT_THRESHOLD - 0.01)
