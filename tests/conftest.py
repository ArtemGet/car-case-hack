"""Общие pytest-фикстуры и генератор синтетических фикстур формата сдачи.

Канонический валидный набор живёт в `tests/fixtures/valid/` и содержит:
  test_query.csv(2) + test_gallery.csv(12) -> embeddings.npy (14, 8) float32.
Из него тесты делают «плохие» варианты в tmp_path.
"""
from __future__ import annotations

import csv
import os
import shutil

import numpy as np
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(TESTS_DIR, "fixtures")
VALID = os.path.join(FIXTURES, "valid")


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def build_valid(root):
    """Создаёт валидный набор из 12 gallery и 2 query."""
    os.makedirs(root, exist_ok=True)
    gallery = [f"g{i:02d}" for i in range(12)]
    queries = ["q1", "q2"]

    _write_csv(os.path.join(root, "test_query.csv"),
               [["image_id", "x", "y", "w", "h"]] + [[q, 0, 0, 10, 10] for q in queries])
    _write_csv(os.path.join(root, "test_gallery.csv"),
               [["image_id", "x", "y", "w", "h"]] + [[g, 0, 0, 10, 10] for g in gallery])

    # ровно 10 кандидатов (галерея >= 10)
    _write_csv(os.path.join(root, "submission.csv"),
               [[q] + gallery[:10] for q in queries])

    _write_csv(os.path.join(root, "candidates.csv"),
               [["query_id", "gallery_id", "confidence"],
                ["q1", "g00", "0.91"]])  # q2 отсутствует = отказ

    emb = np.random.default_rng(0).standard_normal((len(queries) + len(gallery), 8))
    np.save(os.path.join(root, "embeddings.npy"), emb.astype(np.float32))
    return root


def make_variant(tmp_path, mutate):
    """Копирует валидный набор, применяет mutate(dir) и возвращает каталог."""
    dst = tmp_path / "variant"
    shutil.copytree(VALID, dst)
    mutate(str(dst))
    return str(dst)


@pytest.fixture(scope="session")
def valid_dir():
    # materialize canonical fixtures on disk under tests/fixtures/valid
    return build_valid(VALID)
