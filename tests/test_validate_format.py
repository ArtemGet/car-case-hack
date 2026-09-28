"""pytest для tools/validate_format.py.

Покрытие:
  * example_submission организаторов (8 gallery, кандидатов не 10) — валиден;
  * синтетика в tests/fixtures/valid — валидна;
  * плохие варианты: 9 кандидатов при галерее >=10, дубли, неизвестный
    gallery_id, пропущенный query, пустой gallery_id в candidates, битая
    форма embeddings, отказ (нет строки) — не ошибка.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import make_variant

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
VALIDATOR = os.path.join(ROOT, "tools", "validate_format.py")
# Самодостаточная фикстура в git (docs/ в сдачу не входит).
EXAMPLE = os.path.join(TESTS_DIR, "fixtures", "example_submission")


def run_validator(out_dir, query, gallery):
    proc = subprocess.run(
        [sys.executable, VALIDATOR, "--out", out_dir,
         "--query", query, "--gallery", gallery, "--json"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    rep = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return proc.returncode, rep


def _run_fixture(out_dir):
    return run_validator(
        out_dir,
        os.path.join(out_dir, "test_query.csv"),
        os.path.join(out_dir, "test_gallery.csv"),
    )


def _rewrite_submission(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


# --------------------------- example_submission ---------------------------

def test_example_submission_is_valid():
    code, rep = run_validator(
        EXAMPLE,
        os.path.join(EXAMPLE, "test_query.csv"),
        os.path.join(EXAMPLE, "test_gallery.csv"))
    assert code == 0, rep
    assert rep["ok"] is True
    names = {c["name"]: c for c in rep["checks"]}
    assert names["submission.csv"]["level"] == "ok"
    assert names["embeddings.npy"]["level"] == "ok"


def test_example_has_8_gallery_and_not_10_candidates():
    with open(os.path.join(EXAMPLE, "test_gallery.csv"), encoding="utf-8-sig") as f:
        n_gallery = sum(1 for _ in csv.reader(f)) - 1
    assert n_gallery == 8
    with open(os.path.join(EXAMPLE, "submission.csv"), encoding="utf-8-sig") as f:
        first = next(csv.reader(f))
    assert len(first) - 1 == 8 != 10


def test_example_embeddings_shape_matches():
    code, rep = run_validator(
        EXAMPLE,
        os.path.join(EXAMPLE, "test_query.csv"),
        os.path.join(EXAMPLE, "test_gallery.csv"))
    emb = [c for c in rep["checks"] if c["name"] == "embeddings.npy"][0]
    assert emb["level"] == "ok"
    assert "13" in emb["detail"] and "float32" in emb["detail"]


# ------------------------------- valid synthetic -------------------------------

def test_valid_fixture_passes(valid_dir):
    code, rep = _run_fixture(valid_dir)
    assert code == 0, rep
    assert rep["ok"] is True


def test_refusal_missing_row_is_not_error(valid_dir):
    # q2 отсутствует в candidates.csv — это отказ, не ошибка
    code, rep = _run_fixture(valid_dir)
    assert code == 0
    cand = [c for c in rep["checks"] if c["name"] == "candidates.csv"][0]
    assert cand["level"] == "ok"
    assert "отказ" in cand["detail"]


# ------------------------------- bad variants -------------------------------

def test_candidate_count_must_be_10(tmp_path):
    def mutate(d):
        gallery = [f"g{i:02d}" for i in range(12)]
        _rewrite_submission(os.path.join(d, "submission.csv"),
                            [["q1"] + gallery[:9], ["q2"] + gallery[:9]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("ожидалось 10" in e for e in rep["errors"])


def test_duplicate_candidate_fails(tmp_path):
    def mutate(d):
        gallery = [f"g{i:02d}" for i in range(12)]
        _rewrite_submission(os.path.join(d, "submission.csv"),
                            [["q1"] + gallery[:9] + [gallery[0]],
                             ["q2"] + gallery[:10]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("дубли" in e for e in rep["errors"])


def test_unknown_gallery_id_fails(tmp_path):
    def mutate(d):
        gallery = [f"g{i:02d}" for i in range(12)]
        _rewrite_submission(os.path.join(d, "submission.csv"),
                            [["q1"] + gallery[:9] + ["zzz"],
                             ["q2"] + gallery[:10]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("неизвестные gallery_id" in e for e in rep["errors"])


def test_missing_query_fails(tmp_path):
    def mutate(d):
        gallery = [f"g{i:02d}" for i in range(12)]
        _rewrite_submission(os.path.join(d, "submission.csv"),
                            [["q1"] + gallery[:10]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("отсутствуют query" in e for e in rep["errors"])


def test_embeddings_wrong_rows_fails(tmp_path):
    def mutate(d):
        np.save(os.path.join(d, "embeddings.npy"),
                np.zeros((5, 8), dtype=np.float32))
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("строк, ожидалось" in e for e in rep["errors"])


def test_empty_gallery_id_in_candidates_fails(tmp_path):
    def mutate(d):
        _rewrite_submission(
            os.path.join(d, "candidates.csv"),
            [["query_id", "gallery_id", "confidence"],
             ["q1", "g00", "0.9"],
             ["q2", "", "0.5"]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("пустых gallery_id" in e for e in rep["errors"])


def test_bad_confidence_fails(tmp_path):
    def mutate(d):
        _rewrite_submission(
            os.path.join(d, "candidates.csv"),
            [["query_id", "gallery_id", "confidence"],
             ["q1", "g00", "not_a_number"]])
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("confidence" in e for e in rep["errors"])


def test_missing_candidates_file_fails(tmp_path):
    def mutate(d):
        os.remove(os.path.join(d, "candidates.csv"))
    out = make_variant(tmp_path, mutate)
    code, rep = _run_fixture(out)
    assert code == 1
    assert any("candidates.csv" in e for e in rep["errors"])
