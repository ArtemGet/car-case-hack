"""W2-5: contract-level tests for the offline batch runner (CPU, no weights).

Exercises :func:`service.infer.run.rank_and_write` on synthetic embeddings and
asserts the three-artefact contract (INTERFACES.md §2) plus determinism.
No GPU, no model files, no network.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from service.infer import run as runner
from tools.validate_format import validate

QUERY_IDS = [f"q{i}" for i in range(5)]
GALLERY_IDS = [f"g{i}" for i in range(12)]


def _write_split(path, ids):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "x", "y", "w", "h"])
        for i in ids:
            w.writerow([i, 0, 0, 10, 10])


def _embeddings(seed=0, n=5, g=12, d=16):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, d)).astype(np.float32)
    gr = rng.standard_normal((g, d)).astype(np.float32)
    return runner.rerank.l2norm(q), runner.rerank.l2norm(gr)


def test_rank_and_write_contract(tmp_path):
    q, g = _embeddings()
    out = tmp_path / "out"
    info = runner.rank_and_write(str(out), q, g, QUERY_IDS, GALLERY_IDS,
                                 variant="dino", threshold=0.0)

    # exactly the three artefacts
    assert sorted(os.listdir(out)) == ["candidates.csv", "embeddings.npy",
                                       "submission.csv"]

    # submission: no header, query_id + 10 gids, no dups
    with open(out / "submission.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == len(QUERY_IDS)
    for row in rows:
        assert len(row) == 11
        assert len(set(row[1:])) == 10
        assert all(x in GALLERY_IDS for x in row[1:])

    # embeddings: float32, query file order then gallery, D unchanged
    emb = np.load(out / "embeddings.npy")
    assert emb.dtype == np.float32
    assert emb.shape == (len(QUERY_IDS) + len(GALLERY_IDS), q.shape[1])
    assert np.allclose(emb[: len(QUERY_IDS)], q)
    assert np.allclose(emb[len(QUERY_IDS):], g)

    # candidates: header + only accepted (threshold 0.0 accepts all)
    with open(out / "candidates.csv", encoding="utf-8", newline="") as f:
        cand = list(csv.reader(f))
    assert cand[0] == ["query_id", "gallery_id", "confidence"]
    assert len(cand) - 1 == len(QUERY_IDS)
    assert info["accepted"] == len(QUERY_IDS)


def test_refusal_omits_row(tmp_path):
    q, g = _embeddings()
    # a threshold above every score refuses everyone -> zero data rows
    runner.rank_and_write(str(tmp_path / "out"), q, g, QUERY_IDS, GALLERY_IDS,
                          variant="dino", threshold=2.0)
    with open(tmp_path / "out" / "candidates.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["query_id", "gallery_id", "confidence"]
    assert rows[1:] == []


def test_validate_format_passes(tmp_path):
    q, g = _embeddings()
    out = tmp_path / "out"
    runner.rank_and_write(str(out), q, g, QUERY_IDS, GALLERY_IDS,
                          variant="dino", threshold=0.0)
    qcsv = tmp_path / "q.csv"
    gcsv = tmp_path / "g.csv"
    _write_split(qcsv, QUERY_IDS)
    _write_split(gcsv, GALLERY_IDS)
    rep = validate(str(out), str(qcsv), str(gcsv))
    assert rep["ok"], rep["errors"]


def test_byte_determinism(tmp_path):
    q, g = _embeddings()
    a, b = tmp_path / "a", tmp_path / "b"
    runner.rank_and_write(str(a), q, g, QUERY_IDS, GALLERY_IDS,
                          variant="dino", threshold=0.3)
    runner.rank_and_write(str(b), q, g, QUERY_IDS, GALLERY_IDS,
                          variant="dino", threshold=0.3)
    for name in ("submission.csv", "embeddings.npy", "candidates.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
