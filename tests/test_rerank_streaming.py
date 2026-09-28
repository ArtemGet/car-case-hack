"""Streaming invariant tests for ``reid.rerank`` (W2-1).

Hard rule (INTERFACES.md §5): for any query ``q``, ``rank(q)`` must not change
when OTHER queries are permuted, removed, or replaced. The gallery and the
current query are the only permitted inputs; hflip is forbidden.

These tests use tiny synthetic arrays -- no dataset, no model, no GPU.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from reid import rerank

MODULE_PATH = os.path.abspath(rerank.__file__)


def _embeddings(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return rerank.l2norm(x)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    d = 32
    n_g = 60
    n_q = 10
    gallery = _embeddings(n_g, d, seed=1)
    # queries = a shifted version of some gallery rows + noise (realistic)
    base = gallery[rng.integers(0, n_g, size=n_q)]
    queries = rerank.l2norm(base + 0.05 * rng.standard_normal((n_q, d)).astype(np.float32))
    return queries, gallery


def _order_for(queries, target, gallery, **kw):
    return rerank.k_reciprocal_order(queries[target], gallery, **kw)[0]


# ---------------------------------------------------------------------------
# Streaming: no dependence on other queries
# ---------------------------------------------------------------------------
def test_batch_row_equals_single(data):
    queries, gallery = data
    t = 3
    single = _order_for(queries, t, gallery, k1=10, k2=2, lam=0.3, pool_size=30)
    orders, _ = rerank.rerank_batch(queries, gallery, k1=10, k2=2, lam=0.3,
                                    pool_size=30)
    assert np.array_equal(orders[t], single)


def test_permuting_other_queries_is_invariant(data):
    queries, gallery = data
    t = 3
    single = _order_for(queries, t, gallery, k1=10, k2=2, lam=0.3, pool_size=30)

    perm = np.random.default_rng(7).permutation(len(queries))
    new_t = int(np.where(perm == t)[0][0])
    shuffled = queries[perm]
    assert np.array_equal(_order_for(shuffled, new_t, gallery,
                                     k1=10, k2=2, lam=0.3, pool_size=30), single)


def test_removing_other_queries_is_invariant(data):
    queries, gallery = data
    t = 3
    single = _order_for(queries, t, gallery, k1=10, k2=2, lam=0.3, pool_size=30)
    # keep only the target query; drop every other query entirely
    only = queries[[t]]
    assert np.array_equal(_order_for(only, 0, gallery,
                                     k1=10, k2=2, lam=0.3, pool_size=30), single)


def test_replacing_other_queries_is_invariant(data):
    queries, gallery = data
    t = 3
    single = _order_for(queries, t, gallery, k1=10, k2=2, lam=0.3, pool_size=30)

    mutated = queries.copy()
    mutated[[i for i in range(len(queries)) if i != t]] = _embeddings(
        len(queries) - 1, queries.shape[1], seed=99)
    assert np.array_equal(_order_for(mutated, t, gallery,
                                     k1=10, k2=2, lam=0.3, pool_size=30), single)


def test_prepare_then_rank_matches_oneshot(data):
    queries, gallery = data
    q = queries[3]
    prep = rerank.prepare_query(q, gallery, pool_size=30)
    order_prep, scores_prep = rerank.rank_prepared(prep, k1=10, k2=2, lam=0.3)
    order_one, scores_one = rerank.k_reciprocal_order(
        q, gallery, k1=10, k2=2, lam=0.3, pool_size=30)
    assert np.array_equal(order_prep, order_one)
    assert np.allclose(scores_prep, scores_one)


def test_no_other_query_argument_exists():
    """The public entry points must take exactly ONE query, not a query set."""
    import inspect

    sig = inspect.signature(rerank.k_reciprocal_order)
    assert "query" in sig.parameters
    assert "queries" not in sig.parameters
    sig2 = inspect.signature(rerank.prepare_query)
    assert "query" in sig2.parameters
    assert "queries" not in sig2.parameters


# ---------------------------------------------------------------------------
# Ordering well-formedness
# ---------------------------------------------------------------------------
def test_order_is_full_permutation(data):
    queries, gallery = data
    order, scores = rerank.k_reciprocal_order(queries[0], gallery, k1=10,
                                              k2=2, lam=0.3, pool_size=30)
    assert order.shape == (len(gallery),)
    assert sorted(order.tolist()) == list(range(len(gallery)))
    assert np.all(np.diff(scores) <= 1e-6)          # non-increasing


def test_determinism(data):
    queries, gallery = data
    a = rerank.k_reciprocal_order(queries[2], gallery, k1=10, k2=2, lam=0.3,
                                  pool_size=30)
    b = rerank.k_reciprocal_order(queries[2], gallery, k1=10, k2=2, lam=0.3,
                                  pool_size=30)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_small_gallery_edge_cases():
    g = _embeddings(3, 8, seed=5)
    q = _embeddings(1, 8, seed=6)[0]
    order, _ = rerank.k_reciprocal_order(q, g, k1=10, k2=5, lam=0.3,
                                         pool_size=100)
    assert sorted(order.tolist()) == [0, 1, 2]


def test_pool_size_capped():
    g = _embeddings(5, 8, seed=5)
    q = _embeddings(1, 8, seed=6)[0]
    prep = rerank.prepare_query(q, g, pool_size=999)
    assert len(prep.pool) == 5


# ---------------------------------------------------------------------------
# Query-side TTA (no hflip)
# ---------------------------------------------------------------------------
def test_fuse_is_unit_norm_and_order_invariant():
    rng = np.random.default_rng(3)
    views = [rerank.l2norm(rng.standard_normal((16,)).astype(np.float32))
             for _ in range(3)]
    a = rerank.fuse_embeddings(views)
    b = rerank.fuse_embeddings(views[::-1])
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert np.allclose(a, b, atol=1e-6)


def test_fuse_single_view_is_identity():
    v = rerank.l2norm(np.arange(8, dtype=np.float32))
    assert np.allclose(rerank.fuse_embeddings([v]), v, atol=1e-6)


def test_fuse_weighted_selects_view():
    rng = np.random.default_rng(4)
    v1 = rerank.l2norm(rng.standard_normal((16,)).astype(np.float32))
    v2 = rerank.l2norm(rng.standard_normal((16,)).astype(np.float32))
    fused = rerank.fuse_embeddings([v1, v2], weights=[1.0, 0.0])
    assert np.allclose(fused, v1, atol=1e-6)


def test_no_hflip_in_module():
    """No horizontal-flip operation may appear in the re-ranking code.

    Parses the AST (so the word is still allowed in prose/docstrings, where the
    rule is stated) and rejects any identifier or attribute containing 'flip'.
    """
    import ast

    with open(MODULE_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert "flip" not in node.id.lower(), node.id
        elif isinstance(node, ast.Attribute):
            assert "flip" not in node.attr.lower(), node.attr
