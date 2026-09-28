"""Per-query k-reciprocal re-ranking and query-side TTA (W2-1 / W2-2).

Streaming contract (INTERFACES.md §5, 00_BRIEF.md §5)
----------------------------------------------------
The ranking produced for a query ``q`` depends ONLY on ``q``'s own embedding
and the whole gallery. Nothing in this module may look at any other query.
k-reciprocal is computed inside the TOP-K candidate pool of a SINGLE query
relative to the gallery -- explicitly allowed by the organisers.
Query-side TTA averages several scales/crops of ONE image; hflip is forbidden
(vehicles are not left/right symmetric, see reid/data/aug.py).

Re-rank only changes the *order* of gallery items (submission / candidates);
it never rewrites ``embeddings.npy``.

Public API
----------
    l2norm(x)
    fuse_embeddings(embs, weights=None)          # query-side TTA (no hflip)
    k_reciprocal_order(query, gallery, k1, k2, lam, pool_size=...)
        -> (order, scores)  # highest score first; order indexes gallery rows
    rerank_batch(queries, gallery, ...)          # convenience loop (row == single)

Algorithm (Zhong et al., CVPR 2017, restricted to one query + gallery):
    1. pool = top-``pool_size`` gallery neighbours of ``q`` by cosine.
    2. local graph = {q} ∪ pool; distances = (1 - cosine) / 2 in [0, 1].
    3. k-reciprocal neighbourhood R(x, k1) and its expansion R*(x, k1).
    4. query-expansion V_qe over the k2 nearest local neighbours.
    5. Jaccard distance of ``q`` against every local node.
    6. final distance = (1 - lam) * jaccard + lam * original cosine distance.
Non-pool gallery items keep their original distance, so the full gallery
ordering is a strict re-ranking of the local pool (top-K) only.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "l2norm",
    "fuse_embeddings",
    "prepare_query",
    "rank_prepared",
    "k_reciprocal_order",
    "rerank_batch",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (safe against zero rows)."""
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def fuse_embeddings(embs, weights=None) -> np.ndarray:
    """Query-side TTA: L2-normalised weighted mean of several views of ONE image.

    Parameters
    ----------
    embs : sequence of (D,) or (N, D) float arrays
        Embeddings of the SAME query image at different scales/crops.
        Must NOT contain hflip views.
    weights : sequence of float, optional
        Per-view weights (default: uniform).

    Returns
    -------
    (D,) float32 L2-normalised fused embedding.
    """
    arrs = [l2norm(np.asarray(e, dtype=np.float32).reshape(-1)) for e in embs]
    if not arrs:
        raise ValueError("fuse_embeddings: empty input")
    stack = np.stack(arrs, axis=0)
    if weights is None:
        w = np.ones((stack.shape[0],), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != stack.shape[0]:
            raise ValueError("fuse_embeddings: weights length mismatch")
    w = w / np.clip(w.sum(), 1e-12, None)
    fused = (stack * w[:, None]).sum(axis=0)
    return l2norm(fused)


# ---------------------------------------------------------------------------
# k-reciprocal internals
# ---------------------------------------------------------------------------
def _initial_rank(dist: np.ndarray) -> np.ndarray:
    """Rank every node's nearest neighbours (ascending; node itself is rank 0)."""
    return np.argsort(dist, axis=1, kind="stable")


def _reciprocal_sets(initial_rank: np.ndarray, k1: int):
    """R(i, k1) = mutual top-k1 neighbours. Returns a list of index arrays."""
    n = initial_rank.shape[0]
    k1 = max(1, min(int(k1), n - 1))
    out = []
    for i in range(n):
        fwd = initial_rank[i, : k1 + 1]
        bwd = initial_rank[fwd, : k1 + 1]
        rec = fwd[(bwd == i).any(axis=1)]
        out.append(np.asarray(rec, dtype=np.int64))
    return out


def _expand_reciprocal(R, k1: int):
    """R*(i) = R(i) ∪ { R(j) : |R(i) ∩ R(j)| >= 2/3 |R(i)| }  (paper).

    The expansion candidates are the members of R(i) itself, which is exactly
    the standard k-reciprocal expansion. Each R(i) is small (<= k1), so this
    loop is cheap.
    """
    out = []
    for i, base in enumerate(R):
        acc = set(base.tolist())
        base_list = base.tolist()
        need = (2.0 / 3.0) * len(base_list)
        for cand in base_list:
            rc = R[int(cand)].tolist()
            inter = len(acc.intersection(rc))
            if inter >= need and inter > 0:
                acc.update(rc)
        out.append(np.fromiter(sorted(acc), dtype=np.int64))
    return out


def _build_V(initial_rank: np.ndarray, dist: np.ndarray, k1: int) -> np.ndarray:
    """k-reciprocal feature matrix V (rows are local nodes)."""
    n = dist.shape[0]
    R = _reciprocal_sets(initial_rank, k1)
    Rstar = _expand_reciprocal(R, k1)
    V = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = Rstar[i]
        w = np.exp(-dist[i, idx].astype(np.float64)).astype(np.float32)
        s = float(w.sum())
        V[i, idx] = w / s if s > 0 else 0.0
    return V


def _query_expand(V: np.ndarray, initial_rank: np.ndarray, k2: int) -> np.ndarray:
    """V_qe[i] = mean of V over i's k2 nearest neighbours (self included)."""
    n = V.shape[0]
    k2 = max(1, min(int(k2), n))
    out = np.zeros_like(V)
    for i in range(n):
        idx = initial_rank[i, :k2]
        out[i] = V[idx].mean(axis=0)
    return out


def _jaccard_row(V: np.ndarray, i: int = 0) -> np.ndarray:
    """Jaccard distance from local node ``i`` to every local node."""
    vi = V[i]
    mins = np.minimum(vi[None, :], V).sum(axis=1)
    return 1.0 - mins / np.clip(2.0 - mins, 1e-12, None)


# ---------------------------------------------------------------------------
# Prepared per-query state (query + gallery only)
# ---------------------------------------------------------------------------
class _Prepared:
    """Everything needed to score one query, independent of any other query."""

    __slots__ = ("pool", "dist_local", "initial_rank", "orig_dist", "n_gallery")

    def __init__(self, pool, dist_local, initial_rank, orig_dist, n_gallery):
        self.pool = pool                  # (K,) gallery indices
        self.dist_local = dist_local      # (K+1, K+1) local cosine distance
        self.initial_rank = initial_rank  # (K+1, K+1) local ranks
        self.orig_dist = orig_dist        # (Ng,) query->gallery cosine distance
        self.n_gallery = n_gallery


def prepare_query(query: np.ndarray, gallery: np.ndarray,
                  pool_size: int = 100) -> _Prepared:
    """Build the query-local candidate pool and distances.

    Uses ONLY ``query`` and ``gallery`` -- no other query, ever.
    """
    q = l2norm(np.asarray(query, dtype=np.float32).reshape(-1))
    g = l2norm(np.asarray(gallery, dtype=np.float32))
    n_g = g.shape[0]
    if n_g == 0:
        raise ValueError("prepare_query: empty gallery")

    sim = g @ q                                    # (Ng,) cosine
    k = int(min(max(int(pool_size), 1), n_g))
    pool = np.argsort(-sim, kind="stable")[:k]

    # original cosine distance in [0, 1] (cosine in [-1, 1])
    orig_dist = ((1.0 - sim) / 2.0).astype(np.float32)

    # local node 0 = query, nodes 1..k = pool
    local = np.vstack([q[None, :], g[pool]])
    S = np.clip(local @ local.T, -1.0, 1.0)
    dist_local = ((1.0 - S) / 2.0).astype(np.float32)
    dist_local = np.clip(dist_local, 0.0, 1.0)
    np.fill_diagonal(dist_local, 0.0)
    return _Prepared(pool, dist_local, _initial_rank(dist_local), orig_dist, n_g)


def rank_prepared(prep: _Prepared, k1: int = 20, k2: int = 2,
                  lam: float = 0.3):
    """Re-rank the gallery for a prepared query. Returns ``(order, scores)``.

    The top-``pool_size`` candidates are re-ordered by the k-reciprocal
    distance (fused with the original distance by ``lam``); every gallery item
    outside the pool keeps its original relative order and is placed AFTER the
    pool. Restricting the re-ordering to the query's own top-K pool is exactly
    what the protocol allows, and it avoids a distance-scale discontinuity at
    the pool boundary.

    ``order`` is a full permutation of gallery rows, best first; ``scores`` is
    a monotone confidence (higher = closer) aligned with ``order``.
    """
    n_g = prep.n_gallery
    dist_local = prep.dist_local
    V = _build_V(prep.initial_rank, dist_local, k1)
    Vqe = _query_expand(V, prep.initial_rank, k2)

    jac = _jaccard_row(Vqe, i=0)[1:].astype(np.float64)      # (k,) pool nodes
    orig = dist_local[0, 1:].astype(np.float64)              # (k,) pool nodes

    # Put both signals on a common [0, 1] scale inside the pool so that
    # ``lam`` is a true interpolation: lam=0 -> pure k-reciprocal, lam=1 -> the
    # original cosine order (the baseline).
    jac_n = _minmax(jac)
    orig_n = _minmax(orig)
    final_local = (1.0 - float(lam)) * jac_n + float(lam) * orig_n

    pool_order = prep.pool[np.argsort(final_local, kind="stable")]
    rest_mask = np.ones(n_g, dtype=bool)
    rest_mask[prep.pool] = False
    rest = np.nonzero(rest_mask)[0]
    rest_order = rest[np.argsort(prep.orig_dist[rest], kind="stable")]
    order = np.concatenate([pool_order, rest_order])

    # Monotone distance aligned with ``order``: pool first, rest strictly after.
    full = np.empty(n_g, dtype=np.float64)
    full[order[: len(pool_order)]] = np.sort(final_local)
    offset = (float(final_local.max()) + 1.0) if len(final_local) else 1.0
    full[rest_order] = offset + prep.orig_dist[rest_order].astype(np.float64)
    scores = (1.0 - full[order]).astype(np.float32)
    return order, scores


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def k_reciprocal_order(query: np.ndarray, gallery: np.ndarray,
                       k1: int = 20, k2: int = 2, lam: float = 0.3,
                       pool_size: int = 100):
    """One-shot per-query k-reciprocal re-ranking (convenience wrapper)."""
    prep = prepare_query(query, gallery, pool_size=pool_size)
    return rank_prepared(prep, k1=k1, k2=k2, lam=lam)


def rerank_batch(queries: np.ndarray, gallery: np.ndarray,
                 k1: int = 20, k2: int = 2, lam: float = 0.3,
                 pool_size: int = 100, progress: bool = False):
    """Apply :func:`k_reciprocal_order` to each query independently.

    This is a thin loop: row ``i`` of the result is byte-for-byte identical to
    calling :func:`k_reciprocal_order` on ``queries[i]`` alone. It exists only
    for convenience in submission generation; it holds no cross-query state.
    """
    queries = np.asarray(queries, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    n_q = queries.shape[0]
    n_g = gallery.shape[0]
    orders = np.empty((n_q, n_g), dtype=np.int64)
    scores = np.empty((n_q, n_g), dtype=np.float32)
    for i in range(n_q):
        orders[i], scores[i] = k_reciprocal_order(
            queries[i], gallery, k1=k1, k2=k2, lam=lam, pool_size=pool_size)
        if progress and (i + 1) % 50 == 0:
            print(f"    rerank {i + 1}/{n_q}", flush=True)
    return orders, scores
