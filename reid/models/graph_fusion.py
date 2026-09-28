"""Heterogeneous k-reciprocal graph fusion (CPU-only).

The champion builds the k-reciprocal neighbourhood graph and the final blend in
the SAME concatenated space.  Here we decouple them: the jaccard term can come
from one backbone's local graph while the base distance comes from another
(concat / sig / dino).  Everything is computed inside ONE query's top-K pool
relative to the gallery (streaming invariant).  Metrics: OFFICIAL evaluator.

    python -m reid.models.graph_fusion --out runs/exp-0062-graphfusion \
        --json reports/exp-0062-graphfusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models.dba_postproc import write_sub
from reid.models.fusion_postproc import build_post, flat

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def local_prep(qrow, gallery, pool):
    """Return (pool_idx, S_g, S_q) local cosine blocks.

    S_g : (K,K) gallery-pool self cosine; S_q : (K,) query->pool cosine.
    """
    sim = gallery @ qrow
    pool = np.argsort(-sim, kind="stable")[:pool]
    gp = gallery[pool]                       # (K, D)
    S_g = np.clip(gp @ gp.T, -1.0, 1.0)
    S_q = np.clip((gp @ qrow), -1.0, 1.0)
    return pool, S_g, S_q


def jaccard_q(S_g, S_q, k1, k2):
    """k-reciprocal jaccard distance from query (node 0) into the pool.

    Builds the (K+1) local distance graph, exactly as reid.rerank does, and
    returns the jaccard vector over the K pool nodes.
    """
    K = S_g.shape[0]
    S = np.empty((K + 1, K + 1), np.float32)
    S[0, 0] = 1.0
    S[0, 1:] = S_q
    S[1:, 0] = S_q
    S[1:, 1:] = S_g
    dist = np.clip((1.0 - S) / 2.0, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    ir = rerank._initial_rank(dist)
    V = rerank._build_V(ir, dist, k1)
    Vqe = rerank._query_expand(V, ir, k2)
    return rerank._jaccard_row(Vqe, i=0)[1:].astype(np.float64)


def eval_order(orders, scores, ids, evaldir, emb, tag):
    os.makedirs(evaldir, exist_ok=True)
    q_ids, g_ids, gt, qcsv, gcsv = ids
    sub = os.path.join(evaldir, "submission.csv")
    np.save(os.path.join(evaldir, "embeddings.npy"), emb)
    write_sub(sub, q_ids, g_ids, orders, scores.astype(np.float32))
    rep = flat(run_official(gt_csv=gt, submission=sub,
                            candidates=os.path.join(evaldir, "candidates.csv"),
                            embeddings=os.path.join(evaldir, "embeddings.npy"),
                            query=qcsv, gallery=gcsv,
                            json_out=os.path.join(evaldir, "official.json")))
    print(f"{tag:30s} mAP@10={rep['mAP@10']:.5f} R1={rep['Rank-1']:.5f} "
          f"R5={rep['Rank-5']:.5f}", flush=True)
    return {"tag": tag, **rep}


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dino = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == dino.shape[0] == nq + len(vg)
    ng = len(vg)

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    cq, cg = build_post(sig, dino, nq, {"w": args.w})   # concat L2
    sq, sg = l2(sig[:nq]), l2(sig[nq:])
    dq, dg = l2(dino[:nq]), l2(dino[nq:])
    emb = np.vstack([cq, cg]).astype(np.float32)

    evaldir = os.path.join(out, "eval")
    results = []

    # precompute per-query jaccard + orig for each space
    spaces = {"concat": (cq, cg), "sig": (sq, sg), "dino": (dq, dg)}
    # full-gallery base cosine per space (for ordering the out-of-pool rest)
    OFULL = {k: (qq @ gg.T).astype(np.float64)
             for k, (qq, gg) in spaces.items()}
    J = {k: np.empty((nq, args.pool)) for k in spaces}
    O = {k: np.empty((nq, args.pool)) for k in spaces}
    POOL = []
    for i in range(nq):
        # pool always from concat (the deployed candidate set)
        pool, S_g_c, S_q_c = local_prep(cq[i], cg, args.pool)
        POOL.append(pool)
        for k, (qq, gg) in spaces.items():
            gp = gg[pool]
            S_g = np.clip(gp @ gp.T, -1.0, 1.0)
            S_q = np.clip(gp @ qq[i], -1.0, 1.0)
            J[k][i] = jaccard_q(S_g, S_q, args.k1, args.k2)
            O[k][i] = (1.0 - S_q) / 2.0

    def build(jac_key, base_key):
        orders = np.empty((nq, ng), np.int64)
        scores = np.empty((nq, ng), np.float32)
        for i in range(nq):
            jac_n = rerank._minmax(J[jac_key][i])
            orig_n = rerank._minmax(O[base_key][i])
            fl = (1.0 - args.lam) * jac_n + args.lam * orig_n
            pool = POOL[i]
            po = pool[np.argsort(fl, kind="stable")]
            mask = np.ones(ng, bool); mask[pool] = False
            rest = np.nonzero(mask)[0]
            ro = rest[np.argsort(OFULL[base_key][i][rest], kind="stable")]
            orders[i] = np.concatenate([po, ro])
            full = np.empty(ng, np.float64)
            full[po] = np.sort(fl)
            off = float(fl.max()) + 1.0
            full[ro] = off + OFULL[base_key][i][ro]
            scores[i] = (1.0 - full[orders[i]]).astype(np.float32)
        return orders, scores

    combos = [
        ("ref_jcat_bcat", "concat", "concat"),
        ("jcat_bsig", "concat", "sig"),
        ("jcat_bdino", "concat", "dino"),
        ("jsig_bcat", "sig", "concat"),
        ("jdino_bcat", "dino", "concat"),
    ]
    for tag, jk, bk in combos:
        o, sc = build(jk, bk)
        results.append(eval_order(o, sc, ids, os.path.join(evaldir, tag), emb, tag))

    # averaged jaccard (concat+sig+dino)/3 with concat base
    o = np.empty((nq, ng), np.int64); sc = np.empty((nq, ng), np.float32)
    for i in range(nq):
        jac = (J["concat"][i] + J["sig"][i] + J["dino"][i]) / 3.0
        jac_n = rerank._minmax(jac); orig_n = rerank._minmax(O["concat"][i])
        fl = (1.0 - args.lam) * jac_n + args.lam * orig_n
        pool = POOL[i]; po = pool[np.argsort(fl, kind="stable")]
        mask = np.ones(ng, bool); mask[pool] = False
        rest = np.nonzero(mask)[0]
        ro = rest[np.argsort(OFULL["concat"][i][rest], kind="stable")]
        o[i] = np.concatenate([po, ro])
        full = np.empty(ng, np.float64); full[po] = np.sort(fl)
        full[ro] = float(fl.max()) + 1.0 + OFULL["concat"][i][ro]
        sc[i] = (1.0 - full[o[i]]).astype(np.float32)
    results.append(eval_order(o, sc, ids, os.path.join(evaldir, "javg_bcat"),
                              emb, "javg_bcat"))

    best = max(results, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "heterogeneous graph fusion",
              "w": args.w, "pool": args.pool, "k1": args.k1, "k2": args.k2,
              "lam": args.lam, "results": results, "best": best}
    with open(os.path.join(out, "graph_fusion_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
