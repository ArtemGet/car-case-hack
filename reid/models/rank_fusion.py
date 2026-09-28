"""Rank / score fusion of the two champion backbones (CPU-only postproc).

Champion (exp-0057) fuses SigLIP2 NaFlex 512d and DINOv2-B 512d by *vector
concatenation* (``L2(concat[L2(sig), 0.8*L2(dino)])``) and then runs one
k-reciprocal pass on the concatenated space.  Concatenation of L2-normalised
blocks is mathematically a weighted cosine sum, so the cosine *score* side is
already covered.  The genuinely different angle is to fuse at the **rank /
score level after re-ranking each backbone separately**, because the two
retrieval manifolds have different neighbourhood graphs.

Methods compared (all: query + gallery only, per-query; streaming invariant
holds -- RRF/score fusion is computed per query over the gallery):

    concat_cos          cosine of the concat fusion (reference base)
    concat_rr           champion: k-reciprocal on concat fusion
    sig_rr / dino_rr    k-reciprocal in each single-backbone space
    rrf_cos             reciprocal-rank fusion of the two cosine rankings
    rrf_rr              reciprocal-rank fusion of the two re-ranked lists
    rrf_all             RRF(concat_rr, sig_rr, dino_rr)
    wscore_rr           per-query min-max score fusion of sig_rr / dino_rr
    concat_rr_plus_rrf  concat_rr order re-scored by a small RRF(rr_sig,rr_dino)

Metrics: OFFICIAL ``evaluate.py`` only (via reid.eval.harness).

    python -m reid.models.rank_fusion --out runs/exp-0059-rankfusion \
        --json reports/exp-0059-rankfusion.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models.fusion_postproc import build_post, flat

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def write_sub(path, q_ids, g_ids, orders, scores, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])
    with open(os.path.join(os.path.dirname(path), "candidates.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def rr_pass(q, g, rr_cfg):
    """k-reciprocal pass -> (order (nq,ng), score (nq,ng))."""
    preps = [rerank.prepare_query(q[i], g, pool_size=rr_cfg["pool_size"])
             for i in range(q.shape[0])]
    o = np.empty((q.shape[0], g.shape[0]), np.int64)
    sc = np.empty((q.shape[0], g.shape[0]), np.float32)
    for i in range(q.shape[0]):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=rr_cfg["k1"],
                                           k2=rr_cfg["k2"], lam=rr_cfg["lam"])
    return o, sc


def ranks_from_order(order):
    """(nq,ng) order -> rank matrix (0 = best)."""
    nq, ng = order.shape
    r = np.empty((nq, ng), np.int64)
    rows = np.arange(nq)[:, None]
    r[rows, order] = np.arange(ng)[None, :]
    return r


def rrf(rank_list, k=60.0):
    """Reciprocal-rank fusion; rank_list = list of (nq,ng) rank matrices."""
    acc = np.zeros_like(rank_list[0], dtype=np.float64)
    for r in rank_list:
        acc += 1.0 / (k + r.astype(np.float64))
    return acc


def minmax_rows(x):
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.clip(hi - lo, 1e-12, None)


def order_from_score(score):
    return np.argsort(-score, axis=1, kind="stable")


def eval_order(orders, scores, ids, evaldir, emb, tag):
    os.makedirs(evaldir, exist_ok=True)
    q_ids, g_ids, gt, qcsv, gcsv = ids
    sub = os.path.join(evaldir, "submission.csv")
    np.save(os.path.join(evaldir, "embeddings.npy"), emb)
    write_sub(sub, q_ids, g_ids, orders, scores)
    rep = flat(run_official(gt_csv=gt, submission=sub,
                            candidates=os.path.join(evaldir, "candidates.csv"),
                            embeddings=os.path.join(evaldir, "embeddings.npy"),
                            query=qcsv, gallery=gcsv,
                            json_out=os.path.join(evaldir, "official.json")))
    print(f"{tag:26s} mAP@10={rep['mAP@10']:.5f} R1={rep['Rank-1']:.5f} "
          f"R5={rep['Rank-5']:.5f} mAPfull={rep['mAP_full']:.5f} "
          f"mINP={rep['mINP']:.5f}", flush=True)
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
    ap.add_argument("--rrf-k", type=float, default=60.0)
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

    rr_cfg = {"k1": args.k1, "k2": args.k2, "lam": args.lam,
              "pool_size": args.pool}

    # component feature spaces (L2)
    s = l2(sig); d = l2(dino)
    sq, sg = s[:nq], s[nq:]
    dq, dg = d[:nq], d[nq:]

    # concat fusion
    cq, cg = build_post(sig, dino, nq, {"w": args.w})
    emb_concat = np.vstack([cq, cg]).astype(np.float32)

    print("[rr] sig ...", flush=True)
    sig_o, sig_s = rr_pass(sq, sg, rr_cfg)
    print("[rr] dino ...", flush=True)
    dino_o, dino_s = rr_pass(dq, dg, rr_cfg)
    print("[rr] concat ...", flush=True)
    cat_o, cat_s = rr_pass(cq, cg, rr_cfg)

    results = []

    # reference: concat cosine + champion concat rr
    cos_cat = cq @ cg.T
    results.append(eval_order(order_from_score(cos_cat), cos_cat, ids,
                              os.path.join(out, "concat_cos"), emb_concat,
                              "concat_cos"))
    results.append(eval_order(cat_o, cat_s, ids,
                              os.path.join(out, "concat_rr"), emb_concat,
                              "concat_rr"))
    results.append(eval_order(sig_o, sig_s, ids,
                              os.path.join(out, "sig_rr"), emb_concat, "sig_rr"))
    results.append(eval_order(dino_o, dino_s, ids,
                              os.path.join(out, "dino_rr"), emb_concat,
                              "dino_rr"))

    r_cos_sig = ranks_from_order(order_from_score(sq @ sg.T))
    r_cos_dino = ranks_from_order(order_from_score(dq @ dg.T))
    r_sig = ranks_from_order(sig_o)
    r_dino = ranks_from_order(dino_o)
    r_cat = ranks_from_order(cat_o)

    k = args.rrf_k
    combos = {
        "rrf_cos": [r_cos_sig, r_cos_dino],
        "rrf_rr": [r_sig, r_dino],
        "rrf_all": [r_cat, r_sig, r_dino],
        "rrf_cat_sig": [r_cat, r_sig],
        "rrf_cat_dino": [r_cat, r_dino],
    }
    for tag, rl in combos.items():
        sc = rrf(rl, k=k)
        results.append(eval_order(order_from_score(sc), sc.astype(np.float32),
                                  ids, os.path.join(out, tag), emb_concat, tag))

    # per-query min-max score fusion of the two re-ranked lists
    ws = minmax_rows(sig_s.astype(np.float64)) + minmax_rows(dino_s.astype(np.float64))
    results.append(eval_order(order_from_score(ws), ws.astype(np.float32), ids,
                              os.path.join(out, "wscore_rr"), emb_concat,
                              "wscore_rr"))
    # weighted 0.5/0.5 vs 0.6/0.4 already symmetric; try ratio 0.6 sig /0.4 dino
    ws2 = 0.6 * minmax_rows(sig_s.astype(np.float64)) + 0.4 * minmax_rows(dino_s.astype(np.float64))
    results.append(eval_order(order_from_score(ws2), ws2.astype(np.float32), ids,
                              os.path.join(out, "wscore_rr_64"), emb_concat,
                              "wscore_rr_64"))

    best = max(results, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "rank/score fusion",
              "w": args.w, "rr": rr_cfg, "rrf_k": k, "results": results,
              "best": best}
    with open(os.path.join(out, "rank_fusion_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
