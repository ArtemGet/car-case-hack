"""Thin per-query re-ranker on the champion concat fusion (CPU-only).

The champion ranks by a k-reciprocal pass on the fused feature.  Here we probe a
cheap, label-free "thin" re-ranker that only ever touches ONE query + the whole
gallery (streaming invariant):

  1. gallery-side hubness correction (CSLS-style, gallery-only neighbour stats):
        r_q = mean cosine of q to its k nearest gallery items
        r_g = mean cosine of g to its k nearest GALLERY items   (gallery-only!)
        s'  = cos(q,g) - beta * 0.5*(r_q + r_g)
  2. gallery-side score smoothing: a gallery item's score is smoothed with its
     own top-k gallery neighbours (a score-domain analogue of DBA):
        s'(q,g) = (1-a) cos(q,g) + a * mean_{g' in topk_gallery(g)} cos(q,g')

Both are then optionally followed by the champion k-reciprocal pass.  Nothing
looks at any other query.  Metrics: OFFICIAL evaluator.

    python -m reid.models.hubness_rerank --out runs/exp-0061-hubness \
        --json reports/exp-0061-hubness.json
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


def eval_scores(score, ids, evaldir, emb, tag):
    os.makedirs(evaldir, exist_ok=True)
    q_ids, g_ids, gt, qcsv, gcsv = ids
    sub = os.path.join(evaldir, "submission.csv")
    np.save(os.path.join(evaldir, "embeddings.npy"), emb)
    orders = np.argsort(-score, axis=1, kind="stable")
    write_sub(sub, q_ids, g_ids, orders, score.astype(np.float32))
    rep = flat(run_official(gt_csv=gt, submission=sub,
                            candidates=os.path.join(evaldir, "candidates.csv"),
                            embeddings=os.path.join(evaldir, "embeddings.npy"),
                            query=qcsv, gallery=gcsv,
                            json_out=os.path.join(evaldir, "official.json")))
    print(f"{tag:34s} mAP@10={rep['mAP@10']:.5f} R1={rep['Rank-1']:.5f} "
          f"R5={rep['Rank-5']:.5f} mINP={rep['mINP']:.5f}", flush=True)
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

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    q, g = build_post(sig, dino, nq, {"w": args.w})
    emb = np.vstack([q, g]).astype(np.float32)
    ng = g.shape[0]
    cos = q @ g.T                                  # (nq, ng)
    evaldir = os.path.join(out, "eval")
    results = []

    # champion reference: k-reciprocal on concat
    preps = [rerank.prepare_query(q[i], g, pool_size=args.pool) for i in range(nq)]
    o = np.empty((nq, ng), np.int64); sc = np.empty((nq, ng), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=8, k2=3, lam=0.5)
    # rr score aligned to the gallery index (rank_prepared returns it in `order`)
    rr_g = np.empty((nq, ng), np.float32)
    rr_g[np.arange(nq)[:, None], o] = sc
    results.append(eval_scores(rr_g, ids, os.path.join(evaldir, "ref_rr"), emb,
                               "ref_concat_rr"))

    # preprocessing: gallery-only self-similarity (ng, ng)
    gn = l2(g)
    Sg = gn @ gn.T

    def rerank_from_score(score, rr_cfg):
        # re-order via k-reciprocal on the FUSED feat but start from a score is
        # not possible; k-reciprocal stays feature-based. So we approximate the
        # "then rr" variants by blending the base score with the rr score.
        pass

    # --- CSLS-style gallery-only hubness correction --------------------------
    for kg in (5, 10, 20):
        # r_g: mean cosine of g to its k nearest gallery neighbours (excl self)
        idx = np.argsort(-Sg, axis=1, kind="stable")[:, 1:kg + 1]
        r_g = np.take_along_axis(Sg, idx, axis=1).mean(axis=1)          # (ng,)
        # r_q: mean cosine of q to its k nearest gallery
        idxq = np.argsort(-cos, axis=1, kind="stable")[:, :kg]
        r_q = np.take_along_axis(cos, idxq, axis=1).mean(axis=1)        # (nq,)
        for beta in (0.5, 1.0):
            s2 = cos - beta * 0.5 * (r_q[:, None] + r_g[None, :])
            results.append(eval_scores(s2, ids, os.path.join(
                evaldir, f"csls_k{kg}_b{beta}"), emb, f"csls_k{kg}_b{beta}"))
            # blend with rr scores (rr is a monotone confidence)
            combined = 0.5 * _row_z(s2) + 0.5 * _row_z(rr_g)
            results.append(eval_scores(combined, ids, os.path.join(
                evaldir, f"csls_k{kg}_b{beta}_rr"), emb,
                f"csls_k{kg}_b{beta}+rr"))

    # --- gallery-side score smoothing (score-domain DBA) ---------------------
    for ks in (3, 5, 10):
        idx = np.argsort(-Sg, axis=1, kind="stable")[:, :ks]
        nb_cos = cos[:, idx]                      # (nq, ng, ks)
        sm = nb_cos.mean(axis=2)                  # (nq, ng)
        for al in (0.2, 0.35, 0.5):
            s2 = (1.0 - al) * cos + al * sm
            results.append(eval_scores(s2, ids, os.path.join(
                evaldir, f"smooth_k{ks}_a{al}"), emb, f"smooth_k{ks}_a{al}"))
            combined = 0.5 * _row_z(s2) + 0.5 * _row_z(rr_g)
            results.append(eval_scores(combined, ids, os.path.join(
                evaldir, f"smooth_k{ks}_a{al}_rr"), emb,
                f"smooth_k{ks}_a{al}+rr"))

    best = max(results, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best, ensure_ascii=False), flush=True)
    report = {"agent": "ml-trainer", "task": "thin per-query re-ranker",
              "w": args.w, "pool": args.pool, "results": results, "best": best}
    with open(os.path.join(out, "hubness_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


def _row_z(x):
    """Per-query z-normalisation so two score matrices are comparable."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.clip(sd, 1e-12, None)


if __name__ == "__main__":
    raise SystemExit(main())
