#!/usr/bin/env python
"""End-to-end check: does the deployed ONNX fusion reproduce exp-0038's mAP?

The perf bench proves the fused pipeline is fast; this proves it is still
*correct*. It runs the exact deployment backends (DINOv2 fp16 ONNX + SigLIP2
fp16 ONNX, partial JPEG decode, shared crop) over the frozen hold-out val split,
writes submission/candidates/embeddings, and scores them with the official
``evaluate.py`` — baseline cosine and per-query k-reciprocal, the exp-0038 recipe.

Reference (exp-0038, PyTorch extractors): baseline mAP@10 0.6452, rerank 0.6842.

CLI::

    python -m reid.export.verify_fusion_onnx --dino-model artifacts/dinov2_b_fp16.onnx \\
        --siglip-model artifacts/siglip2_fp16.onnx \\
        --val-dir runs/exp-0038-ens \
        --dataset "docs/<ds>/dataset" --out runs/verify-fusion-onnx \\
        --json reports/verify_fusion_onnx.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from reid.data.io import image_path
from reid.eval.harness import run_official

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rows(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for r in df.itertuples(index=False):
        items.append((r.image_id, int(r.x), int(r.y), int(r.w), int(r.h)))
    return df, items


def _flat(rep):
    r = rep.get("ranking", {}); fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"), "mAP_full": fr.get("mAP_full"),
            "mINP": fr.get("mINP"), "n_scored": r.get("n_scored"),
            "n_openset_excluded": r.get("n_openset_excluded")}


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Verify ONNX fusion vs exp-0038.")
    ap.add_argument("--dino-model", default=os.path.join(REPO, "artifacts", "dinov2_b_fp16.onnx"))
    ap.add_argument("--siglip-model", default=os.path.join(REPO, "artifacts", "siglip2_fp16.onnx"))
    ap.add_argument("--val-dir", default=os.path.join(REPO, "runs", "exp-0038-ens"))
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--fusion-w", type=float, default=0.6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k1", type=int, default=6)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "verify-fusion-onnx"))
    ap.add_argument("--json", default=os.path.join(REPO, "reports", "verify_fusion_onnx.json"))
    args = ap.parse_args(argv)
    if not args.dataset:
        print("ERROR: --dataset required", file=sys.stderr)
        return 2

    from tools.bench_perf import DinoOnnxBackend, SigLip2OnnxBackend, FusionBackend
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino = DinoOnnxBackend(args.dino_model, device)
    sig = SigLip2OnnxBackend(args.siglip_model, device)
    backend = FusionBackend(dino, sig, w=args.fusion_w)

    qdf, qrows = _rows(os.path.join(args.val_dir, "query.csv"))
    gdf, grows = _rows(os.path.join(args.val_dir, "gallery.csv"))
    rows = qrows + grows
    nq = len(qrows)
    print(f"val: query={nq} gallery={len(grows)} batch={args.batch_size}", flush=True)

    embs = []
    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = [(image_path(args.dataset, r[0]), (r[1], r[2], r[3], r[4]))
                 for r in rows[i:i + args.batch_size]]
        embs.append(backend.extract(batch))
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    q, g = emb[:nq], emb[nq:]
    print(f"extracted {emb.shape} in {time.time() - t0:.0f}s", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    gt = os.path.join(out, "gt.csv")
    if not os.path.exists(gt):
        gt = os.path.join(args.val_dir, "gt.csv")
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    qdf.to_csv(qcsv, index=False); gdf.to_csv(gcsv, index=False)
    np.save(os.path.join(out, "embeddings.npy"), emb)
    sub = os.path.join(out, "submission.csv"); cand = os.path.join(out, "candidates.csv")
    g_ids = gdf["image_id"].tolist()

    def write(sub_orders, sub_scores):
        with open(sub, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for i, qid in enumerate(qdf["image_id"]):
                w.writerow([qid] + [g_ids[j] for j in sub_orders[i, :10]])
        with open(cand, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["query_id", "gallery_id", "confidence"])
            for i, qid in enumerate(qdf["image_id"]):
                j = int(sub_orders[i, 0]); w.writerow([qid, g_ids[j], f"{float(sub_scores[i, 0]):.6f}"])

    sims = q @ g.T
    orders = np.argsort(-sims, axis=1, kind="stable")
    write(orders, sims)
    base = _flat(run_official(gt_csv=gt, submission=sub, candidates=cand,
                              embeddings=os.path.join(out, "embeddings.npy"),
                              query=qcsv, gallery=gcsv,
                              json_out=os.path.join(out, "official_baseline.json")))
    print("ONNX fusion baseline", json.dumps(base, ensure_ascii=False), flush=True)

    from reid import rerank
    o = np.empty((nq, len(g)), np.int64); sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        prep = rerank.prepare_query(q[i], g, pool_size=args.pool)
        o[i], sc[i] = rerank.rank_prepared(prep, k1=args.k1, k2=args.k2, lam=args.lam)
    write(o, sc)
    rr = _flat(run_official(gt_csv=gt, submission=sub, candidates=cand,
                            embeddings=os.path.join(out, "embeddings.npy"),
                            query=qcsv, gallery=gcsv,
                            json_out=os.path.join(out, "official_rerank.json")))
    print("ONNX fusion rerank  ", json.dumps(rr, ensure_ascii=False), flush=True)

    report = {
        "agent": "perf-engineer", "task": "W2-3 ONNX fusion verification",
        "dino_model": args.dino_model, "siglip_model": args.siglip_model,
        "fusion_w": args.fusion_w, "config": {"pool": args.pool, "k1": args.k1,
                                              "k2": args.k2, "lam": args.lam},
        "baseline": base, "rerank": rr,
        "reference_exp0038": {"baseline_mAP@10": 0.6451957806932296,
                              "rerank_mAP@10": 0.6841663742216464},
        "val": {"n_query": nq, "n_gallery": len(g)},
        "latency_extract_s": time.time() - t0,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
