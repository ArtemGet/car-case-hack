"""Fuse an EXTERNAL vehicle-ReID extractor with our in-house champion + re-rank.

The external SigLIP2 NaFlex model (public weights, see
``runs/exp-0033-zs-siglip2/external_sources.json``) is a far stronger frozen
extractor than our DINOv2-B (val mAP@10 0.602 vs 0.419). Their errors are only
partly correlated, so L2-normalising each and concatenating with a weight ``w``
improves retrieval further; per-query k-reciprocal re-ranking then adds its usual
gain. Everything is scored with the OFFICIAL ``evaluate.py`` on the frozen
hold-out val split (seed 42). No hflip; each query independent.

    python -m reid.models.ensemble_eval \
        --sig-run runs/exp-0032-zs-siglip2 \
        --dino-run runs/exp-0007/val_ema \
        --w 0.6 --pool 200 --k1 8 --k2 3 --lam 0.6 \
        --dataset "docs/<ds>/dataset" --out runs/exp-0039-ens \
        --json reports/exp-0039.json
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def flat(rep):
    r = rep.get("ranking", {}); fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"), "Rank-5": r.get("Rank-5"),
            "mAP_full": fr.get("mAP_full"), "mINP": fr.get("mINP"),
            "n_scored": r.get("n_scored"), "n_openset_excluded": r.get("n_openset_excluded")}


def write_sub(path, q_ids, g_ids, orders, scores, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])
    with open(os.path.join(os.path.dirname(path), "candidates.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0]); w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", required=True)
    ap.add_argument("--dino-run", required=True)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not args.dataset:
        print("ERROR: --dataset required", file=sys.stderr); return 2

    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    # SigLIP2 run may hold raw_embeddings.npy (unmodified by re-rank) or embeddings.npy
    sig_path = os.path.join(args.sig_run, "raw_embeddings.npy")
    if not os.path.exists(sig_path):
        sig_path = os.path.join(args.sig_run, "embeddings.npy")
    sig = np.load(sig_path).astype(np.float32)
    dino_path = os.path.join(args.dino_run, "embeddings.npy")
    dino = np.load(dino_path).astype(np.float32)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq)
    assert sig.shape[0] == dino.shape[0] == nq + len(vg), (sig.shape, dino.shape)
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)

    sn, dn = l2(sig), l2(dino)
    q = l2(np.concatenate([sn[:nq], args.w * dn[:nq]], axis=1))
    g = l2(np.concatenate([sn[nq:], args.w * dn[nq:]], axis=1))

    # baseline (raw cosine on fused embeddings)
    sims = q @ g.T
    o0 = np.argsort(-sims, axis=1, kind="stable")
    sub = os.path.join(out, "submission.csv")
    write_sub(sub, q_ids, g_ids, o0, sims)
    np.save(os.path.join(out, "embeddings.npy"), np.vstack([q, g]).astype(np.float32))
    m_base = flat(run_official(gt_csv=gt, submission=sub,
                               candidates=os.path.join(out, "candidates.csv"),
                               embeddings=os.path.join(out, "embeddings.npy"),
                               query=qcsv, gallery=gcsv,
                               json_out=os.path.join(out, "official_baseline.json")))
    print("fused baseline", json.dumps(m_base, ensure_ascii=False), flush=True)

    # per-query k-reciprocal
    preps = [rerank.prepare_query(q[i], g, pool_size=args.pool) for i in range(nq)]
    o = np.empty((nq, len(g)), np.int64); sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=args.k1, k2=args.k2, lam=args.lam)
    write_sub(sub, q_ids, g_ids, o, sc)
    m_rr = flat(run_official(gt_csv=gt, submission=sub,
                             candidates=os.path.join(out, "candidates.csv"),
                             embeddings=os.path.join(out, "embeddings.npy"),
                             query=qcsv, gallery=gcsv,
                             json_out=os.path.join(out, "official_rerank.json")))
    print("fused rerank", json.dumps(m_rr, ensure_ascii=False), flush=True)

    # streaming invariant on the fused raw embeddings
    ok = True
    for t in (0, 5, 17):
        t = t % nq
        single = rerank.k_reciprocal_order(q[t], g, k1=args.k1, k2=args.k2, lam=args.lam,
                                           pool_size=args.pool)[0]
        perm = np.random.default_rng(7).permutation(nq)
        nt = int(np.where(perm == t)[0][0])
        shuf = rerank.k_reciprocal_order(q[perm][nt], g, k1=args.k1, k2=args.k2, lam=args.lam,
                                         pool_size=args.pool)[0]
        ok = ok and np.array_equal(single, shuf) and np.array_equal(single, shuf)
    print("streaming invariant:", "OK" if ok else "FAIL", flush=True)

    report = {"agent": "ml-trainer", "task": "external-pretrain ensemble",
              "sig_source": sig_path, "dino_source": dino_path,
              "config": {"w": args.w, "pool": args.pool, "k1": args.k1, "k2": args.k2,
                         "lam": args.lam},
              "baseline": m_base, "rerank": m_rr, "streaming_ok": bool(ok),
              "split": {"seed": 42, "n_query": nq, "n_gallery": len(vg)}}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
