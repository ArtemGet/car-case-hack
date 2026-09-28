"""EX-4 CPU fusion grid: does a re-trained in-domain DINOv2 branch help fusion?

Compares, on the frozen hold-out val (seed 42) and exclusively through the
official ``evaluate.py``:

  champion : L2(concat[L2(sig), 0.8*L2(dino_exp0007)]) + k-reciprocal pool300
  challenger: same, but the DINOv2 branch is the newly trained checkpoint
              (its val bank produced by ``reid.models.val_bank``).

SigLIP2 bank and the split are identical to the champion, so the only variable
is the DINOv2 branch. Streaming invariant untouched (per-query k-reciprocal).

    python -m reid.models.ex4_fusion \
        --new-dino runs/exp-0078-ex4-triplet/val_features.npz \
        --out runs/exp-0078-ex4-triplet/fusion --json reports/exp-0078-ex4.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models.fusion_postproc import build_post, evaluate_cfg, flat, write_sub

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHAMP_W = 0.8
CHAMP_RR = {"k1": 8, "k2": 3, "lam": 0.5, "pool_size": 300}


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def eval_alone(feat, nq, ids, evaldir, rr_cfg, tag):
    """DINOv2-only ranking (no fusion): base cosine + k-reciprocal."""
    q_ids, g_ids, gt, qcsv, gcsv = ids
    z = l2(feat.astype(np.float32))
    q, g = np.ascontiguousarray(z[:nq]), np.ascontiguousarray(z[nq:])
    sub = os.path.join(evaldir, "submission.csv")
    emb = os.path.join(evaldir, "embeddings.npy")
    np.save(emb, np.vstack([q, g]).astype(np.float32))
    sims = q @ g.T
    o0 = np.argsort(-sims, axis=1, kind="stable")
    write_sub(sub, q_ids, g_ids, o0, sims)
    m_base = flat(run_official(gt_csv=gt, submission=sub,
                               candidates=os.path.join(evaldir, "candidates.csv"),
                               embeddings=emb, query=qcsv, gallery=gcsv,
                               json_out=os.path.join(evaldir, "official_base.json")))
    preps = [rerank.prepare_query(q[i], g, pool_size=rr_cfg["pool_size"])
             for i in range(nq)]
    o = np.empty((nq, len(g)), np.int64)
    sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=rr_cfg["k1"],
                                           k2=rr_cfg["k2"], lam=rr_cfg["lam"])
    write_sub(sub, q_ids, g_ids, o, sc)
    m_rr = flat(run_official(gt_csv=gt, submission=sub,
                             candidates=os.path.join(evaldir, "candidates.csv"),
                             embeddings=emb, query=qcsv, gallery=gcsv,
                             json_out=os.path.join(evaldir, "official_rr.json")))
    row = {"tag": tag, "algo": "alone", "base_mAP@10": m_base["mAP@10"],
           "mAP@10": m_rr["mAP@10"], "Rank-1": m_rr["Rank-1"],
           "Rank-5": m_rr["Rank-5"], "mINP": m_rr["mINP"],
           "base_mINP": m_base["mINP"]}
    print(f"ALONE {tag:44s} base={m_base['mAP@10']:.4f} rr={m_rr['mAP@10']:.4f} "
          f"R1={m_rr['Rank-1']:.4f} mINP={m_rr['mINP']:.4f}", flush=True)
    return row


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--old-dino-run", default="runs/W2-5-val-dino",
                    help="deployed (ONNX fp16) DINOv2 bank -> champion reference")
    ap.add_argument("--old-pt", default="",
                    help="PyTorch re-extraction of exp-0007 (fair control)")
    ap.add_argument("--new-dino", required=True,
                    help="npz with q_tta/g_tta (val_bank) or embeddings.npy")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    def _load(p):
        if p.endswith(".npz"):
            z = np.load(p)
            return np.vstack([z["q_tta"], z["g_tta"]]).astype(np.float32)
        return np.load(p).astype(np.float32)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    sig = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    old = np.load(os.path.join(args.old_dino_run, "embeddings.npy")).astype(np.float32)
    ctrl = _load(args.old_pt) if args.old_pt else old
    new = _load(args.new_dino)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    n = nq + len(vg)
    assert sig.shape[0] == old.shape[0] == ctrl.shape[0] == new.shape[0] == n, \
        (sig.shape, old.shape, ctrl.shape, new.shape, n)
    assert sig.shape[1] == old.shape[1] == ctrl.shape[1] == new.shape[1] == 512, \
        (sig.shape, old.shape, ctrl.shape, new.shape)

    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    # sanity: PyTorch re-extract (ctrl) vs deployed ONNX-fp16 bank
    cos = (l2(old) * l2(ctrl)).sum(1)
    print(f"[sanity] deployed-vs-pytorch-old cosine mean={cos.mean():.4f} "
          f"min={cos.min():.4f}", flush=True)

    evaldir = os.path.join(out, "eval")
    os.makedirs(evaldir, exist_ok=True)

    rows = []
    # (0) deployed bank -> official champion reference (fp16 ONNX component)
    rows.append(evaluate_cfg({"w": CHAMP_W}, sig, old, nq, ids, out, evaldir,
                             CHAMP_RR, f"deployed-bank w={CHAMP_W} pool300"))
    # (1) PyTorch exp-0007 bank -> fair control for the new model
    rows.append(evaluate_cfg({"w": CHAMP_W}, sig, ctrl, nq, ids, out, evaldir,
                             CHAMP_RR, f"pytorch-old w={CHAMP_W} pool300"))
    # (2) new DINOv2 alone
    rows.append(eval_alone(new, nq, ids, evaldir, CHAMP_RR, "new-dino alone pool300"))
    # (3) new DINOv2 fusion weight sweep at champion rr
    for w in (0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
        rows.append(evaluate_cfg({"w": w}, sig, new, nq, ids, out, evaldir,
                                 CHAMP_RR, f"new-dino w={w} pool300"))
    # (4) pool200 (pre-pool300 champion rr) at champion w, for reference
    rr200 = dict(CHAMP_RR); rr200["pool_size"] = 200
    rows.append(evaluate_cfg({"w": CHAMP_W}, sig, new, nq, ids, out, evaldir,
                             rr200, f"new-dino w={CHAMP_W} pool200"))

    deployed, control = rows[0], rows[1]
    challengers = rows[2:]
    best = max(challengers, key=lambda r: r["mAP@10"] or 0)
    delta = (best["mAP@10"] or 0) - (control["mAP@10"] or 0)
    delta_deployed = (best["mAP@10"] or 0) - (deployed["mAP@10"] or 0)
    report = {
        "agent": "ml-trainer", "task": "EX-4 in-domain retrain + fusion",
        "champion_w": CHAMP_W, "champion_rr": CHAMP_RR,
        "deployed_bank": deployed, "control_pytorch_old": control,
        "results": rows, "best_challenger": best,
        "delta_vs_control": delta, "delta_vs_deployed": delta_deployed,
        "verdict": "improved" if delta > 1e-6 else "not_improved",
    }
    with open(os.path.join(out, "ex4_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nDEPLOYED {deployed['mAP@10']:.5f}  CONTROL(pytorch old) "
          f"{control['mAP@10']:.5f}  BEST {best['mAP@10']:.5f}  "
          f"delta_ctrl={delta:+.5f} delta_deployed={delta_deployed:+.5f}  "
          f"verdict={report['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
