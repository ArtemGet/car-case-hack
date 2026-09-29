"""tools/calibrate_siglip_cpu.py — CPU-only refusal-threshold calibration for the
SigLIP2 demo path (service.infer.cpu_backend, CPUExecutionProvider, PIL preproc).

This is the CPU counterpart of ``reid.calibrate``'s GPU calibration. It builds the
same frozen val split (``reid.data.splits.holdout_val(train.csv, 0.2, 0.2, 42)``:
375 query / 1528 gallery / 81 open-set), extracts L2-normalised SigLIP2 CPU
embeddings with the torch-free CPU backend, scores each query by the **pure
cosine top-1** (no re-rank — exactly what ``service/api`` does on CPU), and picks
the threshold maximising ``0.7*F1 + 0.3*TNR``. F1/TNR are then re-checked with the
official ``evaluate.py``.

Fully offline. Never touches CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_SIGLIP_ONNX = os.path.join("artifacts", "siglip2_fp16.onnx")


def resolve_dataset(dataset: str) -> str:
    p = os.path.abspath(dataset)
    if os.path.exists(os.path.join(p, "train.csv")):
        return p
    for n in os.listdir(p):
        cand = os.path.join(p, n)
        if os.path.isdir(cand):
            if os.path.exists(os.path.join(cand, "train.csv")):
                return cand
            for m in os.listdir(cand):
                deep = os.path.join(cand, m)
                if os.path.isdir(deep) and os.path.exists(
                        os.path.join(deep, "train.csv")):
                    return deep
    raise SystemExit(f"train.csv не найден в {p}")


def _resolve_weights(explicit: str | None) -> str:
    p = os.path.abspath(explicit) if explicit else os.path.join(
        _ROOT, DEFAULT_SIGLIP_ONNX)
    if not os.path.exists(p):
        raise SystemExit(f"не найден SigLIP2 ONNX: {p}")
    return p


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=os.path.join("runs", "siglip-cpu-calib"))
    ap.add_argument("--json", default=os.path.join(
        "reports", "calibration_siglip_cpu.json"))
    ap.add_argument("--siglip-weights", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from reid.data.io import TRAIN_COLUMNS, read_csv
    from reid.data.splits import holdout_val
    from reid import calibrate as C

    ds = resolve_dataset(args.dataset)
    train = read_csv(os.path.join(ds, "train.csv"), required=TRAIN_COLUMNS)
    _train, vq, vg = holdout_val(train, val_fraction=0.2,
                                 open_set_fraction=0.2, seed=args.seed)
    q_ids = vq["image_id"].tolist()
    g_ids = vg["image_id"].tolist()
    n_q, n_g = len(vq), len(vg)
    print(f"[calib] dataset={ds}")
    print(f"[calib] val query={n_q} gallery={n_g}", flush=True)

    run_dir = os.path.join(os.path.abspath(args.out), "val")
    os.makedirs(run_dir, exist_ok=True)
    emb_path = os.path.join(run_dir, "embeddings.npy")
    vq.to_csv(os.path.join(run_dir, "query.csv"), index=False)
    vg.to_csv(os.path.join(run_dir, "gallery.csv"), index=False)

    extract_s = None
    if os.path.exists(emb_path):
        emb = np.load(emb_path).astype(np.float32)
        if emb.shape[0] != n_q + n_g:
            raise SystemExit(f"стale embeddings {emb.shape} != {n_q}+{n_g}")
        print(f"[calib] reuse embeddings {emb_path} {emb.shape}")
    else:
        from service.infer.cpu_backend import (SiglipCpuBackend, rss_mb,
                                               set_cpu_determinism)
        set_cpu_determinism(args.seed)
        wi = _resolve_weights(args.siglip_weights)
        print(f"[calib] SigLIP2 CPU weights={wi} threads={args.threads} "
              f"batch={args.batch_size} rss={rss_mb():.0f}MB", flush=True)
        t0 = time.time()
        be = SiglipCpuBackend(wi, threads=args.threads)
        q = be.extract(vq, os.path.join(ds, "images"),
                       batch_size=args.batch_size)
        g = be.extract(vg, os.path.join(ds, "images"),
                       batch_size=args.batch_size)
        extract_s = time.time() - t0
        emb = np.vstack([q, g]).astype(np.float32)
        np.save(emb_path, emb)
        print(f"[calib] extract {extract_s:.1f}s "
              f"({extract_s / (n_q + n_g) * 1000.0:.0f} ms/frame) "
              f"rss={rss_mb():.0f}MB dim={emb.shape[1]}", flush=True)

    q = C.rerank.l2norm(emb[:n_q])
    g = C.rerank.l2norm(emb[n_q:])

    # Pure cosine ordering (NO re-rank): argsort of the cosine matrix, best first.
    sims = q @ g.T
    order = np.argsort(-sims, axis=1).astype(np.int64)

    has_match, top1_correct = C._labels(vq, vg, order, g_ids)
    n_openset = int(np.sum(~has_match))
    scores = C.score_confidences(q, g, order, mode="cosine").astype(np.float64)

    report = C.calibrate(scores, has_match, top1_correct)
    chosen = report["chosen"]
    curve = report["curve"]
    print(f"[calib] chosen thr={chosen['threshold']:.10f} "
          f"F1={chosen['F1']:.6f} TNR={chosen['TNR']:.6f} "
          f"score={chosen['score_0.7F1+0.3TNR']:.6f} "
          f"TP/FP/FN/TN={chosen['TP']}/{chosen['FP']}/{chosen['FN']}/"
          f"{chosen['TN']} open-set={n_openset}", flush=True)

    # --- artifacts + official re-check --------------------------------------
    gt = os.path.join(run_dir, "gt.csv")
    sub = os.path.join(run_dir, "submission.csv")
    cand = os.path.join(run_dir, "candidates.csv")
    C._write_gt(vq, vg, gt)
    C._write_submission(sub, q_ids, g_ids, order)
    n_written = C.write_candidates(cand, q_ids, g_ids, order, scores,
                                   chosen["threshold"])
    official = C.run_official(gt_csv=gt, submission=sub, candidates=cand,
                              embeddings=emb_path,
                              query=os.path.join(run_dir, "query.csv"),
                              gallery=os.path.join(run_dir, "gallery.csv"),
                              json_out=os.path.join(
                                  run_dir, "official_calibrated.json"))
    oc = official.get("candidates", {})
    print(f"[calib] candidates rows={n_written} "
          f"official F1={oc.get('F1')} TNR={oc.get('TNR')}", flush=True)

    out = {
        "task": "CPU_SIGLIP2_CALIB",
        "agent": "calibration-agent",
        "variant": "siglip_cpu",
        "pipeline": ("SigLIP2 ONNX (CPUExecutionProvider) + PIL aspect-preserving "
                     "preproc (service.infer.cpu_backend); score = pure cosine "
                     "top-1, NO re-rank (matches service/api CPU)"),
        "config": {
            "siglip_onnx": _resolve_weights(args.siglip_weights),
            "threads": args.threads,
            "batch_size": args.batch_size,
            "preproc": "PIL (service.infer.cpu_backend)",
            "rerank": None,
            "score_mode": "cosine",
        },
        "split": {"seed": args.seed, "n_query": n_q, "n_gallery": n_g,
                  "n_openset": n_openset},
        "selection_rule": "max 0.7*F1 + 0.3*TNR (tie: F1, then TNR, then lower thr)",
        "chosen": chosen,
        "pr_auc_raw": report["pr_auc_raw"],
        "pr_auc_official": oc.get("PR-AUC"),
        "curve": curve,
        "official_calibrated": official,
        "n_candidates_rows": n_written,
        "extract_time_s": extract_s,
        "artifacts": {
            "run_dir": run_dir.replace("\\", "/"),
            "candidates": cand.replace("\\", "/"),
            "submission": sub.replace("\\", "/"),
            "embeddings": emb_path.replace("\\", "/"),
        },
        "verdict": "pending",
    }
    json_path = os.path.abspath(args.json)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[calib] report -> {json_path}", flush=True)
    return 0 if abs(chosen["F1"] - (oc.get("F1") or 0)) < 1e-9 else 3


if __name__ == "__main__":
    raise SystemExit(main())
