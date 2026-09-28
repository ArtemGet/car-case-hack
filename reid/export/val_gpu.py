#!/usr/bin/env python
"""Official val evaluation of the GPU-preproc fusion path (perf-engineer).

Runs the **exact** frozen post-processing contract of :mod:`service.infer.run`
(k-reciprocal pool300 + refusal threshold + 10-neighbour submission) but builds
the query/gallery embeddings through the GPU-preprocessing backends in
:mod:`tools.bench_perf` (``--preproc gpu``). Metrics come from the organisers'
``evaluate.py`` via :func:`reid.eval.harness.run_official` — the single source
of truth.

Purpose: prove that moving decode-crop/normalise from CPU PIL to CUDA does not
change val mAP, and get the official mAP for the TTA-economy variants
(``--dino-tta 224`` = drop the 280 graph).

CLI::

    python -m reid.export.val_gpu --dataset-dir "docs/<ds>/dataset" \\
        --variant fusion --dino-tta 224,280 --out runs/perf/val_gpu_tta \\
        --json reports/opt_gpu_val_tta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import service.infer.run as runner  # noqa: E402
from reid.data.io import TRAIN_COLUMNS, read_csv  # noqa: E402
from reid.data.splits import holdout_val  # noqa: E402
from reid.eval.harness import run_official  # noqa: E402
from service.infer.val_eval import _flat, _write_gt  # noqa: E402

DINO_ONNX = os.path.join("artifacts", "dinov2_b_fp16.onnx")
DINO_ONNX_280 = os.path.join("artifacts", "dinov2_b_280_fp16.onnx")
SIGLIP_ONNX = os.path.join("artifacts", "siglip2_fp16.onnx")


def _bench_args(variant, scales, w, draft_factor, preproc="gpu", stage_size=0):
    return SimpleNamespace(
        variant=variant, device="cuda", preproc=preproc,
        dino_model=os.path.join(REPO, DINO_ONNX),
        dino_model_280=os.path.join(REPO, DINO_ONNX_280),
        siglip_model=os.path.join(REPO, SIGLIP_ONNX),
        fusion_w=w, draft_factor=draft_factor, input_size=224,
        stage_size=int(stage_size),
        dino_tta=",".join(str(s) for s in scales) if scales else None,
    )


def _extract(backend, df, images_dir, batch_size=64, tag=""):
    import torch

    rows = list(df.itertuples(index=False))
    items = [(os.path.join(images_dir, f"{r.image_id}.jpg"),
              (int(r.x), int(r.y), int(r.w), int(r.h))) for r in rows]
    outs = []
    for i in range(0, len(items), batch_size):
        outs.append(backend.extract(items[i:i + batch_size]))
        if (i // batch_size) % 8 == 0:
            print(f"  [{tag}] {min(i + batch_size, len(items))}/{len(items)}",
                  flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return np.concatenate(outs, axis=0).astype(np.float32)


def evaluate(dataset_dir, out_dir, *, variant="fusion", scales=(224, 280),
             w=0.8, draft_factor=1.0, preproc="gpu", batch_size=64,
             threshold=None, seed=42, stage_size=0):
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(dataset_dir, "images")

    train = read_csv(os.path.join(dataset_dir, "train.csv"),
                     required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(train, val_fraction=0.2, open_set_fraction=0.2,
                            seed=seed)
    vq, vg = vq.reset_index(drop=True), vg.reset_index(drop=True)
    q_ids = vq["image_id"].astype(str).tolist()
    g_ids = vg["image_id"].astype(str).tolist()
    print(f"[val_gpu] query={len(vq)} gallery={len(vg)} variant={variant} "
          f"scales={list(scales)} preproc={preproc}", flush=True)

    # import here so CPU-only imports of this module stay cheap
    from tools.bench_perf import build_variant_backend

    backend, desc, weight_files = build_variant_backend(
        _bench_args(variant, scales, w, draft_factor, preproc, stage_size))
    print(f"[val_gpu] backend={desc}", flush=True)

    t0 = time.time()
    g = _extract(backend, vg, images_dir, batch_size, tag="gallery")
    q = _extract(backend, vq, images_dir, batch_size, tag="query")
    extract_s = time.time() - t0
    print(f"[val_gpu] extract {extract_s:.1f}s  q={q.shape} g={g.shape}",
          flush=True)

    thr = float(threshold) if threshold is not None else runner.resolve_threshold(
        "fusion" if variant == "fusion" else variant)
    winfo = runner.rank_and_write(out_dir, q, g, q_ids, g_ids, variant=variant,
                                  threshold=thr, top_k=10)

    gt = os.path.join(out_dir, "gt.csv")
    _write_gt(vq, vg, gt)
    qcsv, gcsv = os.path.join(out_dir, "query.csv"), os.path.join(out_dir, "gallery.csv")
    vq.to_csv(qcsv, index=False)
    vg.to_csv(gcsv, index=False)
    official = run_official(
        gt_csv=gt, submission=os.path.join(out_dir, "submission.csv"),
        candidates=os.path.join(out_dir, "candidates.csv"),
        embeddings=os.path.join(out_dir, "embeddings.npy"),
        query=qcsv, gallery=gcsv,
        json_out=os.path.join(out_dir, "official_val.json"))
    metrics = _flat(official)
    print(f"[val_gpu] official {json.dumps(metrics, ensure_ascii=False)}",
          flush=True)

    return {
        "task": "PERF-opt", "agent": "perf-engineer",
        "variant": variant, "preproc": preproc,
        "device": "NVIDIA GeForce RTX 4090",
        "config": {"tta_scales": list(scales), "fusion_w": w,
                   "rerank": runner.RERANK[variant] if variant in runner.RERANK else None,
                   "threshold": thr, "draft_factor": draft_factor,
                   "batch_size": batch_size, "stage_size": stage_size},
        "split": {"seed": seed, "n_query": len(vq), "n_gallery": len(vg)},
        "extract_s": extract_s, "metrics": metrics,
        "write": {k: winfo[k] for k in ("accepted", "refused", "refusal_rate")},
        "weights_mb": sum(os.path.getsize(p) for p in weight_files) / 1024 / 1024,
        "artifacts": {"dir": os.path.abspath(out_dir).replace("\\", "/")},
    }


def _find_dataset_dir():
    docs = os.path.join(REPO, "docs")
    for name in sorted(os.listdir(docs)):
        for cand in (os.path.join(docs, name), os.path.join(docs, name, "dataset")):
            if (os.path.isfile(os.path.join(cand, "train.csv"))
                    and os.path.isdir(os.path.join(cand, "images"))):
                return cand
    raise SystemExit("dataset-dir not found; pass --dataset-dir")


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="GPU-preproc fusion val eval")
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--variant", default="fusion")
    ap.add_argument("--dino-tta", default="224,280")
    ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--draft-factor", type=float, default=1.0)
    ap.add_argument("--preproc", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--stage-size", type=int, default=0,
                    help="GPU only: 2-stage letterbox (e.g. 320) then resize; "
                         "reproduces the train_320 crop-cache pipeline")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    scales = tuple(int(s) for s in args.dino_tta.split(",") if s.strip())
    rep = evaluate(args.dataset_dir or _find_dataset_dir(), args.out,
                   variant=args.variant, scales=scales, w=args.w,
                   draft_factor=args.draft_factor, preproc=args.preproc,
                   batch_size=args.batch_size, threshold=args.threshold,
                   stage_size=args.stage_size)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"[val_gpu] report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
