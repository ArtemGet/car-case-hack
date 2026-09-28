"""service.infer.val_eval — val-прогон раннера + официальные метрики (W2-5).

Прогоняет ТОТ ЖЕ пайплайн, что и :mod:`service.infer.run`, но на
замороженном hold-out val-сплите (``reid.data.splits.holdout_val``, seed 42),
считает порог отказа по формуле ``reid.calibrate`` и оценивает результат
официальным ``evaluate.py`` через ``reid.eval.harness`` (единственный источник
истины по метрикам).

Также умеет проверять детерминизм: повторный полный прогон и побайтовое
сравнение трёх файлов.

Пример:
    python -m service.infer.val_eval --dataset-dir "docs/<ds>/dataset" \
        --variant fusion --out runs/W2-5-val-fusion \
        --json reports/infer_val.json --check-determinism
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from reid.calibrate import _labels, calibrate
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official

from . import run as runner
from .backends import set_determinism

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_gt(val_query, val_gallery, path):
    import pandas as pd
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy()
    q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy()
    g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def _flat(rep):
    r = rep.get("ranking", {})
    fr = rep.get("full_ranking", {})
    c = rep.get("candidates", {})
    return {
        "mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
        "Rank-5": r.get("Rank-5"), "mAP_full": fr.get("mAP_full"),
        "mINP": fr.get("mINP"), "n_scored": r.get("n_scored"),
        "n_openset_excluded": r.get("n_openset_excluded"),
        "F1": c.get("F1"), "TNR": c.get("TNR"), "PR-AUC": c.get("PR-AUC"),
        "TP": c.get("TP"), "FP": c.get("FP"), "FN": c.get("FN"), "TN": c.get("TN"),
        "score_0.7F1+0.3TNR": (0.7 * c["F1"] + 0.3 * c["TNR"]
                               if c.get("F1") is not None and c.get("TNR") is not None
                               else None),
    }


def evaluate(variant, dataset_dir, out_dir, *, dino_ckpt=None,
             dino_onnx=None, dino_onnx_280=None,
             siglip_weights=None, dino_tta=None, w=runner.FUSION_W,
             batch_size=64, num_workers=0, siglip_threads=8, device="auto",
             crop_cache=None, seed=42, preproc=runner.DEFAULT_PREPROC,
             draft_factor=1.0, stage_size=0, allow_train_preproc=False):
    """Run the deployed pipeline on val, calibrate threshold, official metrics.

    ``crop_cache`` defaults to ``None``: the train_320 cache is train-only and
    inflated val by ~+0.014 — the runner guards against it (see
    ``runner._guard_train_preproc``). ``preproc`` defaults to the deploy GPU path.
    """
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(dataset_dir, "images")

    train = read_csv(os.path.join(dataset_dir, "train.csv"),
                     required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(train, val_fraction=0.2, open_set_fraction=0.2,
                            seed=seed)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    n_q, n_g = len(vq), len(vg)
    runner._log(f"[val] query={n_q} gallery={n_g} variant={variant}")

    qcsv = os.path.join(out_dir, "query.csv")
    gcsv = os.path.join(out_dir, "gallery.csv")
    gt = os.path.join(out_dir, "gt.csv")
    vq.to_csv(qcsv, index=False)
    vg.to_csv(gcsv, index=False)
    _write_gt(vq, vg, gt)

    t0 = time.time()
    q, g, q_ids, g_ids, info = runner.build_embeddings(
        variant, images_dir, vq, vg, device=device, dino_ckpt=dino_ckpt,
        dino_onnx=dino_onnx, dino_onnx_280=dino_onnx_280,
        siglip_weights=siglip_weights, dino_tta=dino_tta, w=w,
        batch_size=batch_size, num_workers=num_workers,
        siglip_threads=siglip_threads, crop_cache=crop_cache, seed=seed,
        preproc=preproc, draft_factor=draft_factor, stage_size=stage_size,
        allow_train_preproc=allow_train_preproc)
    extract_s = time.time() - t0

    orders, _rr, conf = runner.rank_embeddings(q, g, variant)
    has_match, top1_correct = _labels(vq, vg, orders, g_ids)
    n_openset = int(np.sum(~has_match))
    cal = calibrate(conf, has_match, top1_correct)
    chosen = cal["chosen"]
    runner._log(f"[val] open-set={n_openset}/{n_q} "
                f"chosen threshold={chosen['threshold']:.6f} "
                f"F1={chosen['F1']:.4f} TNR={chosen['TNR']:.4f}")

    winfo = runner.rank_and_write(out_dir, q, g, q_ids, g_ids, variant=variant,
                                  threshold=chosen["threshold"], top_k=10)

    # fp16 hand-off for calibration-agent: the fused embeddings as produced by
    # THIS fp16-ONNX pipeline (float16 copy) + the cross-query comparable cosine
    # top-1 confidence per query (score.npy / score.csv), so the threshold can be
    # recalibrated on the fp16 score scale without re-running the GPU extractor.
    emb_fp16 = os.path.join(out_dir, "embeddings_fp16.npy")
    score_npy = os.path.join(out_dir, "score.npy")
    score_csv = os.path.join(out_dir, "score.csv")
    np.save(emb_fp16, np.vstack([q, g]).astype(np.float16))
    np.save(score_npy, np.asarray(conf, dtype=np.float32))
    with open(score_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            w.writerow([qid, g_ids[int(orders[i, 0])], f"{float(conf[i]):.6f}"])

    official = run_official(
        gt_csv=gt, submission=os.path.join(out_dir, "submission.csv"),
        candidates=os.path.join(out_dir, "candidates.csv"),
        embeddings=os.path.join(out_dir, "embeddings.npy"),
        query=qcsv, gallery=gcsv,
        json_out=os.path.join(out_dir, "official_val.json"))
    metrics = _flat(official)
    runner._log(f"[val] official {json.dumps(metrics, ensure_ascii=False)}")

    report = {
        "task": "W2-5", "agent": "inference-service",
        "variant": variant, "split": {"seed": seed, "n_query": n_q,
                                      "n_gallery": n_g, "n_openset": n_openset},
        "config": {"tta_scales": info.get("tta_scales"),
                   "preproc": info.get("preproc"),
                   "gpu_desc": info.get("gpu_desc"),
                   "fusion_w": info.get("fusion_w"),
                   "rerank": runner.RERANK[variant],
                   "crop_cache": crop_cache,
                   "threshold": chosen["threshold"]},
        "extract_s": extract_s,
        "chosen": chosen,
        "metrics": metrics,
        "write": {k: winfo[k] for k in
                  ("accepted", "refused", "refusal_rate",
                   "score_cosine_top1", "artifacts")},
        "artifacts": {
            "dir": os.path.abspath(out_dir).replace("\\", "/"),
            "embeddings": os.path.join(out_dir, "embeddings.npy").replace("\\", "/"),
            "embeddings_fp16": emb_fp16.replace("\\", "/"),
            "score_npy": score_npy.replace("\\", "/"),
            "score_csv": score_csv.replace("\\", "/"),
        },
    }
    return report


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def check_determinism(variant, dataset_dir, out_a, out_b, **kw):
    """Rebuild into ``out_b`` and byte-compare the three artefacts with ``out_a``."""
    evaluate(variant, dataset_dir, out_b, **kw)
    res = {}
    ok = True
    names = ["submission.csv", "embeddings.npy", "candidates.csv"]
    # Hand-off extras: compare too when the first run produced them.
    names += [n for n in ("embeddings_fp16.npy", "score.npy")
              if os.path.exists(os.path.join(out_a, n))]
    for name in names:
        a = _sha(os.path.join(out_a, name))
        b = _sha(os.path.join(out_b, name))
        same = a == b
        ok = ok and same
        res[name] = {"sha_a": a, "sha_b": b, "identical": same}
    res["all_identical"] = ok
    return res


def _find_dataset_dir() -> str:
    """Locate the dataset dir (contains train.csv + images/) under docs/."""
    docs = os.path.join(REPO, "docs")
    if os.path.isdir(docs):
        for name in sorted(os.listdir(docs)):
            cand = os.path.join(docs, name)
            if (os.path.isfile(os.path.join(cand, "train.csv"))
                    and os.path.isdir(os.path.join(cand, "images"))):
                return cand
            ds = os.path.join(cand, "dataset")
            if (os.path.isfile(os.path.join(ds, "train.csv"))
                    and os.path.isdir(os.path.join(ds, "images"))):
                return ds
    raise SystemExit("не найден dataset-dir (train.csv+images) — укажите --dataset-dir")


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Val-прогон + официальные метрики (W2-5)")
    ap.add_argument("--dataset-dir", default=None,
                    help="каталог с train.csv+images/ (по умолчанию — автопоиск)")
    ap.add_argument("--variant", default="fusion", choices=list(runner.VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dino-ckpt", default=None)
    ap.add_argument("--dino-onnx", default=None)
    ap.add_argument("--dino-onnx-280", default=None)
    ap.add_argument("--siglip-weights", default=None)
    ap.add_argument("--dino-tta", default=None,
                    help="DINOv2 TTA-скейлы, напр. '224,280' (абляция); "
                         "по умолчанию TTA ВЫКЛЮЧЕНА — 224-only (variant A)")
    ap.add_argument("--w", type=float, default=runner.FUSION_W)
    ap.add_argument("--preproc", default=runner.DEFAULT_PREPROC,
                    choices=list(runner.PREPROC))
    ap.add_argument("--draft-factor", type=float, default=1.0)
    ap.add_argument("--stage-size", type=int, default=0,
                    help="[репродукция] 2-stage letterbox (train_320); "
                         "запрещён без --allow-train-preproc")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--siglip-threads", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--crop-cache", default=None,
                    help="[train-only, запрещён] кэш кропов train_320; "
                         "по умолчанию и для вала не используется")
    ap.add_argument("--allow-train-preproc", action="store_true",
                    help="аварийный обход защиты train_320 (только репродукция)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--determinism-out", default=None)
    args = ap.parse_args(argv)

    set_determinism(args.seed)
    dataset_dir = args.dataset_dir or _find_dataset_dir()
    runner._log(f"[val] dataset_dir={dataset_dir}")
    kw = dict(dino_ckpt=args.dino_ckpt, dino_onnx=args.dino_onnx,
              dino_onnx_280=args.dino_onnx_280,
              siglip_weights=args.siglip_weights,
              dino_tta=args.dino_tta, w=args.w, batch_size=args.batch_size,
              num_workers=args.num_workers, siglip_threads=args.siglip_threads,
              device=args.device, crop_cache=args.crop_cache, seed=args.seed,
              preproc=args.preproc, draft_factor=args.draft_factor,
              stage_size=args.stage_size,
              allow_train_preproc=args.allow_train_preproc)

    report = evaluate(args.variant, dataset_dir, args.out, **kw)
    if args.check_determinism:
        det_out = args.determinism_out or (args.out + "-b")
        runner._log(f"[determinism] second full run -> {det_out}")
        det = check_determinism(args.variant, dataset_dir, args.out, det_out, **kw)
        report["determinism"] = det
        runner._log(f"[determinism] all_identical={det['all_identical']}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        runner._log(f"[val] report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
