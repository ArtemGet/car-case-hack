#!/usr/bin/env python
"""Convert the external SigLIP2 NaFlex ONNX to fp16 and validate it.

The public bundle (``occurra/vehicle_reid_siglip2_naflex_512d``) ships an fp32
ONNX (360 MB, see ``runs/exp-0032-zs-siglip2/external_sources.json``). We convert
its **weights** to fp16 (I/O stays fp32, ``keep_io_types=True``) and verify the
conversion against the fp32 graph on real validation crops:

* ``max_abs_diff`` / ``cosine_min`` on the 512-d embeddings, and
* ``top10_overlap`` of each query's nearest gallery neighbours — the candidate-
  ranking red flag from the perf-engineer role card.

CLI::

    python -m reid.export.convert_siglip_fp16 \
        --src runs/external/vehicle_reid_siglip2_naflex_512d.onnx \
        --out artifacts/siglip2_fp16.onnx \
        --dataset "docs/<ds>/dataset" --query <val query.csv> --gallery <val gallery.csv>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

__all__ = ["convert_fp16", "validate_numeric"]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCH = 16
MAX_PATCHES = 256


def convert_fp16(src: str, out: str) -> str:
    """Weight-convert an fp32 ONNX to fp16 in place (I/O kept fp32)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    model = onnx.load(src)
    fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16, out)
    return out


def _session(path: str, device: str = "cuda"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.startswith("cuda") else ["CPUExecutionProvider"])
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _patchify(img, max_patches: int = MAX_PATCHES):
    """Natural-aspect PIL crop -> (256,768) patches, mask, (rows, cols)."""
    from PIL import Image
    w, h = img.size
    aspect = w / max(1, h)
    rows = max(1, int(round((max_patches / aspect) ** 0.5)))
    cols = max(1, int(round(max_patches / rows)))
    if rows * cols > max_patches:
        cols = max_patches // rows
    rows, cols = int(rows), int(cols)
    img = img.resize((cols * PATCH, rows * PATCH), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    patches = arr.reshape(rows, PATCH, cols, PATCH, 3).transpose(0, 2, 1, 3, 4)
    patches = patches.reshape(rows * cols, PATCH * PATCH * 3)
    pv = np.zeros((MAX_PATCHES, PATCH * PATCH * 3), np.float32)
    mask = np.zeros((MAX_PATCHES,), np.int64)
    pv[: patches.shape[0]] = patches
    mask[: patches.shape[0]] = 1
    return pv, mask, (rows, cols)


def _load_batch(csv_path: str, dataset_dir: str, limit: int):
    import pandas as pd
    from PIL import Image
    from reid.data.io import image_path

    df = pd.read_csv(csv_path)
    pvs, masks, shapes, keep = [], [], [], []
    for row in df.itertuples(index=False):
        p = image_path(dataset_dir, str(row.image_id))
        if not os.path.isfile(p):
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            ow, oh = im.size
            x0 = min(max(int(round(row.x)), 0), ow - 1)
            y0 = min(max(int(round(row.y)), 0), oh - 1)
            x1 = min(max(int(round(row.x + row.w)), x0 + 1), ow)
            y1 = min(max(int(round(row.y + row.h)), y0 + 1), oh)
            crop = im.crop((x0, y0, x1, y1))
        pv, m, sh = _patchify(crop)
        pvs.append(pv); masks.append(m); shapes.append(sh); keep.append(True)
        if limit and len(pvs) >= limit:
            break
    if not pvs:
        raise RuntimeError(f"no crops loaded from {csv_path}")
    return (np.stack(pvs), np.stack(masks).astype(np.int64),
            np.stack(shapes).astype(np.int64))


def _l2(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def _run(sess, pv, mask, shapes, chunk: int = 16):
    """Run the NaFlex ONNX in chunks (large batches blow up CUDA EP memory)."""
    names = [i.name for i in sess.get_inputs()]
    outs = []
    n = pv.shape[0]
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        feed = {names[0]: pv[s:e], names[1]: mask[s:e], names[2]: shapes[s:e]}
        outs.append(np.asarray(sess.run(None, feed)[0], dtype=np.float32))
    return np.concatenate(outs, axis=0)


def validate_numeric(src: str, out: str, dataset_dir: str, query_csv: str,
                     gallery_csv: str, n_query: int = 64, n_gallery: int = 256,
                     topk: int = 10, device: str = "cuda") -> dict:
    s32 = _session(src, device)
    s16 = _session(out, device)
    q = _load_batch(query_csv, dataset_dir, n_query)
    g = _load_batch(gallery_csv, dataset_dir, n_gallery)

    def emb(sess, batch):
        return _l2(_run(sess, *batch))

    eq32 = emb(s32, q); eg32 = emb(s32, g)
    eq16 = emb(s16, q); eg16 = emb(s16, g)
    diff_q = np.abs(_run(s32, *q) - _run(s16, *q))
    diff_g = np.abs(_run(s32, *g) - _run(s16, *g))

    sims32 = eq32 @ eg32.T
    sims16 = eq16 @ eg16.T
    k = min(topk, eg32.shape[0] - 1)
    o32 = np.argsort(-sims32, axis=1)[:, :k]
    o16 = np.argsort(-sims16, axis=1)[:, :k]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(o32, o16)]))

    return {
        "n_query": int(eq32.shape[0]), "n_gallery": int(eg32.shape[0]),
        "max_abs_diff": float(max(diff_q.max(), diff_g.max())),
        "mean_abs_diff": float((diff_q.mean() + diff_g.mean()) / 2),
        "cosine_min": float(min((eq32 * eq16).sum(1).min(), (eg32 * eg16).sum(1).min())),
        "top10_overlap": overlap,
        "dim": int(eq32.shape[1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SigLIP2 NaFlex ONNX -> fp16 + validate.")
    ap.add_argument("--src", default=os.path.join(
        REPO, "runs", "external", "vehicle_reid_siglip2_naflex_512d.onnx"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "siglip2_fp16.onnx"))
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--query", default=os.path.join(
        REPO, "runs", "exp-0032-zs-siglip2", "query.csv"))
    ap.add_argument("--gallery", default=os.path.join(
        REPO, "runs", "exp-0032-zs-siglip2", "gallery.csv"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    convert_fp16(args.src, args.out)
    print(f"converted: {args.out} ({os.path.getsize(args.out)/1024/1024:.1f} MB)")
    rep = {"src": args.src, "onnx": args.out,
           "size_mb": os.path.getsize(args.out) / 1024 / 1024}
    if args.dataset:
        rep["validation"] = validate_numeric(
            args.src, args.out, args.dataset, args.query, args.gallery,
            device=args.device)
        v = rep["validation"]
        print(f"validation: max_abs_diff={v['max_abs_diff']:.3e} "
              f"cosine_min={v['cosine_min']:.6f} top10_overlap={v['top10_overlap']:.3f}")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
