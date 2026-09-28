#!/usr/bin/env python
"""Export the in-house DINOv2-B champion (exp-0007) to ONNX fp16 and validate.

The checkpoint is a :class:`reid.models.ReIDModel` (``backbone dinov2_b`` @224,
GeM + Linear/BN/PReLU neck + BNNeck, emb_dim 512). For deployment only the
**embedding** path matters, so the ArcFace classifier is dropped; the exported
graph returns the 512-d BNNeck output (L2 normalisation stays downstream).

Validation is not a toy tensor comparison: the ONNX model and the PyTorch model
embed the *same real validation crops*, and we report

* ``max_abs_diff`` / ``mean_abs_diff`` / ``cosine_min`` on those embeddings, and
* ``top10_overlap`` — the mean fraction of shared gallery indices in each
  query's top-10 nearest-neighbour list. A drop here is the perf-engineer red
  flag ("different candidate ranking"), even when the numeric diff is small.

CLI::

    python -m reid.export.export_dino_onnx --model runs/exp-0007/best.pt \
        --out artifacts/dinov2_b_fp16.onnx --fp16 \
        --dataset "docs/<ds>/dataset" --query <val query.csv> --gallery <val gallery.csv>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.export.export_onnx import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _convert_fp16,
    _session,
)
from reid.data.crop import crop_vehicle

__all__ = [
    "DinoEmbedder",
    "build_dino_embedder",
    "export_dino",
    "validate_retrieval",
]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DinoEmbedder(nn.Module):
    """Inference-only embedding path of :class:`ReIDModel` (no ArcFace)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.gem = model.gem
        self.neck = model.neck
        self.bnneck = model.bnneck
        self.gem_eps = float(getattr(model.gem, "eps", 1e-6))

    def _gem(self, x: torch.Tensor) -> torch.Tensor:
        # Same GeM as reid.models.head.GeM, but with adaptive_avg_pool2d(.,1)
        # instead of avg_pool2d(x, (x.size(-2), x.size(-1))): the latter's
        # dynamic kernel does not export to ONNX with the legacy tracer.
        p = self.gem.p
        x = x.float().clamp(min=self.gem_eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x.pow(1.0 / p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bnneck(self.neck(self._gem(self.backbone(x))))


def build_dino_embedder(ckpt_path: str, device: str = "cpu",
                        pretrained: bool = False) -> DinoEmbedder:
    """Rebuild the exp-0007 ReIDModel from its ``best.pt`` and strip ArcFace."""
    from reid.models import build_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config", {}))
    sd = ck["state_dict"]
    num_classes = int(sd["arcface.weight"].shape[0]) if "arcface.weight" in sd else 1
    model = build_model(
        backbone=cfg.get("backbone", "dinov2_b"),
        num_classes=num_classes,
        emb_dim=int(ck.get("emb_dim", cfg.get("emb_dim", 512))),
        pretrained=pretrained,
        margin=float(cfg.get("margin", 0.3)),
        scale=float(cfg.get("scale", 30.0)),
        gem_p=float(cfg.get("gem_p", 3.0)),
        image_size=int(cfg.get("image_size", 224)),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("arcface.")]
    if missing:
        raise RuntimeError(f"checkpoint load mismatch (missing): {missing[:5]}")
    model.eval()
    return DinoEmbedder(model).to(device).eval()


def export_dino(ckpt_path: str, out_path: str, input_size: int = 224,
                opset: int = 17, fp16: bool = False, dynamic_batch: bool = True,
                device: str = "cpu") -> str:
    """Export the DINOv2 embedder to ONNX (optionally fp16 weights)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    model = build_dino_embedder(ckpt_path, device=device, pretrained=False)
    dummy = torch.randn(1, 3, input_size, input_size)
    dynamic_axes = ({"input": {0: "batch"}, "embedding": {0: "batch"}}
                    if dynamic_batch else None)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["input"], output_names=["embedding"],
            dynamic_axes=dynamic_axes, opset_version=opset,
            do_constant_folding=True, dynamo=False,
        )
    if fp16:
        _convert_fp16(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Real-image validation
# --------------------------------------------------------------------------- #
def _load_crops(csv_path: str, dataset_dir: str, size: int, limit: int):
    import pandas as pd
    from reid.data.io import image_path

    df = pd.read_csv(csv_path)
    items = []
    for row in df.itertuples(index=False):
        p = image_path(dataset_dir, str(row.image_id))
        if not os.path.isfile(p):
            continue
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB")
            crop = crop_vehicle(im, int(row.x), int(row.y), int(row.w),
                                int(row.h), target=size)
        items.append(crop)
        if limit and len(items) >= limit:
            break
    return items


def _to_tensor(crops):
    from torchvision.transforms import functional as TF
    ts = [TF.normalize(TF.to_tensor(c), IMAGENET_MEAN, IMAGENET_STD)
          for c in crops]
    return torch.stack(ts, dim=0)


def _top10_overlap(ref: np.ndarray, got: np.ndarray, topk: int = 10) -> float:
    def l2(a):
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    r, g = l2(ref.astype(np.float64)), l2(got.astype(np.float64))
    sim_r = r @ r.T
    sim_g = g @ g.T
    np.fill_diagonal(sim_r, -np.inf)
    np.fill_diagonal(sim_g, -np.inf)
    k = min(topk, r.shape[0] - 1)
    if k <= 0:
        return 1.0
    orr = np.argsort(-sim_r, axis=1)[:, :k]
    org = np.argsort(-sim_g, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(orr, org)]))


def validate_retrieval(ckpt_path: str, onnx_path: str, dataset_dir: str,
                       query_csv: str, gallery_csv: str, input_size: int = 224,
                       n_query: int = 64, n_gallery: int = 256,
                       device: str = "cuda") -> dict:
    """Embed real crops with PyTorch and ONNX; compare numbers AND ranking."""
    model = build_dino_embedder(ckpt_path, device="cpu", pretrained=False)
    dev = torch.device(device)
    model = model.to(dev).eval()

    crops_q = _load_crops(query_csv, dataset_dir, input_size, n_query)
    crops_g = _load_crops(gallery_csv, dataset_dir, input_size, n_gallery)
    if not crops_q or not crops_g:
        raise RuntimeError("no crops loaded for validation")

    xq = _to_tensor(crops_q)
    xg = _to_tensor(crops_g)
    x = torch.cat([xq, xg], dim=0)
    with torch.no_grad():
        ref = model(x.to(dev)).detach().float().cpu().numpy()

    sess = _session(onnx_path, device=device)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    got = np.asarray(sess.run([out_name], {in_name: x.numpy().astype(np.float32)})[0],
                     dtype=np.float32)

    diff = np.abs(ref - got)
    rn = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
    gn = got / np.clip(np.linalg.norm(got, axis=1, keepdims=True), 1e-12, None)
    cosine = (rn * gn).sum(axis=1)

    nq = len(crops_q)
    overlap = _top10_overlap(ref[nq:], got[nq:], topk=min(10, nq))
    return {
        "n_query": nq,
        "n_gallery": len(crops_g),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_min": float(cosine.min()),
        "top10_overlap": float(overlap),
        "dim": int(ref.shape[1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export DINOv2-B champion to ONNX.")
    ap.add_argument("--model", default=os.path.join(REPO, "runs", "exp-0007", "best.pt"))
    ap.add_argument("--out", default=os.path.join(REPO, "artifacts", "dinov2_b_fp16.onnx"))
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--query", default=os.path.join(REPO, "runs", "exp-0007", "val_ema", "query.csv"))
    ap.add_argument("--gallery", default=os.path.join(REPO, "runs", "exp-0007", "val_ema", "gallery.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    export_dino(args.model, args.out, input_size=args.input_size,
                opset=args.opset, fp16=args.fp16, device="cpu")
    print(f"exported: {args.out} ({os.path.getsize(args.out)/1024/1024:.1f} MB)")

    rep = {"onnx": args.out, "fp16": args.fp16,
           "size_mb": os.path.getsize(args.out) / 1024 / 1024}
    if args.dataset:
        rep["validation"] = validate_retrieval(
            args.model, args.out, args.dataset, args.query, args.gallery,
            input_size=args.input_size, device=args.device)
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
