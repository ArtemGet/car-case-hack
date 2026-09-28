#!/usr/bin/env python
"""Export a PyTorch vehicle-ReID embedding model to ONNX and validate it.

Contract (docs/_workspace/INTERFACES.md, METRICS.md §«Производительность»):

* input  : ``float32`` NCHW, ImageNet-normalised, square crop (default 224);
* output : ``(N, D)`` embedding (L2 normalisation happens downstream);
* the batch axis is dynamic (``dynamic_axes``), so the same graph serves the
  ``batch 1/8/16/32`` throughput protocol;
* optional **fp16** weight conversion (I/O stays fp32, cast is internal);
* validation reports the **max abs diff** against PyTorch and the **top-k
  neighbour overlap** (a different candidate ranking is a red flag for
  perf-engineer, see the role card).

CLI::

    python -m reid.export.export_onnx --model runs/<id>/best.pt --out artifacts/model.onnx [--fp16]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "EmbeddingModel",
    "build_embedder",
    "export_onnx",
    "validate_onnx",
    "default_input",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class EmbeddingModel(nn.Module):
    """Backbone + flatten head producing an ``(N, D)`` embedding.

    The classification head of the torchvision backbone is dropped; the global
    average-pooled feature is flattened. For ``resnet*`` this is a 512-d vector.
    """

    def __init__(self, backbone: nn.Module, out_dim: int | None = None):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            self.out_dim = self.backbone.fc.in_features if out_dim is None else out_dim
            self.backbone.fc = nn.Identity()
        else:
            self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return torch.flatten(x, 1)


def build_embedder(arch: str = "resnet18", pretrained: bool = False,
                   out_dim: int | None = None) -> EmbeddingModel:
    """Build a torchvision backbone wrapped as an embedding model.

    ``pretrained=False`` keeps export offline/deterministic (no download); the
    benchmark's ``--dummy`` path uses exactly this model.
    """
    from torchvision import models

    if not hasattr(models, arch):
        raise ValueError(f"unknown torchvision arch: {arch!r}")
    weights = "DEFAULT" if pretrained else None
    backbone = getattr(models, arch)(weights=weights)
    return EmbeddingModel(backbone, out_dim=out_dim)


def default_input(input_size: int = 224, batch: int = 1,
                  seed: int = 0) -> torch.Tensor:
    """Deterministic dummy NCHW input for tracing / validation."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 3, input_size, input_size, generator=g)


def export_onnx(model: nn.Module, out_path: str, input_size: int = 224,
                opset: int = 17, fp16: bool = False, dynamic_batch: bool = True,
                device: str = "cpu", seed: int = 0) -> str:
    """Trace ``model`` to ONNX and write it to ``out_path``.

    Returns the written path (``.onnx``). When ``fp16=True`` a float32 graph is
    first exported and then weight-converted in place; I/O stays float32.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    dev = torch.device(device)
    model = model.eval().to(dev)
    dummy = default_input(input_size, batch=1, seed=seed).to(dev)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "embedding": {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    if fp16:
        _convert_fp16(out_path)
    return out_path


def _convert_fp16(onnx_path: str) -> None:
    """Convert float32 ONNX weights to float16 in place (I/O kept float32)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(onnx_path)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, onnx_path)


def _session(onnx_path: str, device: str = "cpu"):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def validate_onnx(model: nn.Module, onnx_path: str, input_size: int = 224,
                  n: int = 8, topk: int = 10, device: str = "cpu",
                  seed: int = 0) -> dict:
    """Compare ONNX vs PyTorch embeddings.

    Returns ``{"max_abs_diff", "mean_abs_diff", "cosine_min", "topk_overlap",
    "topk"}``. ``topk_overlap`` is the mean fraction of shared indices in the
    top-k neighbour list of each row (1.0 = identical ranking).
    """
    dev = torch.device(device)
    model = model.eval().to(dev)
    x = default_input(input_size, batch=n, seed=seed)

    with torch.no_grad():
        ref = model(x.to(dev)).detach().cpu().float().numpy()

    sess = _session(onnx_path, device=device)
    out_name = sess.get_outputs()[0].name
    in_name = sess.get_inputs()[0].name
    got = sess.run([out_name], {in_name: x.numpy().astype(np.float32)})[0]

    diff = np.abs(ref - got)
    ref_n = ref / np.clip(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12, None)
    got_n = got / np.clip(np.linalg.norm(got, axis=1, keepdims=True), 1e-12, None)
    cosine = (ref_n * got_n).sum(axis=1)

    overlap = _topk_overlap(ref, got, topk)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_min": float(cosine.min()),
        "topk_overlap": float(overlap),
        "topk": int(topk),
        "n": int(n),
    }


def _topk_overlap(ref: np.ndarray, got: np.ndarray, topk: int) -> float:
    """Mean Jaccard-style overlap of top-k cosine neighbours per row."""
    def norm(a):
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

    r = norm(ref.astype(np.float64))
    g = norm(got.astype(np.float64))
    n = r.shape[0]
    if n < 2:
        return 1.0
    sim_r = r @ r.T
    sim_g = g @ g.T
    np.fill_diagonal(sim_r, -np.inf)
    np.fill_diagonal(sim_g, -np.inf)
    k = min(topk, n - 1)
    order_r = np.argsort(-sim_r, axis=1)[:, :k]
    order_g = np.argsort(-sim_g, axis=1)[:, :k]
    fracs = [len(set(a) & set(b)) / k for a, b in zip(order_r, order_g)]
    return float(np.mean(fracs))


def _load_torch_model(path: str, device: str) -> nn.Module:
    """Best-effort load of a TorchScript / pickled ``nn.Module`` checkpoint."""
    try:
        return torch.jit.load(path, map_location=device)
    except Exception:
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, nn.Module):
            return obj
        if isinstance(obj, dict) and "state_dict" in obj:
            model = build_embedder()
            model.load_state_dict(obj["state_dict"])
            return model
        raise ValueError(f"cannot interpret checkpoint {path!r} (type {type(obj)!r})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a ReID model to ONNX.")
    ap.add_argument("--model", required=True, help="checkpoint (.pt/.pth) or 'dummy:resnet18'")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true", help="convert weights to fp16")
    ap.add_argument("--no-dynamic-batch", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if args.model.startswith("dummy"):
        arch = args.model.split(":", 1)[1] if ":" in args.model else "resnet18"
        model = build_embedder(arch=arch, pretrained=False)
    else:
        model = _load_torch_model(args.model, args.device)

    out = export_onnx(
        model, args.out, input_size=args.input_size, opset=args.opset,
        fp16=args.fp16, dynamic_batch=not args.no_dynamic_batch, device="cpu",
    )
    rep = validate_onnx(model, out, input_size=args.input_size, device=args.device)
    print(f"exported: {out}")
    print(f"validation: max_abs_diff={rep['max_abs_diff']:.3e} "
          f"cosine_min={rep['cosine_min']:.6f} "
          f"top{rep['topk']}_overlap={rep['topk_overlap']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
