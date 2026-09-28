#!/usr/bin/env python
"""Performance bench that replays the organisers' measurement protocol.

Protocol (docs/_workspace/00_BRIEF.md §4.3, METRICS.md §«Производительность»):

* The measured unit is the **full ``extract()`` cycle**: read file from disk ->
  JPEG decode -> bbox crop (``reid/data/crop.py``, aspect-preserving) ->
  preprocess -> forward -> postprocess -> L2-normalise. Gallery search and
  re-ranking are **excluded**.
* ``latency_b1`` — median over ``--latency-runs`` (default 300) single-image
  extracts after ``--warmup`` (default 50) warmups, with a CUDA synchronise
  before/after every timed iteration.
* ``throughput`` — best sustained FPS over batch sizes 1/8/16/32, each run for
  at least ``--throughput-seconds`` (default 10 s). The best value scores.
* ``peak_vram_mb`` — max device memory in use across the whole run (sampled).
* ``weights_mb`` — total size of the model weight files (< 2 GiB gate).

Variants (``--variant``):
  * ``square``  — legacy single-model path (``--model`` .onnx/.pt or ``--dummy``);
  * ``dino``    — in-house DINOv2-B ONNX fp16 (``--dino-model``), draft-decode;
  * ``siglip2`` — external SigLIP2 NaFlex ONNX fp16 (``--siglip-model``);
  * ``fusion``  — DINOv2 + SigLIP2 with a **shared** decode, concatenated
                  ``L2([L2(sig), w * L2(dino)])`` (``--fusion-w``).

CLI::

    python tools/bench_perf.py --model artifacts/model.onnx --images /in/images \\
        --query /in/test_query.csv --gallery /in/test_gallery.csv \\
        --json reports/bench.json --device cuda

    python tools/bench_perf.py --variant fusion --dino-model artifacts/dinov2_b_fp16.onnx \\
        --siglip-model artifacts/siglip2_fp16.onnx --fusion-w 0.6 --images <images> \\
        --query <q.csv> --gallery <g.csv> --json reports/bench_fusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from reid.data.crop import crop_vehicle  # noqa: E402

MB = 1024 * 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHT_EXTS = {
    ".pt", ".pth", ".bin", ".onnx", ".engine", ".plan",
    ".safetensors", ".ckpt", ".trt", ".pb", ".tflite", ".npz",
}
SIGLIP_PATCH = 16
SIGLIP_MAX_PATCHES = 256


# --------------------------------------------------------------------------- #
# image item list
# --------------------------------------------------------------------------- #
def build_items(images_dir: str, csv_paths) -> list:
    """Collect ``(abs_path, (x, y, w, h))`` rows from the CSV files, in order."""
    import pandas as pd

    items = []
    missing = 0
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            p = os.path.join(images_dir, f"{row.image_id}.jpg")
            if not os.path.isfile(p):
                missing += 1
                continue
            items.append((p, (int(row.x), int(row.y), int(row.w), int(row.h))))
    if missing:
        print(f"[bench] WARNING: {missing} image(s) referenced by CSV not found")
    return items


# --------------------------------------------------------------------------- #
# ONNX Runtime helpers
# --------------------------------------------------------------------------- #
_ORT_CUDA = False


def enable_ort_cuda() -> None:
    """Make torch's bundled CUDA/cuDNN DLLs discoverable by ORT (Windows).

    ORT's CUDAExecutionProvider needs cublas/cublasLt/cuDNN on the DLL search
    path. PyTorch ships them under ``torch/lib``; without this the provider
    silently fails to load and everything falls back to CPU.
    """
    global _ORT_CUDA
    if _ORT_CUDA:
        return
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass
    _ORT_CUDA = True


def ort_session(path: str, device: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        enable_ort_cuda()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def siglip_grid(w: int, h: int, max_patches: int = SIGLIP_MAX_PATCHES):
    """NaFlex patch grid (rows, cols) for a natural-aspect crop."""
    aspect = w / max(1, h)
    rows = max(1, int(round((max_patches / aspect) ** 0.5)))
    cols = max(1, int(round(max_patches / rows)))
    if rows * cols > max_patches:
        cols = max_patches // rows
    return int(rows), int(cols)


def _open_drafted(path: str, bbox, desired_short: int, draft_factor: float):
    """Open a JPEG with a partial decode (``Image.draft``), return (img, sx, sy).

    ``desired_short`` is the wanted number of pixels on the BBox's short side in
    the *decoded* image; ``draft_factor`` (>1) keeps a safety margin. DCT scaling
    can only downsample, so when the BBox is already small the full image is
    decoded (better-quality path, no upscaling).
    """
    from PIL import Image

    x, y, w, h = bbox
    im = Image.open(path)
    ow, oh = im.size
    if draft_factor and draft_factor > 0 and ow > 1 and oh > 1:
        short = max(1, min(int(w), int(h)))
        desired = max(1, int(round(desired_short * float(draft_factor))))
        step = max(1, short // desired)
        if step > 1:
            im.draft("RGB", (max(1, ow // step), max(1, oh // step)))
    im = im.convert("RGB")
    nw, nh = im.size
    return im, nw / float(ow), nh / float(oh)


def _raw_crop(img, bbox, sx: float, sy: float, target_long: int, target_short: int):
    """Natural-aspect crop of the (already scaled) BBox, resized to the grid."""
    from PIL import Image

    x, y, w, h = bbox
    iw, ih = img.size
    x0 = min(max(int(round(x * sx)), 0), iw - 1)
    y0 = min(max(int(round(y * sy)), 0), ih - 1)
    x1 = min(max(int(round((x + w) * sx)), x0 + 1), iw)
    y1 = min(max(int(round((y + h) * sy)), y0 + 1), ih)
    crop = img.crop((x0, y0, x1, y1))
    return crop.resize((target_short, target_long), Image.BILINEAR)


def _patchify(img, rows: int, cols: int) -> np.ndarray:
    """PIL crop (already rows*16 x cols*16) -> (rows*cols, 16*16*3) patches."""
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # mean=std=0.5
    p = SIGLIP_PATCH
    patches = arr.reshape(rows, p, cols, p, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(patches.reshape(rows * cols, p * p * 3))


def _normalize_square(crop):
    from torchvision.transforms import functional as TF

    t = TF.to_tensor(crop)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.numpy()


# --------------------------------------------------------------------------- #
# legacy backends (single square model, torch / onnx)
# --------------------------------------------------------------------------- #
class TorchBackend:
    kind = "torch"

    def __init__(self, model, device: str, size: int = 224):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.size = size

    def preprocess(self, crop):
        from torchvision.transforms import functional as TF

        t = TF.to_tensor(crop)
        return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, batch_tensor):
        with self.torch.no_grad():
            out = self.model(batch_tensor.to(self.device))
        return out.detach().float().cpu()

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


class OnnxBackend:
    kind = "onnx"

    def __init__(self, path: str, device: str, size: int = 224):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.size = size
        self.sess = ort_session(path, device)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def preprocess(self, crop):
        from torchvision.transforms import functional as TF

        t = TF.to_tensor(crop)
        return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, batch_tensor):
        arr = batch_tensor.numpy().astype(np.float32)
        out = self.sess.run([self.output_name], {self.input_name: arr})[0]
        import torch

        return torch.from_numpy(np.asarray(out)).float()

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


# --------------------------------------------------------------------------- #
# model loading (legacy)
# --------------------------------------------------------------------------- #
def _param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def load_backend(args):
    """Return ``(backend, model_desc, weights_mb, weight_files)``."""
    device = args.device
    if args.dummy:
        from reid.export.export_onnx import build_embedder

        model = build_embedder(arch="resnet18", pretrained=False)
        weights_mb = _param_bytes(model) / MB
        return (TorchBackend(model, device, args.input_size),
                "dummy:resnet18", weights_mb, [])

    path = args.model
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".onnx":
        size_mb = os.path.getsize(path) / MB
        return (OnnxBackend(path, device, args.input_size), path, size_mb,
                [path])

    if ext in (".pt", ".pth", ".ckpt"):
        import torch
        try:
            model = torch.jit.load(path, map_location=device)
        except Exception:
            from reid.export.export_onnx import _load_torch_model

            model = _load_torch_model(path, device)
        size_mb = os.path.getsize(path) / MB
        return (TorchBackend(model, device, args.input_size), path, size_mb,
                [path])

    raise ValueError(f"unsupported model extension {ext!r}; use .onnx/.pt/.pth")


# --------------------------------------------------------------------------- #
# extraction atom (legacy)
# --------------------------------------------------------------------------- #
def _decode_crop(path, bbox, size):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return crop_vehicle(im, bbox[0], bbox[1], bbox[2], bbox[3], target=size)


def l2_normalize(x):
    """Row-wise L2 normalisation of a torch tensor ``(N, D)``."""
    import torch

    return x / torch.clamp(x.norm(p=2, dim=1, keepdim=True), min=1e-12)


def extract_batch(backend, items, size):
    """Full extract() for a list of items -> L2-normalised ``(N, D)`` tensor."""
    import torch

    tensors = [backend.preprocess(_decode_crop(p, b, size)) for p, b in items]
    batch = torch.stack(tensors, dim=0)
    out = backend.forward(batch)
    return l2_normalize(out)


# --------------------------------------------------------------------------- #
# multi-backbone deployment backends
# --------------------------------------------------------------------------- #
class ExtractBackend:
    """Full-cycle backend: ``extract(items)`` -> L2-normalised (N, D) numpy."""

    kind = "base"
    weight_files: list = []

    def prepare(self, items):  # CPU-heavy: read/decode/crop/preprocess
        raise NotImplementedError

    def forward(self, prep):  # GPU: run the model(s)
        raise NotImplementedError

    def sync(self):
        pass

    def extract(self, items):
        return self.forward(self.prepare(items))


class DinoOnnxBackend(ExtractBackend):
    """In-house DINOv2-B @224, aspect-preserving square crop + draft decode."""

    kind = "dino"

    def __init__(self, path: str, device: str, size: int = 224,
                 draft_factor: float = 1.2):
        self.device = device
        self.size = int(size)
        self.draft_factor = float(draft_factor)
        self.sess = ort_session(path, device)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.weight_files = [path]

    def prepare(self, items):
        out = np.empty((len(items), 3, self.size, self.size), np.float32)
        for i, (path, bbox) in enumerate(items):
            img, sx, sy = _open_drafted(path, bbox, self.size, self.draft_factor)
            crop = crop_vehicle(img, bbox[0] * sx, bbox[1] * sy,
                                bbox[2] * sx, bbox[3] * sy, target=self.size)
            out[i] = _normalize_square(crop)
        return out

    def forward(self, prep):
        raw = np.asarray(self.sess.run([self.output_name],
                                       {self.input_name: prep})[0], dtype=np.float32)
        return l2(raw)

    def sync(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()


class SigLip2OnnxBackend(ExtractBackend):
    """External SigLIP2 NaFlex: natural-aspect crop, patch-16 tokenisation."""

    kind = "siglip2"

    def __init__(self, path: str, device: str, draft_factor: float = 1.2,
                 max_patches: int = SIGLIP_MAX_PATCHES, chunk: int = 16):
        self.device = device
        self.draft_factor = float(draft_factor)
        self.max_patches = int(max_patches)
        self.chunk = int(chunk)
        self.sess = ort_session(path, device)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_name = self.sess.get_outputs()[0].name
        self.weight_files = [path]

    def _item(self, path, bbox):
        x, y, w, h = bbox
        rows, cols = siglip_grid(w, h, self.max_patches)
        short_grid = min(rows, cols) * SIGLIP_PATCH
        img, sx, sy = _open_drafted(path, bbox, short_grid, self.draft_factor)
        crop = _raw_crop(img, bbox, sx, sy, rows * SIGLIP_PATCH,
                         cols * SIGLIP_PATCH)
        return _patchify(crop, rows, cols), rows, cols

    def prepare(self, items):
        b = len(items)
        pv = np.zeros((b, self.max_patches, SIGLIP_PATCH * SIGLIP_PATCH * 3),
                      np.float32)
        mask = np.zeros((b, self.max_patches), np.int64)
        shapes = np.zeros((b, 2), np.int64)
        for i, (path, bbox) in enumerate(items):
            patches, rows, cols = self._item(path, bbox)
            pv[i, : patches.shape[0]] = patches
            mask[i, : patches.shape[0]] = 1
            shapes[i] = (rows, cols)
        return pv, mask, shapes

    def forward(self, prep):
        pv, mask, shapes = prep
        n = pv.shape[0]
        outs = []
        for s in range(0, n, self.chunk):
            e = min(n, s + self.chunk)
            feed = {self.in_names[0]: pv[s:e], self.in_names[1]: mask[s:e],
                    self.in_names[2]: shapes[s:e]}
            outs.append(np.asarray(self.sess.run([self.out_name], feed)[0],
                                   dtype=np.float32))
        return l2(np.concatenate(outs, axis=0))

    def sync(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()


class FusionBackend(ExtractBackend):
    """SigLIP2 + w * DINOv2, concat L2, **one shared decode per item**.

    Supports optional query-side TTA: ``dino_backends`` maps a DINOv2 input size
    (224/280) to its static ONNX graph. One shared JPEG decode feeds every crop;
    the per-scale DINOv2 embeddings are L2-normalised, averaged and re-normalised
    (``rerank.fuse_embeddings`` semantics), then concatenated with SigLIP2.
    """

    kind = "fusion"

    def __init__(self, dino_backends, sig: SigLip2OnnxBackend,
                 w: float = 0.6):
        # dino_backends: {size: DinoOnnxBackend} (insertion order = base first)
        self.dino_backends = dict(dino_backends)
        self.sizes = list(self.dino_backends.keys())
        if not self.sizes:
            raise ValueError("FusionBackend needs >=1 DINOv2 backend")
        self.base_size = self.sizes[0]
        self.dino = self.dino_backends[self.base_size]
        self.sig = sig
        self.w = float(w)
        self.device = self.dino.device
        files = sorted({p for b in self.dino_backends.values()
                        for p in b.weight_files} | set(sig.weight_files))
        self.weight_files = files

    def prepare(self, items):
        n = len(items)
        arrs = {s: np.empty((n, 3, s, s), np.float32) for s in self.sizes}
        pv = np.zeros((n, SIGLIP_MAX_PATCHES,
                       SIGLIP_PATCH * SIGLIP_PATCH * 3), np.float32)
        mask = np.zeros((n, SIGLIP_MAX_PATCHES), np.int64)
        shapes = np.zeros((n, 2), np.int64)
        for i, (path, bbox) in enumerate(items):
            x, y, w, h = bbox
            rows, cols = siglip_grid(w, h, SIGLIP_MAX_PATCHES)
            short_grid = min(rows, cols) * SIGLIP_PATCH
            # one draft decode good enough for EVERY crop (both DINO scales + sig)
            desired_short = max(max(self.sizes), short_grid)
            img, sx, sy = _open_drafted(path, bbox, desired_short,
                                        self.dino.draft_factor)
            for s in self.sizes:
                sq = crop_vehicle(img, x * sx, y * sy, w * sx, h * sy,
                                  target=s)
                arrs[s][i] = _normalize_square(sq)
            raw = _raw_crop(img, bbox, sx, sy, rows * SIGLIP_PATCH,
                            cols * SIGLIP_PATCH)
            patches = _patchify(raw, rows, cols)
            pv[i, : patches.shape[0]] = patches
            mask[i, : patches.shape[0]] = 1
            shapes[i] = (rows, cols)
        return arrs, pv, mask, shapes

    def forward(self, prep):
        arrs, pv, mask, shapes = prep
        views = [l2(self.dino_backends[s].forward(arrs[s]))
                 for s in self.sizes]
        if len(views) > 1:
            d = l2(np.mean(np.stack(views, axis=0), axis=0))
        else:
            d = views[0]
        s = l2(self.sig.forward((pv, mask, shapes)))
        fused = np.concatenate([s, self.w * d], axis=1)
        return l2(fused)

    def sync(self):
        for b in self.dino_backends.values():
            b.sync()
        self.sig.sync()


# --------------------------------------------------------------------------- #
# GPU-preprocess backends (reid.export.gpu_preproc) — CPU decode + GPU
# crop/resize/letterbox/normalise/patchify. Numerically matches the CPU path
# (max pixel delta 1/255, embedding cosine >= 0.99986) at ~0.1 ms/view instead
# of ~7 ms/view.
# --------------------------------------------------------------------------- #
def _to_cuda_roi(roi_np):
    import torch

    return torch.from_numpy(roi_np).permute(2, 0, 1).contiguous().cuda()


class GpuDinoOnnxBackend(ExtractBackend):
    """DINOv2-B with GPU preprocessing (CPU draft-decode + bbox ROI)."""

    kind = "dino"

    def __init__(self, path: str, device: str, size: int = 224,
                 draft_factor: float = 1.0, stage_size: int = 0):
        self.device = device
        self.size = int(size)
        self.draft_factor = float(draft_factor)
        self.stage_size = int(stage_size)
        self.sess = ort_session(path, device)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.out_dim = int(self.sess.get_outputs()[0].shape[-1])
        self.weight_files = [path]

    def prepare(self, items):
        from reid.export.gpu_preproc import decode_roi

        return [decode_roi(p, b, self.size, self.draft_factor)
                for p, b in items]

    def forward(self, rois):
        import torch

        from reid.export.gpu_preproc import (l2_t, letterbox_stage_u8,
                                             normalize_u8, run_onnx_cuda)

        arr = torch.cat(
            [normalize_u8(letterbox_stage_u8(_to_cuda_roi(r), self.size,
                                             self.stage_size)).unsqueeze(0)
             for r in rois], 0).contiguous()
        out = run_onnx_cuda(self.sess, {self.input_name: arr},
                            self.output_name, (arr.shape[0], self.out_dim))
        return l2_t(out).cpu().numpy()

    def sync(self):
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()


class GpuSiglip2OnnxBackend(ExtractBackend):
    """SigLIP2 NaFlex with GPU preprocessing (CPU draft-decode + bbox ROI)."""

    kind = "siglip2"

    def __init__(self, path: str, device: str, draft_factor: float = 1.0,
                 max_patches: int = SIGLIP_MAX_PATCHES, chunk: int = 16):
        self.device = device
        self.draft_factor = float(draft_factor)
        self.max_patches = int(max_patches)
        self.chunk = int(chunk)
        self.sess = ort_session(path, device)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_name = self.sess.get_outputs()[0].name
        self.out_dim = int(self.sess.get_outputs()[0].shape[-1])
        self.weight_files = [path]

    def prepare(self, items):
        from reid.export.gpu_preproc import decode_roi, siglip_grid

        out = []
        for p, b in items:
            _x, _y, w, h = b
            rows, cols = siglip_grid(w, h, self.max_patches)
            short = min(rows, cols) * SIGLIP_PATCH
            out.append((decode_roi(p, b, short, self.draft_factor), rows, cols))
        return out

    def forward(self, prep):
        import torch

        from reid.export.gpu_preproc import l2_t, run_onnx_cuda, siglip_patches

        pv, mask, shapes = [], [], []
        for roi, rows, cols in prep:
            t = _to_cuda_roi(roi)
            p = siglip_patches(t, rows, cols)
            full = torch.zeros(self.max_patches, p.shape[1], device="cuda")
            full[: p.shape[0]] = p
            pv.append(full)
            m = torch.zeros(self.max_patches, dtype=torch.int64, device="cuda")
            m[: p.shape[0]] = 1
            mask.append(m)
            shapes.append(torch.tensor([rows, cols], dtype=torch.int64,
                                       device="cuda"))
        if not pv:
            return np.zeros((0, self.out_dim), np.float32)
        PV, MK, SH = torch.stack(pv), torch.stack(mask), torch.stack(shapes)
        n = PV.shape[0]
        outs = []
        for s in range(0, n, self.chunk):
            e = min(n, s + self.chunk)
            outs.append(run_onnx_cuda(
                self.sess,
                {self.in_names[0]: PV[s:e], self.in_names[1]: MK[s:e],
                 self.in_names[2]: SH[s:e]},
                self.out_name, (e - s, self.out_dim)))
        return l2_t(torch.cat(outs, axis=0)).cpu().numpy()

    def sync(self):
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()


class GpuFusionBackend(ExtractBackend):
    """Fusion with GPU preprocessing and one shared draft-decode per item."""

    kind = "fusion"

    def __init__(self, dino_backends, sig: GpuSiglip2OnnxBackend,
                 w: float = 0.6, stage_size: int = 0):
        self.dino_backends = dict(dino_backends)
        self.sizes = list(self.dino_backends.keys())
        if not self.sizes:
            raise ValueError("GpuFusionBackend needs >=1 DINOv2 backend")
        self.base_size = self.sizes[0]
        self.sig = sig
        self.w = float(w)
        self.max_patches = sig.max_patches
        self.stage_size = int(stage_size)
        self.draft_factor = self.dino_backends[self.base_size].draft_factor
        self.device = self.dino_backends[self.base_size].device
        files = sorted({p for b in self.dino_backends.values()
                        for p in b.weight_files} | set(sig.weight_files))
        self.weight_files = files

    def prepare(self, items):
        from reid.export.gpu_preproc import decode_roi, siglip_grid

        out = []
        for p, b in items:
            _x, _y, w, h = b
            rows, cols = siglip_grid(w, h, self.max_patches)
            short = min(rows, cols) * SIGLIP_PATCH
            desired = max(max(self.sizes), short)
            out.append((decode_roi(p, b, desired, self.draft_factor), rows, cols))
        return out

    def forward(self, prep):
        import torch

        from reid.export.gpu_preproc import (l2_t, letterbox_stage_u8,
                                             normalize_u8, run_onnx_cuda,
                                             siglip_patches)

        views = {s: [] for s in self.sizes}
        pv, mask, shapes = [], [], []
        for roi, rows, cols in prep:
            t = _to_cuda_roi(roi)
            for s in self.sizes:
                views[s].append(normalize_u8(
                    letterbox_stage_u8(t, s, self.stage_size)).unsqueeze(0))
            p = siglip_patches(t, rows, cols)
            full = torch.zeros(self.max_patches, p.shape[1], device="cuda")
            full[: p.shape[0]] = p
            pv.append(full)
            m = torch.zeros(self.max_patches, dtype=torch.int64, device="cuda")
            m[: p.shape[0]] = 1
            mask.append(m)
            shapes.append(torch.tensor([rows, cols], dtype=torch.int64,
                                       device="cuda"))

        n = len(prep)
        dv = []
        for s in self.sizes:
            b = self.dino_backends[s]
            arr = torch.cat(views[s], 0).contiguous()
            out = run_onnx_cuda(b.sess, {b.input_name: arr}, b.output_name,
                                (n, b.out_dim))
            dv.append(l2_t(out))
        d = l2_t(torch.stack(dv, 0).mean(0)) if len(dv) > 1 else dv[0]

        PV, MK, SH = torch.stack(pv), torch.stack(mask), torch.stack(shapes)
        ins = self.sig.in_names
        outs = []
        for a in range(0, n, self.sig.chunk):
            e = min(n, a + self.sig.chunk)
            outs.append(run_onnx_cuda(
                self.sig.sess,
                {ins[0]: PV[a:e], ins[1]: MK[a:e], ins[2]: SH[a:e]},
                self.sig.out_name, (e - a, self.sig.out_dim)))
        sv = l2_t(torch.cat(outs, axis=0))
        return l2_t(torch.cat([sv, self.w * d], axis=1)).cpu().numpy()

    def sync(self):
        for b in self.dino_backends.values():
            b.sync()
        self.sig.sync()


def _dino_scales(args):
    """DINOv2 scales for the run: ``--dino-tta '224,280'`` else ``--input-size``."""
    raw = getattr(args, "dino_tta", None)
    if raw:
        scales = [int(s) for s in str(raw).replace(" ", "").split(",") if s]
    else:
        scales = [int(args.input_size)]
    seen, out = set(), []
    for s in scales:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _dino_path_for_scale(args, size: int) -> str:
    """Resolve the static ONNX graph for one DINOv2 scale."""
    if size == 224:
        p = args.dino_model or args.model
    elif size == 280:
        p = (getattr(args, "dino_model_280", None)
             or os.path.join(_ROOT, "artifacts", "dinov2_b_280_fp16.onnx"))
    else:
        raise SystemExit(f"unsupported DINOv2 TTA scale {size} (only 224/280)")
    if not p or not os.path.isfile(p):
        raise SystemExit(f"DINOv2 ONNX for scale {size} not found: {p}")
    return p


def _build_dino_backends(args, device, scales):
    return {s: DinoOnnxBackend(_dino_path_for_scale(args, s), device, size=s,
                               draft_factor=args.draft_factor)
            for s in scales}


def build_variant_backend(args):
    """Build the requested ``--variant`` backend; returns (backend, desc, files)."""
    device = args.device
    gpu = getattr(args, "preproc", "cpu") == "gpu"
    stage = int(getattr(args, "stage_size", 0) or 0)
    if args.variant == "dino":
        scales = _dino_scales(args)
        if len(scales) == 1:
            if gpu:
                b = GpuDinoOnnxBackend(_dino_path_for_scale(args, scales[0]),
                                       device, size=scales[0],
                                       draft_factor=args.draft_factor,
                                       stage_size=stage)
            else:
                b = _build_dino_backends(args, device, scales)[scales[0]]
            return b, f"dino:{scales[0]}", b.weight_files
        raise SystemExit("--variant dino TTA not wired; use --variant fusion")
    if args.variant == "siglip2":
        path = args.siglip_model or args.model
        if not path:
            raise SystemExit("--variant siglip2 requires --siglip-model")
        if gpu:
            b = GpuSiglip2OnnxBackend(path, device,
                                      draft_factor=args.draft_factor)
        else:
            b = SigLip2OnnxBackend(path, device, draft_factor=args.draft_factor)
        return b, f"siglip2:{path}", b.weight_files
    if args.variant == "fusion":
        if not (args.dino_model and args.siglip_model):
            raise SystemExit("--variant fusion requires --dino-model and --siglip-model")
        scales = _dino_scales(args)
        if gpu:
            dino_backends = {
                s: GpuDinoOnnxBackend(_dino_path_for_scale(args, s), device,
                                      size=s, draft_factor=args.draft_factor,
                                      stage_size=stage)
                for s in scales}
            sig = GpuSiglip2OnnxBackend(args.siglip_model, device,
                                        draft_factor=args.draft_factor)
            b = GpuFusionBackend(dino_backends, sig, w=args.fusion_w,
                                 stage_size=stage)
        else:
            dino_backends = _build_dino_backends(args, device, scales)
            sig = SigLip2OnnxBackend(args.siglip_model, device,
                                     draft_factor=args.draft_factor)
            b = FusionBackend(dino_backends, sig, w=args.fusion_w)
        return (b, f"fusion(w={args.fusion_w},tta={list(scales)},"
                   f"preproc={'gpu' if gpu else 'cpu'})",
                b.weight_files)
    raise ValueError(args.variant)


# --------------------------------------------------------------------------- #
# measurements (legacy square)
# --------------------------------------------------------------------------- #
def measure_latency(backend, items, size, warmup, runs):
    import torch

    backend.sync()
    for i in range(warmup):
        extract_batch(backend, [items[i % len(items)]], size)
    backend.sync()

    times = []
    for i in range(runs):
        item = items[i % len(items)]
        backend.sync()
        t0 = time.perf_counter()
        extract_batch(backend, [item], size)
        backend.sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return {
        "median_ms": float(statistics.median(times)),
        "mean_ms": float(statistics.fmean(times)),
        "p90_ms": float(times[min(len(times) - 1, int(0.9 * len(times)))]),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "min_ms": float(times[0]),
        "n_warmup": int(warmup),
        "n_measured": int(runs),
    }


def measure_throughput_one(backend, items, size, batch_size, seconds):
    """Run full-cycle batches until ``seconds`` elapsed; return FPS + counts."""
    n = len(items)
    idx = 0
    done = 0
    backend.sync()
    t0 = time.perf_counter()
    while True:
        batch_items = [items[(idx + k) % n] for k in range(batch_size)]
        idx = (idx + batch_size) % n
        extract_batch(backend, batch_items, size)
        done += batch_size
        if time.perf_counter() - t0 >= seconds:
            break
    backend.sync()
    elapsed = time.perf_counter() - t0
    return {
        "batch_size": int(batch_size),
        "seconds": float(elapsed),
        "images": int(done),
        "fps": float(done / elapsed) if elapsed > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# measurements (multi-backbone) with CPU prefetch and VRAM sampling
# --------------------------------------------------------------------------- #
class VramMonitor(threading.Thread):
    """Sample device memory in use (via torch) and keep the maximum."""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.max_mb = 0.0
        self.base_mb = 0.0
        self._evt = threading.Event()

    def sample(self):
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            return (total - free) / MB
        except Exception:  # noqa: BLE001
            return None

    def run(self):
        while not self._evt.is_set():
            v = self.sample()
            if v is not None:
                self.max_mb = max(self.max_mb, v)
            self._evt.wait(self.interval)

    def stop(self):
        self._evt.set()
        self.join(timeout=2.0)


def measure_latency_extract(backend, items, warmup, runs):
    backend.sync()
    for i in range(warmup):
        backend.extract([items[i % len(items)]])
    backend.sync()
    times = []
    for i in range(runs):
        item = items[i % len(items)]
        backend.sync()
        t0 = time.perf_counter()
        backend.extract([item])
        backend.sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": float(statistics.median(times)),
        "mean_ms": float(statistics.fmean(times)),
        "p90_ms": float(times[min(len(times) - 1, int(0.9 * len(times)))]),
        "std_ms": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "min_ms": float(times[0]),
        "n_warmup": int(warmup),
        "n_measured": int(runs),
    }


def measure_throughput_extract(backend, items, batch_size, seconds,
                               workers: int = 4):
    """Sustained FPS with CPU decode/preprocess pipelined across worker threads.

    The full cycle is CPU-heavy (JPEG decode + crop) while the backbone is small,
    so a single prefetch thread caps throughput at the CPU rate. libjpeg and
    numpy release the GIL, so several workers prepare future batches in parallel
    while the main thread runs the (single) forward — exactly what the batch
    runner does. This is a full-cycle measurement: every image is read from disk,
    decoded, cropped, preprocessed, forwarded, postprocessed and L2-normalised.
    """
    from collections import deque

    n = len(items)
    state = {"idx": 0}

    def next_items():
        i = state["idx"]
        b = [items[(i + k) % n] for k in range(batch_size)]
        state["idx"] = (i + batch_size) % n
        return b

    workers = max(1, int(workers))
    pool = ThreadPoolExecutor(max_workers=workers)
    inflight = deque(pool.submit(backend.prepare, next_items())
                     for _ in range(workers))
    backend.sync()
    t0 = time.perf_counter()
    done = 0
    while True:
        prep = inflight.popleft().result()
        inflight.append(pool.submit(backend.prepare, next_items()))
        backend.forward(prep)
        done += batch_size
        if time.perf_counter() - t0 >= seconds:
            break
    backend.sync()
    elapsed = time.perf_counter() - t0
    pool.shutdown(wait=True)
    return {
        "batch_size": int(batch_size),
        "seconds": float(elapsed),
        "images": int(done),
        "fps": float(done / elapsed) if elapsed > 0 else 0.0,
        "prefetch_workers": workers,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_batches(text):
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def _write_report(args, device, model_desc, backend_kind, items, lat, tp_runs,
                  best, peak_vram_mb, weights_mb, weight_files, extra=None):
    import torch

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "organisers-4.3",
        "variant": getattr(args, "variant", "square"),
        "model": model_desc,
        "backend": backend_kind,
        "device": device,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_size": args.input_size,
        "num_images": len(items),
        "image_source": {"images": args.images, "query": args.query,
                         "gallery": args.gallery},
        "latency_b1_ms": lat["median_ms"],
        "latency_b1": lat,
        "throughput_fps_best": best,
        "throughput": {
            "runs": tp_runs,
            "seconds_per_run": args.throughput_seconds,
            "batches": parse_batches(args.batches),
        },
        "weights_mb": weights_mb,
        "weight_files": [{"path": p, "size_mb": os.path.getsize(p) / MB}
                         for p in weight_files],
        "peak_vram_mb": peak_vram_mb,
        "weights_limit_mb": 2 * 1024,
        "pass_weights_gate": weights_mb < 2 * 1024,
    }
    if extra:
        report.update(extra)
    return report


def _finish(args, report):
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[bench] report -> {args.json}")
    print("=" * 60)
    print(f"variant              : {report.get('variant')}")
    print(f"latency_b1_ms        : {report['latency_b1_ms']:.2f}")
    print(f"throughput_fps_best  : {report['throughput_fps_best']:.1f}")
    print(f"weights_mb           : {report['weights_mb']:.1f} "
          f"(gate<2048: {report['pass_weights_gate']})")
    print(f"peak_vram_mb         : {report['peak_vram_mb']:.1f}")
    print("=" * 60)


def run_multi_backbone(args, items, device):
    backend, model_desc, weight_files = build_variant_backend(args)
    weights_mb = sum(os.path.getsize(p) for p in weight_files) / MB
    print(f"[bench] variant={args.variant} model={model_desc} device={device} "
          f"images={len(items)} draft={args.draft_factor}")

    mon = VramMonitor()
    base = mon.sample()
    mon.base_mb = base or 0.0
    mon.start()

    lat = measure_latency_extract(backend, items, args.warmup, args.latency_runs)
    print(f"[bench] latency_b1 median={lat['median_ms']:.2f} ms "
          f"(p90={lat['p90_ms']:.2f})")

    tp_runs = []
    for bs in parse_batches(args.batches):
        r = measure_throughput_extract(backend, items, bs, args.throughput_seconds,
                                       workers=args.prefetch_workers)
        tp_runs.append(r)
        print(f"[bench] throughput batch={bs:>2}  {r['fps']:8.1f} FPS "
              f"({r['images']} imgs / {r['seconds']:.2f}s)")
    best = max((r["fps"] for r in tp_runs), default=0.0)

    mon.stop()
    peak_vram_mb = mon.max_mb
    extra = {
        "draft_factor": args.draft_factor,
        "preproc": getattr(args, "preproc", "cpu"),
        "fusion_w": args.fusion_w if args.variant == "fusion" else None,
        "baseline_vram_mb": base,
        "peak_vram_mb_nvidia": peak_vram_mb,
    }
    report = _write_report(args, device, model_desc, backend.kind, items, lat,
                           tp_runs, best, peak_vram_mb, weights_mb, weight_files,
                           extra)
    _finish(args, report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ReID performance bench (organisers' protocol).")
    ap.add_argument("--variant", default="auto",
                    choices=["auto", "square", "dino", "siglip2", "fusion"])
    ap.add_argument("--model", default=None,
                    help="path to .onnx / .pt / .pth (ignored with --dummy)")
    ap.add_argument("--dino-model", default=None, help="DINOv2-B ONNX for dino/fusion")
    ap.add_argument("--dino-model-280", default=None,
                    help="DINOv2-B ONNX static @280 graph for TTA "
                         "(default artifacts/dinov2_b_280_fp16.onnx)")
    ap.add_argument("--dino-tta", default=None,
                    help="query-side DINOv2 TTA scales, e.g. '224,280' "
                         "(default: --input-size only)")
    ap.add_argument("--siglip-model", default=None, help="SigLIP2 ONNX for siglip2/fusion")
    ap.add_argument("--fusion-w", type=float, default=0.6,
                    help="weight on the DINOv2 half of the fusion concat")
    ap.add_argument("--draft-factor", type=float, default=1.0,
                    help="JPEG partial-decode margin (Image.draft); libjpeg DCT "
                         "scaling is integer (1/2,1/4,...), so ~1.0 already keeps "
                         "the BBox at >= the network input size and triggers the "
                         "2x decode reduction more often than 1.2")
    ap.add_argument("--preproc", default="cpu", choices=["cpu", "gpu"],
                    help="preprocess stage: 'cpu' (PIL crop_vehicle, default) or "
                         "'gpu' (CPU draft-decode + bbox ROI, then GPU "
                         "resize/letterbox/normalise/patchify; numerically "
                         "equivalent, ~7 ms/view -> ~0.1 ms/view)")
    ap.add_argument("--stage-size", type=int, default=0,
                    help="GPU only: letterbox to this size first, then resize to "
                         "the network input (reproduces the train_320 crop-cache "
                         "2-stage pipeline entirely on the GPU); 0 = off")
    ap.add_argument("--prefetch-workers", type=int, default=4,
                    help="CPU threads decoding/preprocessing future batches")
    ap.add_argument("--dummy", action="store_true",
                    help="use a torchvision resnet18 embedder instead of --model")
    ap.add_argument("--images", required=True, help="flat images directory")
    ap.add_argument("--query", default=None, help="test_query.csv")
    ap.add_argument("--gallery", default=None, help="test_gallery.csv")
    ap.add_argument("--json", default=None, help="output report path")
    ap.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--latency-runs", type=int, default=300)
    ap.add_argument("--throughput-seconds", type=float, default=10.0)
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap the item pool (0 = all); for smoke runs")
    args = ap.parse_args(argv)

    if args.variant == "auto":
        args.variant = "square"

    if args.variant == "square" and not args.dummy and not args.model:
        ap.error("--model is required unless --dummy is given")
    csv_paths = [p for p in (args.query, args.gallery) if p]
    if not csv_paths:
        ap.error("at least one of --query / --gallery is required")

    device = args.device
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("[bench] WARNING: CUDA requested but unavailable -> cpu")
                device = "cpu"
                args.device = "cpu"
        except Exception:
            device = "cpu"
            args.device = "cpu"

    items = build_items(args.images, csv_paths)
    if args.max_images and len(items) > args.max_images:
        items = items[: args.max_images]
    if not items:
        print("[bench] ERROR: no images resolved from --images + CSVs",
              file=sys.stderr)
        return 2

    import torch
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    if args.variant != "square":
        run_multi_backbone(args, items, device)
        return 0

    print(f"[bench] model={'dummy:resnet18' if args.dummy else args.model} "
          f"device={device} images={len(items)} input={args.input_size}")
    backend, model_desc, weights_mb, weight_files = load_backend(args)

    ran_batches = parse_batches(args.batches)
    lat = measure_latency(backend, items, args.input_size,
                          args.warmup, args.latency_runs)
    print(f"[bench] latency_b1 median={lat['median_ms']:.2f} ms "
          f"(p90={lat['p90_ms']:.2f})")

    tp_runs = []
    for bs in ran_batches:
        r = measure_throughput_one(backend, items, args.input_size, bs,
                                   args.throughput_seconds)
        tp_runs.append(r)
        print(f"[bench] throughput batch={bs:>2}  {r['fps']:8.1f} FPS "
              f"({r['images']} imgs / {r['seconds']:.2f}s)")

    best = max((r["fps"] for r in tp_runs), default=0.0)

    peak_vram_mb = 0.0
    if device.startswith("cuda"):
        peak_vram_mb = torch.cuda.max_memory_allocated() / MB

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "organisers-4.3",
        "variant": "square",
        "model": model_desc,
        "backend": backend.kind,
        "device": device,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_size": args.input_size,
        "num_images": len(items),
        "image_source": {"images": args.images, "query": args.query,
                         "gallery": args.gallery},
        "latency_b1_ms": lat["median_ms"],
        "latency_b1": lat,
        "throughput_fps_best": best,
        "throughput": {
            "runs": tp_runs,
            "seconds_per_run": args.throughput_seconds,
            "batches": ran_batches,
        },
        "weights_mb": weights_mb,
        "weight_files": [{"path": p, "size_mb": os.path.getsize(p) / MB}
                         for p in weight_files],
        "peak_vram_mb": peak_vram_mb,
        "weights_limit_mb": 2 * 1024,
        "pass_weights_gate": weights_mb < 2 * 1024,
    }

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[bench] report -> {args.json}")

    print("=" * 60)
    print(f"latency_b1_ms        : {report['latency_b1_ms']:.2f}")
    print(f"throughput_fps_best  : {report['throughput_fps_best']:.1f}")
    print(f"weights_mb           : {report['weights_mb']:.1f} "
          f"(gate<2048: {report['pass_weights_gate']})")
    print(f"peak_vram_mb         : {report['peak_vram_mb']:.1f}")
    print("=" * 60)
    return 0


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
