#!/usr/bin/env python
"""GPU preprocessing for the ReID extract path (perf-engineer).

Motivation (measured on RTX 4090, 2026-09-27, see ``reports/opt_gpu_preproc.json``):
the CPU crop/preprocess stage dominated the full ``extract()`` cycle at batch=1 —
``reid.data.crop.crop_vehicle`` costs ~6.9 ms per view (of which ~4.4 ms is the
``_mean_rgb`` float64 mean), so the champion fusion TTA path (224 + 280 + SigLIP)
spent ~24 ms on CPU preprocessing and only ~11 ms on the GPU forwards. The same
crop/resize/letterbox/normalise done on CUDA costs ~0.1 ms per view.

This module provides the GPU equivalents of the *exact* CPU math:

    decode (PIL ``Image.draft`` + ``convert("RGB")``, libjpeg numerics preserved)
        -> bbox crop (numpy, exact same integer clamping as crop.py)
        -> H2D uint8 ROI
        -> aspect-preserving bilinear resize + mean-colour letterbox (torchvision
           ``v2.functional.resize`` on uint8, which is PIL-bilinear compatible)
        -> ImageNet normalise (fp32) / SigLIP patchify (mean=std=0.5).

Numerical parity with the CPU pipeline (8 real val crops, 224/280): max pixel
delta 1/255, **zero** pixels differing by >2, DINOv2 embedding cosine >= 0.99986.
The candidate ranking is therefore unchanged; the full val mAP check is in
``reports/opt_*_val.json``.

Nothing here touches the network; all weights are local.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PATCH",
    "MAX_PATCHES",
    "siglip_grid",
    "decode_roi",
    "letterbox_u8",
    "normalize_u8",
    "siglip_patches",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16
MAX_PATCHES = 256

_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
_BILINEAR = InterpolationMode.BILINEAR


# --------------------------------------------------------------------------- #
# CPU: decode + bbox ROI (numerically identical to reid.data.crop.crop_vehicle)
# --------------------------------------------------------------------------- #
def decode_roi(path: str, bbox, desired_short: int,
               draft_factor: float = 2.0) -> np.ndarray:
    """Draft-decode ``path`` and return the bbox ROI as uint8 HWC.

    The decoded (possibly DCT-downscaled) frame is cropped to the bbox with the
    same integer clamping as :func:`reid.data.crop.crop_vehicle`; the returned
    ROI is the input to the GPU letterbox / patchify paths.
    """
    from PIL import Image

    x, y, w, h = (float(v) for v in bbox)
    with Image.open(path) as im:
        ow, oh = im.size
        if draft_factor and draft_factor > 0 and ow > 1 and oh > 1:
            short = max(1, min(int(w), int(h)))
            desired = max(1, int(round(int(desired_short) * float(draft_factor))))
            step = max(1, short // desired)
            if step > 1:
                im.draft("RGB", (max(1, ow // step), max(1, oh // step)))
        img = im.convert("RGB")
        nw, nh = img.size
        sx, sy = nw / float(ow), nh / float(oh)
        arr = np.asarray(img)

    xi, yi = int(round(x * sx)), int(round(y * sy))
    wi, hi = int(round(w * sx)), int(round(h * sy))
    x0 = min(max(xi, 0), nw - 1)
    y0 = min(max(yi, 0), nh - 1)
    x1 = min(max(xi + wi, x0 + 1), nw)
    y1 = min(max(yi + hi, y0 + 1), nh)
    return np.ascontiguousarray(arr[y0:y1, x0:x1])


# --------------------------------------------------------------------------- #
# GPU: letterbox / normalise / patchify
# --------------------------------------------------------------------------- #
def _mean_fill_u8(roi_u8: torch.Tensor) -> torch.Tensor:
    """Exact ``int(round(mean_rgb))`` of a uint8 [3,H,W] CUDA tensor."""
    n = roi_u8.shape[1] * roi_u8.shape[2]
    mean = roi_u8.reshape(3, -1).to(torch.int64).sum(1).to(torch.float64) / n
    return mean.round().to(torch.uint8)


def letterbox_u8(roi_u8: torch.Tensor, target: int,
                 fill: str = "mean") -> torch.Tensor:
    """Aspect-preserving bilinear resize + mean-colour letterbox.

    Replicates :func:`reid.data.crop.crop_vehicle` exactly (longer side ==
    ``target``, shorter side padded to a ``target x target`` square) but on the
    GPU. ``roi_u8`` is uint8 ``[3, ch, cw]`` on CUDA; returns uint8
    ``[3, target, target]`` on the same device.
    """
    cw, ch = roi_u8.shape[2], roi_u8.shape[1]
    scale = float(target) / float(max(cw, ch))
    nw = max(1, min(target, int(round(cw * scale))))
    nh = max(1, min(target, int(round(ch * scale))))
    if (nw, nh) != (cw, ch):
        resized = tvF.resize(roi_u8, [nh, nw], interpolation=_BILINEAR,
                             antialias=True)
    else:
        resized = roi_u8
    if fill == "mean":
        fc = _mean_fill_u8(roi_u8)
    else:
        fc = torch.tensor(fill, dtype=torch.uint8, device=roi_u8.device)
    canvas = fc.view(3, 1, 1).expand(3, target, target).clone()
    oy, ox = (target - nh) // 2, (target - nw) // 2
    canvas[:, oy:oy + nh, ox:ox + nw] = resized
    return canvas


def normalize_u8(square_u8: torch.Tensor) -> torch.Tensor:
    """ImageNet normalise a uint8 ``[3,S,S]`` CUDA tensor to fp32 ``[3,S,S]``."""
    x = square_u8.to(torch.float32).div_(255.0)
    return (x - _MEAN_T.to(x.device)) / _STD_T.to(x.device)


def letterbox_stage_u8(roi_u8: torch.Tensor, target: int,
                       stage_size: int = 0) -> torch.Tensor:
    """Letterbox to ``stage_size`` then resize to ``target`` (crop-cache path).

    Reproduces the train_320 crop-cache preprocessing (aspect-preserving square
    crop at 320, then the eval transform's ``Resize(target)``) entirely on the
    GPU, without the 320 JPEG round-trip. ``stage_size=0`` is the direct
    letterbox to ``target``.
    """
    if stage_size and int(stage_size) > 0:
        sq = letterbox_u8(roi_u8, int(stage_size))
        return tvF.resize(sq, [int(target), int(target)],
                          interpolation=_BILINEAR, antialias=True)
    return letterbox_u8(roi_u8, int(target))


def siglip_grid(w, h, max_patches: int = MAX_PATCHES):
    """NaFlex patch grid (rows, cols) for a natural-aspect crop (same as CPU)."""
    aspect = w / max(1, h)
    rows = max(1, int(round((max_patches / aspect) ** 0.5)))
    cols = max(1, int(round(max_patches / rows)))
    if rows * cols > max_patches:
        cols = max_patches // rows
    return int(rows), int(cols)


def siglip_patches(roi_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Natural-aspect ROI -> ``(rows*cols, 16*16*3)`` fp32 CUDA patches.

    Resizes to ``(cols*16, rows*16)`` then maps to the NaFlex patch layout with
    the ``mean=std=0.5`` normalisation used by the SigLIP2 preprocessor.
    """
    t = tvF.resize(roi_u8, [rows * PATCH, cols * PATCH],
                   interpolation=_BILINEAR, antialias=True)
    a = t.to(torch.float32).div_(127.5).sub_(1.0)
    p = PATCH
    a = a.reshape(3, rows, p, cols, p).permute(1, 3, 2, 4, 0)  # rows,cols,p,p,3
    return a.reshape(rows * cols, p * p * 3).contiguous()


# --------------------------------------------------------------------------- #
# Zero-copy ONNX CUDA I/O binding (avoid the D2H -> H2D round-trip per forward)
# --------------------------------------------------------------------------- #
_NP_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
}


def l2_t(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation of a CUDA tensor ``(N, D)``."""
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)


def run_onnx_cuda(sess, feeds: dict, out_name: str, out_shape,
                  device_id: int = 0) -> torch.Tensor:
    """Run an ORT session with inputs/outputs bound to CUDA torch tensors.

    ``feeds`` maps input name -> contiguous CUDA torch tensor. The output is
    allocated on the GPU and returned as a CUDA tensor — no host round-trip.
    """
    io = sess.io_binding()
    for name, t in feeds.items():
        t = t.contiguous()
        io.bind_input(name, device_type="cuda", device_id=device_id,
                      element_type=_NP_DTYPE[t.dtype], shape=tuple(t.shape),
                      buffer_ptr=t.data_ptr())
    out = torch.empty(tuple(out_shape), dtype=torch.float32, device="cuda")
    io.bind_output(out_name, device_type="cuda", device_id=device_id,
                   element_type=np.float32, shape=tuple(out.shape),
                   buffer_ptr=out.data_ptr())
    sess.run_with_iobinding(io)
    return out

