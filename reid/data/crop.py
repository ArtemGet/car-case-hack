"""Aspect-preserving vehicle crop (fit-to-square + pad).

Contract (docs/_workspace/INTERFACES.md, W0-1):
  * crop the BBox,
  * resize so the LONGER side == ``target`` (uniform scale on both axes),
  * pad the shorter side up to a ``target x target`` square,
  * pad background = mean RGB of the cropped content.

No direct ``Resize((224, 224))``: the aspect ratio of the content is never
changed. Accepts a PIL image or a numpy array and returns a PIL RGB image.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .io import image_path


def _as_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def _mean_rgb(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    if arr.size == 0:
        return (0, 0, 0)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(round(v)) for v in mean)


def crop_vehicle(image, x, y, w, h, target=224, fill="mean"):
    """Return an aspect-preserving ``target x target`` crop of the BBox.

    Parameters
    ----------
    image : PIL.Image | numpy.ndarray
        Source frame.
    x, y, w, h : int
        BBox in source-frame pixels (top-left / width / height). Clamped to
        the image bounds; a degenerate BBox falls back to the full frame.
    target : int
        Output square side in pixels.
    fill : "mean" | tuple[int, int, int]
        Padding colour. ``"mean"`` uses the mean RGB of the cropped content.

    Returns
    -------
    PIL.Image.Image
        RGB image of size ``(target, target)``.
    """
    if target <= 0:
        raise ValueError("target must be positive")

    img = _as_pil(image)
    img_w, img_h = img.size

    xi, yi = int(round(x)), int(round(y))
    wi, hi = int(round(w)), int(round(h))

    x0 = min(max(xi, 0), img_w - 1)
    y0 = min(max(yi, 0), img_h - 1)
    x1 = min(max(xi + wi, x0 + 1), img_w)
    y1 = min(max(yi + hi, y0 + 1), img_h)

    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if cw <= 0 or ch <= 0:  # pragma: no cover - guarded by clamps above
        crop = img
        cw, ch = img.size

    fill_color = _mean_rgb(crop) if fill == "mean" else tuple(int(v) for v in fill)

    scale = float(target) / float(max(cw, ch))
    nw = max(1, min(target, int(round(cw * scale))))
    nh = max(1, min(target, int(round(ch * scale))))

    if (nw, nh) != (cw, ch):
        resized = crop.resize((nw, nh), Image.BILINEAR)
    else:
        resized = crop.copy()

    canvas = Image.new("RGB", (target, target), fill_color)
    canvas.paste(resized, ((target - nw) // 2, (target - nh) // 2))
    return canvas


def open_cropped(dataset_dir, image_id, x, y, w, h, target=224,
                 draft_factor=2.0, fill="mean"):
    """Open ``<image_id>.jpg`` with a partial JPEG decode, then crop.

    The source frames are large (~700x500 .. 1740x1050). Full-resolution
    decode dominates data-loading time and starves the GPU, so we ask libjpeg
    to decode only enough resolution for the BBox to keep at least
    ``target * draft_factor`` pixels on its short side (``Image.draft``), and
    scale the BBox accordingly. PNG and already-small images are unaffected.
    """
    path = image_path(dataset_dir, image_id)
    with Image.open(path) as im:
        ow, oh = im.size
        if draft_factor and draft_factor > 0 and ow > 1 and oh > 1:
            short = max(1, min(int(w), int(h)))
            desired = max(int(target), int(round(int(target) * float(draft_factor))))
            step = max(1, short // desired)
            if step > 1:
                dw = max(1, ow // step)
                dh = max(1, oh // step)
                im.draft("RGB", (dw, dh))
        img = im.convert("RGB")
        nw_, nh_ = img.size
        sx = nw_ / float(ow)
        sy = nh_ / float(oh)
        return crop_vehicle(img, x * sx, y * sy, w * sx, h * sy,
                            target=target, fill=fill)
