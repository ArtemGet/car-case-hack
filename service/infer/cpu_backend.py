"""service.infer.cpu_backend — CPU-only SigLIP2 ONNX extractor (demo prototype).

This module is the **prototype / demo** path: it runs the public SigLIP2 NaFlex
vehicle-ReID ONNX graph on ONNX Runtime's CPU execution provider, with a pure
PIL + NumPy pre-process. It deliberately imports **no torch / torchvision** so a
"2 CPU / 2 GB / no GPU" box can serve the demo without the CUDA stack.

Hard rules (see TASKS W2-5 CPU-prototype):

* The session is built with ``providers=["CPUExecutionProvider"]`` **only** and
  the loaded session is asserted to expose exactly that provider list. There is
  no silent fallback in either direction.
* The production GPU path (``service.infer.backends`` + ``--preproc gpu``) is
  untouched and still requires ``CUDAExecutionProvider`` first; CPU is reachable
  only through an explicit ``--device cpu`` (or ``REID_DEVICE=cpu``).
* Only the SigLIP2 branch is supported on CPU (``--variant siglip``). The DINOv2
  backbone is never loaded here, so the two-backbone fusion can never blow the
  demo's RAM budget.

The NaFlex model consumes natural-aspect patch tokens (no square letterbox), so
the "aspect-preserving PIL crop" here is a bbox crop that keeps the source
aspect ratio — the same geometry as :meth:`SiglipBackend._raw_bbox_crop` — and
never distorts the vehicle. The ONNX graph's I/O is float32 even though the file
is named ``*_fp16``; inputs are cast to the graph-declared dtype (float32 here)
and outputs are returned as float32.
"""
from __future__ import annotations

import os
import random

import numpy as np
from PIL import Image

__all__ = ["PATCH", "MAX_PATCHES", "rss_mb", "set_cpu_determinism",
           "SiglipCpuBackend"]

PATCH = 16
MAX_PATCHES = 256


# ---------------------------------------------------------------------------
# Determinism / memory (torch-free)
# ---------------------------------------------------------------------------
def set_cpu_determinism(seed: int = 42) -> None:
    """Seed the RNGs we control on the CPU path (no torch import)."""
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))


def rss_mb() -> float:
    """Current resident set size of this process in MiB (``nan`` if unknown)."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass
    try:  # POSIX fallback
        import resource  # type: ignore

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:  # noqa: BLE001
        return float("nan")


def _raw_bbox_crop(images_dir, image_id, x, y, w, h):
    """Natural-aspect bbox crop (PIL), same geometry as the GPU SigLIP path."""
    path = os.path.join(images_dir, f"{image_id}.jpg")
    with Image.open(path) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        x0 = min(max(int(round(x)), 0), ow - 1)
        y0 = min(max(int(round(y)), 0), oh - 1)
        x1 = min(max(int(round(x + w)), x0 + 1), ow)
        y1 = min(max(int(round(y + h)), y0 + 1), oh)
        return im.crop((x0, y0, x1, y1))


class SiglipCpuBackend:
    """SigLIP2 NaFlex vehicle-ReID extractor on ONNX Runtime CPU EP only."""

    dim = 512
    name = "siglip"
    default_size = 256

    def __init__(self, weights: str, max_patches: int = MAX_PATCHES,
                 threads: int = 8, max_ram_mb: int = 0):
        import onnxruntime as ort

        weights = os.path.abspath(weights)
        if not os.path.exists(weights):
            raise SystemExit(f"не найден SigLIP2 ONNX: {weights}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        # deterministic single-stream execution
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Keep the demo's RSS flat (~0.4 GB observed vs ~2 GB with the arena on):
        # the default CPU memory arena / mem-pattern never release reused buffers.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False

        providers = ["CPUExecutionProvider"]  # exactly one provider
        print(f"[siglip-cpu] loading {weights} providers={providers} "
              f"rss={rss_mb():.0f}MB", flush=True)
        self.sess = ort.InferenceSession(weights, sess_options=so,
                                         providers=providers)
        got = list(self.sess.get_providers())
        if got != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"SigLIP2 CPU session providers={got}; expected exactly "
                "['CPUExecutionProvider'] (refusing GPU/other EP)")
        self.weights = weights
        self.max_patches = int(max_patches)
        self.patch = PATCH
        self.max_ram_mb = int(max_ram_mb or 0)
        self.inputs = self.sess.get_inputs()
        self.in_names = [i.name for i in self.inputs]
        self.in_dtypes = [i.type for i in self.inputs]
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[siglip-cpu] inputs={list(zip(self.in_names, self.in_dtypes))} "
              f"out={self.out_name} rss={rss_mb():.0f}MB", flush=True)
        self._check_ram("after load")

    # -- memory -------------------------------------------------------------
    def _check_ram(self, where: str) -> float:
        cur = rss_mb()
        if not np.isnan(cur):
            print(f"[siglip-cpu] rss {where}: {cur:.0f}MB", flush=True)
        if self.max_ram_mb and not np.isnan(cur) and cur > self.max_ram_mb:
            raise RuntimeError(
                f"[siglip-cpu] RSS {cur:.0f}MB > --max-ram-mb "
                f"{self.max_ram_mb}MB ({where}); уменьшите batch/threads "
                "или поднимите лимит")
        return cur

    # -- pre-process (PIL + NumPy only) ------------------------------------
    def _patchify(self, img, rows, cols):
        img = img.resize((cols * PATCH, rows * PATCH), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        patches = arr.reshape(rows, PATCH, cols, PATCH, 3).transpose(0, 2, 1, 3, 4)
        return patches.reshape(rows * cols, PATCH * PATCH * 3)

    def _feed_batch(self, imgs, size):
        b = len(imgs)
        mp = self.max_patches if not size else min(self.max_patches, int(size))
        pv = np.zeros((b, self.max_patches, PATCH * PATCH * 3), np.float32)
        mask = np.zeros((b, self.max_patches), np.int64)
        shapes = np.zeros((b, 2), np.int64)
        for i, img in enumerate(imgs):
            w, h = img.size
            aspect = w / max(1, h)
            rows = max(1, int(round((mp / aspect) ** 0.5)))
            cols = max(1, int(round(mp / rows)))
            if rows * cols > mp:
                cols = mp // rows
            rows, cols = int(rows), int(cols)
            patches = self._patchify(img, rows, cols)
            pv[i, : patches.shape[0]] = patches
            mask[i, : patches.shape[0]] = 1
            shapes[i] = (rows, cols)
        # cast inputs to the graph-declared dtype (float32 for siglip2_fp16.onnx)
        feed = {}
        for name, dtype, arr in zip(self.in_names, self.in_dtypes,
                                    (pv, mask, shapes)):
            feed[name] = arr.astype(np.float16) if "float16" in dtype else arr
        out = self.sess.run([self.out_name], feed)[0]
        out = np.asarray(out, dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-12, None)

    def extract(self, df, images_dir, size: int = None, batch_size: int = 32,
                cache_dir=None) -> np.ndarray:
        """(N, 512) float32 L2-normalised embeddings in ``df`` row order."""
        size = self.default_size if size is None else int(size)
        rows = list(df.itertuples(index=False))
        n = len(rows)
        chunks, imgs = [], []
        done = 0
        for i, r in enumerate(rows):
            imgs.append(_raw_bbox_crop(images_dir, r.image_id, r.x, r.y, r.w, r.h))
            if len(imgs) == batch_size or i == n - 1:
                chunks.append(self._feed_batch(imgs, size))
                imgs = []
                done = i + 1
                self._check_ram(f"batch {done}/{n}")
                if done % (batch_size * 8) < batch_size:
                    print(f"      siglip-cpu: {done}/{n}", flush=True)
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.concatenate(chunks, axis=0).astype(np.float32)
        assert out.shape[0] == n, (out.shape, n)
        return out
