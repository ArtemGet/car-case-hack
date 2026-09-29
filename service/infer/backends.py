"""service.infer.backends — feature extractors for the offline batch runner.

Two frozen, offline backends are wrapped here:

* :class:`DinoBackend` — our in-house DINOv2-B + GeM + BNNeck champion
  (``runs/exp-0007/best.pt``). Aspect-preserving square crop, optional
  query-side TTA over /14 scales (224 + 280, no hflip).
* :class:`SiglipBackend` — the external public SigLIP2 NaFlex vehicle-ReID
  extractor (``runs/external/vehicle_reid_siglip2_naflex_512d.onnx``,
  sha256 fixed in ``runs/<exp>/external_sources.json``). Natural-aspect patch
  tokenisation, 256-patch budget, ONNX Runtime **CUDA** EP (CUDA is mandatory;
  the CPU EP is kept only as an explicit per-node shape-op fallback and the
  session is verified to have CUDAExecutionProvider first).

Nothing in this module touches the network. All weights are local files.

Determinism (INTERFACES.md / W2-5): seeds fixed, cudnn deterministic, ONNX
threads pinned; ``shuffle=False`` DataLoaders with ``num_workers=0``.
"""
from __future__ import annotations

import glob
import os
import site

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from reid.data.aug import build_eval_transform
from reid.data.crop import crop_vehicle
from reid.models import build_model

__all__ = [
    "set_determinism",
    "resolve_device",
    "image_file",
    "DinoBackend",
    "DinoOnnxBackend",
    "SiglipBackend",
    "GpuExtractor",
    "PATCH",
    "MAX_PATCHES",
]

PATCH = 16
MAX_PATCHES = 256


# ---------------------------------------------------------------------------
# Determinism / device
# ---------------------------------------------------------------------------
def set_determinism(seed: int = 42) -> None:
    """Pin every source of run-to-run randomness we control."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - older torch
        pass


def resolve_device(requested: str = "auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _add_ort_dll_dirs():
    """torch/lib + nvidia/*/lib on the DLL path so the ORT CUDA EP can load.

    Without this on Windows onnxruntime silently falls back to the CPU EP.
    """
    cands = []
    try:
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            cands.append(lib)
    except Exception:  # noqa: BLE001
        pass
    try:
        for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
            cands += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    except Exception:  # noqa: BLE001
        pass
    for d in cands:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:  # noqa: BLE001
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _cuda_ort_providers(require_cuda: bool = True):
    """Return ``[CUDA, CPU]`` providers; raise if CUDA is unavailable.

    Optional ``REID_ORT_*`` env vars cap the CUDA EP arena (see
    :mod:`reid.export.ort_env`); with none set the returned list is exactly the
    previous default.
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    print(f"[ort] available providers: {avail}", flush=True)
    if require_cuda:
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                "CUDAExecutionProvider unavailable; forward must run on CUDA "
                f"(providers={avail}). Refusing CPU fallback.")
        _add_ort_dll_dirs()
        from reid.export.ort_env import apply_cuda_options
        return apply_cuda_options(
            ["CUDAExecutionProvider", "CPUExecutionProvider"])
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Crop I/O (explicit images directory — the CLI takes the folder itself)
# ---------------------------------------------------------------------------
def image_file(images_dir: str, image_id: str) -> str:
    return os.path.join(images_dir, f"{image_id}.jpg")


def _open_cropped(images_dir, image_id, x, y, w, h, target, draft_factor=2.0):
    """Same math as :func:`reid.data.crop.open_cropped`, explicit image path."""
    path = image_file(images_dir, image_id)
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
                            target=target, fill="mean")


def _raw_bbox_crop(images_dir, image_id, x, y, w, h):
    """Natural-aspect bbox crop, no resize (SigLIP2 NaFlex consumes patches)."""
    with Image.open(image_file(images_dir, image_id)) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        x0 = min(max(int(round(x)), 0), ow - 1)
        y0 = min(max(int(round(y)), 0), oh - 1)
        x1 = min(max(int(round(x + w)), x0 + 1), ow)
        y1 = min(max(int(round(y + h)), y0 + 1), oh)
        return im.crop((x0, y0, x1, y1))


class _SquareCropDataset(Dataset):
    """Aspect-preserving square crop -> deterministic eval transform."""

    def __init__(self, df, images_dir, transform, target, cache_dir=None,
                 draft_factor=2.0):
        self.rows = list(df.itertuples(index=False))
        self.images_dir = images_dir
        self.transform = transform
        self.target = int(target)
        self.cache_dir = cache_dir
        self.draft_factor = float(draft_factor)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        cached = (os.path.join(self.cache_dir, r.image_id + ".jpg")
                  if self.cache_dir else None)
        if cached and os.path.exists(cached):
            with Image.open(cached) as im:
                crop = im.convert("RGB")
        else:
            crop = _open_cropped(self.images_dir, r.image_id, r.x, r.y, r.w, r.h,
                                 target=self.target, draft_factor=self.draft_factor)
        return self.transform(crop), i


# ---------------------------------------------------------------------------
# DINOv2-B champion
# ---------------------------------------------------------------------------
class DinoBackend:
    """In-house DINOv2-B + GeM + BNNeck + ArcFace (champion exp-0007)."""

    dim = 512
    name = "dino"

    def __init__(self, checkpoint: str, device, amp: str = "bf16",
                 batch_size: int = 64, num_workers: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError("DinoBackend requires CUDA but torch.cuda is "
                               "unavailable; refusing CPU forward")
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RuntimeError(
                f"DinoBackend requires CUDA (project rule), got device={dev}; "
                "refusing CPU forward")
        print(f"[dino] device={dev} gpu={torch.cuda.get_device_name(0)}",
              flush=True)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = dict(ck.get("config", {}))
        sd = ck["state_dict"]
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.amp = amp
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        model = build_model(
            backbone=ck.get("backbone", cfg.get("backbone", "dinov2_b")),
            num_classes=int(sd["arcface.weight"].shape[0]),
            emb_dim=int(ck.get("emb_dim", cfg.get("emb_dim", 512))),
            pretrained=False,
            margin=float(cfg.get("margin", 0.3)),
            scale=float(cfg.get("scale", 30.0)),
            gem_p=float(cfg.get("gem_p", 3.0)),
            image_size=int(cfg.get("image_size", 224)),
        )
        model.load_state_dict(sd)
        model.to(device).eval()
        self.model = model
        self.device = device
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)

    @torch.no_grad()
    def extract(self, df, images_dir, size: int, cache_dir=None) -> np.ndarray:
        """(N, 512) float32 L2-normalised embeddings in ``df`` row order."""
        transform = build_eval_transform(int(size))
        ds = _SquareCropDataset(df, images_dir, transform, int(size),
                                cache_dir=cache_dir)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers)
        use_amp = self._amp_dtype is not None and self.device.type == "cuda"
        chunks, done = [], 0
        n = len(ds)
        for imgs, _ in loader:
            imgs = imgs.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=self._amp_dtype,
                                enabled=use_amp):
                emb = self.model.embed(imgs)
            chunks.append(emb.float().cpu().numpy())
            done += imgs.shape[0]
            if done % 512 < self.batch_size:
                print(f"      dino@{size}: {done}/{n}", flush=True)
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.concatenate(chunks, axis=0).astype(np.float32)
        assert out.shape[0] == n, (out.shape, n)
        return out


# ---------------------------------------------------------------------------
# In-house DINOv2-B champion, deployed via ONNX Runtime fp16 (CUDA EP)
# ---------------------------------------------------------------------------
class DinoOnnxBackend:
    """DINOv2-B + GeM + BNNeck champion (exp-0007), exported to ONNX fp16.

    This is the **deployed** path (W2-10): the forward runs through ONNX Runtime
    with the CUDA execution provider as the *first* provider. A session whose
    first provider is not ``CUDAExecutionProvider`` raises (no silent CPU
    fallback); ``torch/lib`` + ``nvidia/*/lib`` are put on the DLL path first so
    the CUDA EP can actually load on Windows.

    The graph returns the raw BNNeck output (no L2); this backend L2-normalises
    exactly like the PyTorch :meth:`reid.models.ReIDModel.embed`, so the fusion
    contract (``L2(concat[L2(sig), w * L2(dino)])``) is preserved.

    One instance is bound to one spatial input size (the static ONNX graph
    declares ``[N, 3, S, S]``); TTA uses one instance per scale.
    """

    dim = 512
    name = "dino"

    def __init__(self, weights: str, size: int = 224, threads: int = 8,
                 require_cuda: bool = True, draft_factor: float = 2.0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        providers = _cuda_ort_providers(require_cuda=require_cuda)
        self.sess = ort.InferenceSession(
            weights, sess_options=so, providers=providers)
        if require_cuda:
            got = list(self.sess.get_providers())
            print(f"[dino-onnx] session providers={got}", flush=True)
            if not got or got[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    f"DINOv2 ONNX session is not on CUDA (providers={got}); "
                    "refusing silent CPU forward")
        self.weights = weights
        self.size = int(size)
        self.draft_factor = float(draft_factor)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def extract(self, df, images_dir, size: int = None, batch_size: int = 64,
                cache_dir=None) -> np.ndarray:
        """(N, 512) float32 L2-normalised embeddings in ``df`` row order."""
        size = self.size if size is None else int(size)
        transform = build_eval_transform(size)
        rows = list(df.itertuples(index=False))
        n = len(rows)
        chunks, buf, done = [], [], 0
        for i, r in enumerate(rows):
            cached = (os.path.join(cache_dir, r.image_id + ".jpg")
                      if cache_dir else None)
            if cached and os.path.exists(cached):
                with Image.open(cached) as im:
                    crop = im.convert("RGB")
            else:
                crop = _open_cropped(images_dir, r.image_id, r.x, r.y, r.w, r.h,
                                     target=size, draft_factor=self.draft_factor)
            buf.append(transform(crop).numpy())
            if len(buf) == batch_size or i == n - 1:
                arr = np.ascontiguousarray(np.stack(buf), dtype=np.float32)
                out = np.asarray(
                    self.sess.run([self.out_name], {self.in_name: arr})[0],
                    dtype=np.float32)
                out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True),
                                    1e-12, None)
                chunks.append(out)
                buf = []
                done = i + 1
                if done % 512 < batch_size:
                    print(f"      dino-onnx@{size}: {done}/{n}", flush=True)
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        res = np.concatenate(chunks, axis=0).astype(np.float32)
        assert res.shape[0] == n, (res.shape, n)
        return res


# ---------------------------------------------------------------------------
# External SigLIP2 NaFlex (ONNX, CPU, deterministic)
# ---------------------------------------------------------------------------
class SiglipBackend:
    """Public SigLIP2 NaFlex vehicle-ReID extractor (ONNX Runtime)."""

    dim = 512
    name = "siglip"
    default_size = 256

    def __init__(self, weights: str, max_patches: int = MAX_PATCHES,
                 threads: int = 8, require_cuda: bool = True):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        providers = _cuda_ort_providers(require_cuda=require_cuda)
        self.sess = ort.InferenceSession(
            weights, sess_options=so, providers=providers)
        if require_cuda:
            got = list(self.sess.get_providers())
            print(f"[siglip] session providers={got}", flush=True)
            if not got or got[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    f"SigLIP2 session is not on CUDA (providers={got}); "
                    "refusing silent CPU forward")
        self.weights = weights
        self.max_patches = int(max_patches)
        self.patch = PATCH

    def _patchify(self, img, rows, cols):
        w, h = img.size
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
        out = self.sess.run(None, {"pixel_values": pv,
                                   "pixel_attention_mask": mask,
                                   "spatial_shapes": shapes})[0]
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
                if done % (batch_size * 8) < batch_size:
                    print(f"      siglip: {done}/{n}", flush=True)
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.concatenate(chunks, axis=0).astype(np.float32)
        assert out.shape[0] == n, (out.shape, n)
        return out


# ---------------------------------------------------------------------------
# GPU-preprocess extractor (reid.export.gpu_preproc via tools.bench_perf)
# ---------------------------------------------------------------------------
# The deploy config (variant A) runs the crop/resize/letterbox/normalise stage on
# CUDA (perf-engineer, reid/export/gpu_preproc.py) instead of the CPU PIL path.
# We reuse the perf-engineer's validated GPU backends (the very ones measured in
# exp-0079) rather than duplicating the geometry, then adapt them to the batch
# runner's DataFrame interface. No crop cache is ever consulted here: the
# train_320 cache is train-only (it inflated val by ~+0.014) and is explicitly
# rejected by the caller's guard.
#
# Bench ``--variant`` names differ from the runner's: the runner's ``siglip`` is
# the bench's ``siglip2``.
_GPU_VARIANT = {"fusion": "fusion", "siglip": "siglip2", "dino": "dino"}


def _assert_cuda_sessions(backend) -> list:
    """Return the ORT sessions of a bench GPU backend and assert CUDA is first."""
    sessions = []
    if hasattr(backend, "sess"):
        sessions.append(backend.sess)
    for b in getattr(backend, "dino_backends", {}).values():
        sessions.append(b.sess)
    if hasattr(backend, "sig"):
        sessions.append(backend.sig.sess)
    for s in sessions:
        got = list(s.get_providers())
        if not got or got[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"GPU-preproc ORT session is not on CUDA (providers={got}); "
                "refusing silent CPU forward")
    return sessions


class GpuExtractor:
    """Adapter over the perf-engineer GPU-preprocess backends (CUDA EP).

    ``extract_df`` returns ``(N, D)`` float32 embeddings in ``df`` row order,
    with the same fusion contract as the CPU path
    (``L2(concat[L2(SigLIP2), w * L2(DINOv2)])``). ``scales`` is the DINOv2 grid
    (224-only for the deploy config; 280 is ablation-only). The forward runs on
    ``CUDAExecutionProvider`` exclusively (asserted, no silent CPU fallback).
    """

    def __init__(self, variant, *, dino_onnx, dino_onnx_280=None,
                 siglip_weights=None, scales=(224,), w=0.8,
                 draft_factor=1.0, stage_size=0, device="cuda"):
        from types import SimpleNamespace

        from tools.bench_perf import build_variant_backend

        if variant not in _GPU_VARIANT:
            raise ValueError(f"GPU-preproc не поддержан для variant={variant!r}")
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                f"GPU-preproc требует device=cuda, получено {device!r}")
        scales = tuple(int(s) for s in (scales or (224,)))
        if not scales:
            scales = (224,)
        dino_onnx = os.path.abspath(dino_onnx) if dino_onnx else None
        dino_onnx_280 = os.path.abspath(dino_onnx_280) if dino_onnx_280 else None
        siglip_weights = (os.path.abspath(siglip_weights)
                          if siglip_weights else None)

        _add_ort_dll_dirs()  # torch/lib + nvidia/*/lib on the DLL path
        args = SimpleNamespace(
            variant=_GPU_VARIANT[variant], device="cuda", preproc="gpu",
            dino_model=dino_onnx, dino_model_280=dino_onnx_280,
            siglip_model=siglip_weights, fusion_w=float(w),
            draft_factor=float(draft_factor), input_size=int(scales[0]),
            stage_size=int(stage_size),
            dino_tta=",".join(str(s) for s in scales))
        backend, desc, weight_files = build_variant_backend(args)
        _assert_cuda_sessions(backend)
        self.backend = backend
        self.desc = desc
        self.weight_files = list(weight_files)
        self.scales = scales

    def extract_df(self, df, images_dir, batch_size: int = 64) -> np.ndarray:
        """(N, D) float32 embeddings in ``df`` row order (CUDA forward)."""
        rows = list(df.itertuples(index=False))
        items = [(os.path.join(images_dir, f"{r.image_id}.jpg"),
                  (int(r.x), int(r.y), int(r.w), int(r.h))) for r in rows]
        n = len(items)
        outs = []
        for i in range(0, n, int(batch_size)):
            outs.append(np.asarray(
                self.backend.extract(items[i:i + int(batch_size)]),
                dtype=np.float32))
            done = min(i + int(batch_size), n)
            if (i // int(batch_size)) % 8 == 0:
                print(f"      gpu {self.desc}: {done}/{n}", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        if not outs:
            return np.zeros((0, 0), dtype=np.float32)
        res = np.concatenate(outs, axis=0).astype(np.float32)
        assert res.shape[0] == n, (res.shape, n)
        return res
