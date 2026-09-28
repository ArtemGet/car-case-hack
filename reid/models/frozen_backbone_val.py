"""Frozen, public-image-backbone val feature extraction (GPU, torch).

The champion is a fusion of two frozen extractors (SigLIP2 NaFlex vehicle-ReID +
DINOv2-B). The main remaining lever is a *stronger* frozen backbone. This script
extracts val embeddings (query then gallery, seed-42 hold-out) for a timm or HF
vision backbone, optionally with query-side multi-resolution TTA (mean of
L2-normalised features across scales, no hflip), using the same aspect-preserving
square-pad crop as the rest of the pipeline.

Forward runs on the GPU ONLY: asserts ``torch.cuda.is_available()`` and that the
model is on CUDA. CPU is used only for decode/crop.

    python -m reid.models.frozen_backbone_val --model dinov2l \
        --sizes 336,518 --out runs/exp-0066-dinov2l-val
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from reid.data.crop import open_cropped
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# model key -> (source, identifier, url_repo, kind)
REGISTRY = {
    "dinov2l": ("timm", "vit_large_patch14_reg4_dinov2.lvd142m",
                "https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m"),
    "dinov2l_noreg": ("timm", "vit_large_patch14_dinov2.lvd142m",
                      "https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m"),
    "eva02l": ("timm", "eva02_large_patch14_clip_336.merged2b",
               "https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b"),
    "siglip2so": ("hf_siglip2", "google/siglip2-so400m-patch14-384",
                  "https://huggingface.co/google/siglip2-so400m-patch14-384"),
}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class TimmBackend:
    """timm model -> global-pooled features (num_classes=0)."""

    def __init__(self, name, size, device):
        import timm
        from timm.data import create_transform, resolve_data_config
        self.kind = "timm"
        self.name = name
        self.model = timm.create_model(name, pretrained=True, num_classes=0,
                                       img_size=size)
        self.model.eval().to(device)
        self.device = device
        cfg = resolve_data_config({}, model=self.model)
        cfg["input_size"] = (3, size, size)
        self.transform = create_transform(**cfg)
        self.model_path = _timm_weight_path(name)

    @torch.no_grad()
    def embed_batch(self, imgs):
        x = torch.stack([self.transform(im) for im in imgs])
        x = x.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feat = self.model(x)
        feat = feat.float()
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().astype(np.float32)


class SigLIP2HFBackend:
    """HuggingFace SigLIP2 vision tower -> get_image_features (projected)."""

    def __init__(self, name, size, device):
        from transformers import AutoModel, AutoProcessor
        self.kind = "hf_siglip2"
        self.name = name
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(device)
        self.device = device
        self.model_path = _hf_weight_path(name)

    @torch.no_grad()
    def embed_batch(self, imgs):
        inputs = self.proc(images=imgs, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model.get_image_features(pixel_values=pv)
        if not torch.is_tensor(out):
            for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
                if hasattr(out, attr):
                    out = getattr(out, attr)
                    break
        if out.ndim == 3:          # (B, T, D) -> mean-pool patch tokens
            out = out.mean(dim=1)
        out = torch.nn.functional.normalize(out.float(), dim=1)
        return out.cpu().numpy().astype(np.float32)


def _timm_weight_path(name):
    try:
        from huggingface_hub import hf_hub_download
        repo = f"timm/{name}"
        for fn in ("model.safetensors", "pytorch_model.bin"):
            try:
                return hf_hub_download(repo, fn, local_files_only=True if False else False)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return None


def _hf_weight_path(repo):
    try:
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(repo, "model.safetensors")
        except Exception:  # noqa: BLE001
            return hf_hub_download(repo, "pytorch_model.bin")
    except Exception:  # noqa: BLE001
        return None


def build_backend(kind, name, size, device):
    if kind == "timm":
        return TimmBackend(name, size, device)
    if kind == "hf_siglip2":
        return SigLIP2HFBackend(name, size, device)
    raise ValueError(kind)


def extract(backend, df, dataset_dir, size, bs=16):
    out = []
    buf = []
    t0 = time.time()
    n = len(df)
    for i, r in enumerate(df.itertuples(index=False)):
        img = open_cropped(dataset_dir, r.image_id, r.x, r.y, r.w, r.h,
                           target=size, draft_factor=2.0)
        buf.append(img)
        if len(buf) == bs or i == n - 1:
            out.append(backend.embed_batch(buf))
            buf = []
            if (i + 1) % 320 == 0 or i == n - 1:
                print(f"    {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out, axis=0).astype(np.float32)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--sizes", default="", help="comma scales, e.g. 336,518")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, only use this many query/gallery rows (wiring test)")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available; refusing CPU forward", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__}",
          flush=True)

    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr)
            return 2
        args.dataset = os.path.dirname(cand[0])

    src, ident, url = REGISTRY[args.model]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        sizes = [518 if "dinov2" in ident else 336]

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    if args.smoke:
        vq = vq.head(args.smoke).reset_index(drop=True)
        vg = vg.head(args.smoke).reset_index(drop=True)
    print(f"[val] query={len(vq)} gallery={len(vg)} sizes={sizes}", flush=True)

    per_size_q, per_size_g = {}, {}
    weight_sha = None
    weight_file = ""
    for size in sizes:
        backend = build_backend(src, ident, size, device)
        if backend.model_path:
            weight_file = os.path.basename(backend.model_path)
            if weight_sha is None:
                weight_sha = sha256_file(backend.model_path)
        q = extract(backend, vq, args.dataset, size, bs=args.batch)
        g = extract(backend, vg, args.dataset, size, bs=args.batch)
        per_size_q[size], per_size_g[size] = q, g
        print(f"[size={size}] q={q.shape} g={g.shape}", flush=True)
        del backend
        torch.cuda.empty_cache()

    q = np.mean([per_size_q[s] for s in sizes], axis=0).astype(np.float32)
    g = np.mean([per_size_g[s] for s in sizes], axis=0).astype(np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    g = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "embeddings.npy"),
            np.vstack([q, g]).astype(np.float32))
    for s in sizes:
        np.save(os.path.join(out, f"embeddings_{s}.npy"),
                np.vstack([per_size_q[s], per_size_g[s]]).astype(np.float32))
    vq.to_csv(os.path.join(out, "query.csv"), index=False)
    vg.to_csv(os.path.join(out, "gallery.csv"), index=False)
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(os.path.join(out, "gt.csv"),
                                                index=False)
    meta = {"model_key": args.model, "source": src, "identifier": ident,
            "url": url, "sizes": sizes, "dim": int(q.shape[1]),
            "n_query": len(vq), "n_gallery": len(vg),
            "weight_file": weight_file,
            "weight_sha256": weight_sha,
            "device": torch.cuda.get_device_name(0)}
    with open(os.path.join(out, "external_sources.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] embeddings -> {out}\\embeddings.npy  sha256={weight_sha}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
