"""Training loop for the vehicle ReID extractor (W1-3).

    python -m reid.train --config configs/baseline.yaml --out runs/exp-0004 \
        --dataset "docs/Датасет/dataset"

Produces, under ``--out``:
    best.pt       best-by-val-mAP checkpoint (state_dict + config + metrics)
    report.json   full run report (config, data hash, metrics, artifacts)
    log.jsonl     one JSON line per epoch
    val_latest/   last validation artifacts (submission/candidates/embeddings/gt)

Metric numbers come ONLY from the official ``evaluate.py`` via
:func:`reid.eval.harness.run_official`; this module never recomputes them.

Red lines honoured: ``camera_id`` is never a network input (split / sampling
only); no test data; no hflip (see reid/data/aug.py); per-query ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from reid.data.aug import build_eval_transform, build_train_transform
from reid.data.crop import crop_vehicle, open_cropped
from reid.data.io import image_path, read_csv, TRAIN_COLUMNS
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models import build_model

DEFAULT_DATASET = os.environ.get("DATASET_DIR", "")
MANIFEST = os.path.join("artifacts", "data_manifest.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str | None:
    """Short HEAD sha, or None when not inside a git repo."""
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() or None if p.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class VehicleDataset(Dataset):
    """Vehicle crops from source frames, on-the-fly aspect-preserving crop."""

    def __init__(self, df: pd.DataFrame, dataset_dir: str, transform, target: int,
                 label_col: str = "vehicle_id", cache_dir: str | None = None,
                 draft_factor: float = 2.0):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target = int(target)
        self.label_col = label_col
        self.cache_dir = cache_dir
        self.draft_factor = float(draft_factor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cached = (os.path.join(self.cache_dir, r.image_id + ".jpg")
                  if self.cache_dir else None)
        if cached and os.path.exists(cached):
            crop = Image.open(cached).convert("RGB")
        else:
            # partial JPEG decode keeps the GPU fed on large source frames
            crop = open_cropped(self.dataset_dir, r.image_id, r.x, r.y, r.w, r.h,
                                target=self.target, draft_factor=self.draft_factor)
        tensor = self.transform(crop)
        label = int(r[self.label_col]) if self.label_col in self.df.columns else -1
        return tensor, label, i


class PKSampler:
    """PK batch sampler over ``vehicle_id`` (P identities x K frames).

    ``camera_aware`` biases the K frames of each identity towards DISTINCT
    cameras, so every batch contains cross-camera positives of the same
    vehicle. Cameras are never fed to the network (red line) — only used to
    choose which frames land in a batch.
    """

    def __init__(self, labels, p: int, k: int, seed: int = 0, batches: int = 0,
                 cameras=None, camera_aware: bool = False):
        self.p = int(p)
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        self.cameras = None if cameras is None else np.asarray(cameras)
        self.camera_aware = bool(camera_aware) and self.cameras is not None
        idx_by_id: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            idx_by_id.setdefault(int(lab), []).append(idx)
        self.idx_by_id = {v: np.asarray(ix) for v, ix in idx_by_id.items()}
        self.ids = np.array(sorted(self.idx_by_id))
        if len(self.ids) < self.p:
            raise ValueError(f"need >={self.p} identities, got {len(self.ids)}")
        self.batches = int(batches) if batches > 0 else max(1, len(labels) // (p * k))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.batches

    def _pick(self, pool: np.ndarray, rng) -> list[int]:
        if not self.camera_aware or len(pool) <= self.k:
            rep = len(pool) < self.k
            return [int(x) for x in rng.choice(pool, size=self.k, replace=rep)]

        by_cam: dict[int, list[int]] = {}
        for i in pool:
            by_cam.setdefault(int(self.cameras[i]), []).append(int(i))
        cam_lists = [np.asarray(v) for v in by_cam.values()]
        for lst in cam_lists:
            rng.shuffle(lst)

        sel: list[int] = []
        ptr = [0] * len(cam_lists)
        while len(sel) < self.k:
            progressed = False
            for ci in range(len(cam_lists)):
                if len(sel) >= self.k:
                    break
                if ptr[ci] < len(cam_lists[ci]):
                    sel.append(int(cam_lists[ci][ptr[ci]]))
                    ptr[ci] += 1
                    progressed = True
            if not progressed:
                break
        if len(sel) < self.k:  # fewer distinct cameras than k
            chosen = set(sel)
            rest = [int(i) for i in pool if int(i) not in chosen]
            rng.shuffle(np.asarray(rest))
            for i in rest:
                if len(sel) >= self.k:
                    break
                sel.append(i)
        while len(sel) < self.k:
            sel.append(int(rng.choice(pool)))
        return sel

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches):
            chosen = rng.choice(self.ids, size=self.p, replace=False)
            batch: list[int] = []
            for v in chosen:
                pool = self.idx_by_id[int(v)]
                batch.extend(self._pick(pool, rng))
            yield batch


class EMA:
    """Exponential moving average of model params + buffers.

    Buffers (BatchNorm running stats) are copied verbatim, not averaged, which
    is the standard CLIP/ReID recipe. The averaged copy is evaluated as a
    separate module and is what ``best.pt`` stores.

    ``update`` is a handful of batched ``torch._foreach_*`` kernels (grouped by
    dtype) instead of a per-tensor Python loop, so it stays on the GPU and adds
    no per-step synchronisation. Model parameters only change dtype/device when
    the module is converted, hence the tensor lists are cached in ``__init__``.
    """

    def __init__(self, model, decay: float = 0.999):
        import copy
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._float_groups: dict = {}
        self._int_groups: dict = {}
        self._build_groups()

    def _build_groups(self) -> None:
        self._float_groups.clear()
        self._int_groups.clear()
        for k, v in self.module.state_dict().items():
            groups = self._float_groups if v.dtype.is_floating_point else self._int_groups
            dsts, keys = groups.setdefault(v.dtype, ([], []))
            dsts.append(v)
            keys.append(k)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for dtype, (dsts, keys) in self._float_groups.items():
            srcs = [msd[k].detach() for k in keys]
            try:
                torch._foreach_mul_(dsts, d)
                torch._foreach_add_(dsts, srcs, alpha=1.0 - d)
            except (RuntimeError, TypeError):  # pragma: no cover - CPU/old torch
                for v, s in zip(dsts, srcs):
                    v.mul_(d).add_(s, alpha=1.0 - d)
        for dtype, (dsts, keys) in self._int_groups.items():
            srcs = [msd[k].detach() for k in keys]
            try:
                torch._foreach_copy_(dsts, srcs)
            except (RuntimeError, TypeError):  # pragma: no cover
                for v, s in zip(dsts, srcs):
                    v.copy_(s)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def make_lr_lambda(total_epochs: int, warmup_epochs: int):
    def fn(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        span = max(1, total_epochs - warmup_epochs)
        prog = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return fn


def batch_hard_triplet(emb, labels, margin: float = 0.3) -> torch.Tensor:
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalized embeds.

    For every anchor picks the farthest same-id sample (hardest positive) and the
    closest different-id sample (hardest negative) *inside the batch*. PK
    sampling guarantees >=1 positive per anchor within a batch. Distances are
    squared Euclidean on L2-normalized vectors (== 2 - 2*cosine). Returns the
    mean hinge over anchors with an active positive; 0 when none.
    """
    x = torch.nn.functional.normalize(emb.float(), dim=1)
    d = torch.cdist(x, x, p=2).pow(2)  # squared euclidean
    labels = labels.view(-1)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos = same & ~eye
    neg = ~same
    valid = pos.any(dim=1)
    if not bool(valid.any()):
        return torch.zeros((), device=emb.device)
    d_max = d.masked_fill(~pos, float("-inf")).max(dim=1).values
    d_min = d.masked_fill(~neg, float("inf")).min(dim=1).values
    loss = torch.relu(d_max - d_min + float(margin))
    return loss[valid].mean()


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=0,
                    amp_dtype=None, ema=None, clip_grad=None, channels_last=False,
                    triplet_weight: float = 0.0, triplet_margin: float = 0.3):
    model.train()
    # Keep the running loss as a 0-dim GPU tensor and only read it back at
    # ``log_every`` boundaries: a per-step ``loss.item()`` (plus the two
    # finiteness checks) forces a CUDA sync every step and stalls the pipe.
    loss_sum = None
    n = 0
    use_amp = amp_dtype is not None and device.type == "cuda"
    for step, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, emb = model(imgs, labels)
            loss = criterion(logits, labels)
            if triplet_weight > 0.0:
                loss = loss + triplet_weight * batch_hard_triplet(
                    emb, labels, margin=triplet_margin)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad))
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = imgs.size(0)
        with torch.no_grad():
            batch_loss = loss.detach().float() * bs
        loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
        n += bs

        if log_every and (step + 1) % log_every == 0:
            cur = float(loss_sum) / max(1, n)
            if not math.isfinite(cur):
                raise RuntimeError(
                    f"non-finite loss ({cur}) at step {step} — aborting "
                    "(sanity check: NaN/Inf)")
            if cur > 1e4:
                print(f"    WARNING high loss {cur:.1f} at step {step}", flush=True)
            print(f"    step {step + 1}/{len(loader)} loss={cur:.4f}", flush=True)

    final = float(loss_sum) / max(1, n) if loss_sum is not None else 0.0
    if not math.isfinite(final):
        raise RuntimeError(
            f"non-finite epoch loss ({final}) — aborting (sanity check: NaN/Inf)")
    return final


def loader_kwargs(num_workers: int, prefetch_factor: int = 6) -> dict:
    """DataLoader flags that keep the GPU fed (persistent workers + prefetch).

    On Windows every worker is a fresh ``spawn``ed interpreter that re-imports
    torch, so worker count has a real startup cost (~2 s/worker here). The
    loader saturates at ~8 workers for the 320-px crop cache, so raising it
    past that only inflates cold-start without helping steady state.
    """
    nw = int(num_workers)
    kw = {"num_workers": nw, "pin_memory": True}
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = int(prefetch_factor)
    return kw


class TensorCacheDataset(Dataset):
    """In-memory dataset of already-preprocessed tensors."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return int(self.tensors.shape[0])

    def __getitem__(self, i):
        return self.tensors[i], -1, i


def build_tensor_cache(df, dataset_dir, transform, target, cache_dir=None,
                       draft_factor=2.0, chunk=256, num_workers=0):
    """Decode + transform a split ONCE into a CPU tensor.

    The eval transform is deterministic, so validation can pay the JPEG
    decode/resize cost a single time at startup and thereafter feed the GPU from
    RAM — the extraction becomes GPU-bound instead of data-bound.
    """
    ds = VehicleDataset(df, dataset_dir, transform, target, cache_dir=cache_dir,
                        draft_factor=draft_factor)
    loader = DataLoader(ds, batch_size=int(chunk), shuffle=False,
                        **loader_kwargs(num_workers))
    chunks = [imgs for imgs, _, _ in loader]
    if not chunks:
        return torch.empty(0, 3, int(target), int(target), dtype=torch.float32)
    return torch.cat(chunks, dim=0).contiguous()


def build_eval_loader(df, dataset_dir, transform, target, cache_dir=None,
                      draft_factor=2.0, batch_size=128, num_workers=0,
                      prefetch_factor=6, tensors=None):
    """DataLoader for embedding extraction (reusable across epochs)."""
    if tensors is not None:
        ds = TensorCacheDataset(tensors)
    else:
        ds = VehicleDataset(df, dataset_dir, transform, target, cache_dir=cache_dir,
                            draft_factor=draft_factor)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=False,
                      **loader_kwargs(num_workers, prefetch_factor=prefetch_factor))


@torch.no_grad()
def extract_with_loader(model, loader, device, amp_dtype=None, channels_last=False):
    """Embedding extraction over a pre-built DataLoader.

    Separated from :func:`extract_embeddings` so validation can build its
    loaders ONCE (persistent worker pool) and reuse them every epoch — a fresh
    multiprocessing pool costs ~2 s/worker to spawn on Windows.
    """
    model.eval()
    chunks = []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for imgs, _, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if channels_last:
            imgs = imgs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            emb = model.embed(imgs)
        chunks.append(emb.float().cpu().numpy())
    if not chunks:
        return np.zeros((0, model.emb_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@torch.no_grad()
def extract_embeddings(model, df, dataset_dir, transform, target, device,
                       batch_size=64, num_workers=4, amp_dtype=None,
                       cache_dir=None, draft_factor=2.0, channels_last=False,
                       prefetch_factor=6):
    loader = build_eval_loader(df, dataset_dir, transform, target,
                               cache_dir=cache_dir, draft_factor=draft_factor,
                               batch_size=batch_size, num_workers=num_workers,
                               prefetch_factor=prefetch_factor)
    return extract_with_loader(model, loader, device, amp_dtype=amp_dtype,
                               channels_last=channels_last)


def _write_gt_csv(val_query, val_gallery, path):
    """Official GT format: image_id,vehicle_id,camera_id,split in {query,gallery}."""
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy()
    q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy()
    g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def write_val_artifacts(out_dir, val_query, val_gallery, q_emb, g_emb, top_k=10):
    """Write submission/candidates/embeddings/gt/query/gallery into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "embeddings.npy"),
            np.vstack([q_emb, g_emb]).astype(np.float32))

    g_ids = val_gallery["image_id"].tolist()
    sims = q_emb @ g_emb.T  # both L2-normalized -> cosine
    order = np.argsort(-sims, axis=1, kind="stable")

    with open(os.path.join(out_dir, "submission.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv_writer(f)
        for i, qid in enumerate(val_query["image_id"]):
            k = min(top_k, len(g_ids))
            row = [qid] + [g_ids[j] for j in order[i, :k]]
            w.writerow(row)

    with open(os.path.join(out_dir, "candidates.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv_writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(val_query["image_id"]):
            if len(g_ids) == 0:
                continue
            j = int(order[i, 0])
            w.writerow([qid, g_ids[j], f"{float(sims[i, j]):.6f}"])

    val_query.to_csv(os.path.join(out_dir, "query.csv"), index=False)
    val_gallery.to_csv(os.path.join(out_dir, "gallery.csv"), index=False)
    _write_gt_csv(val_query, val_gallery, os.path.join(out_dir, "gt.csv"))
    return out_dir


def csv_writer(f):
    import csv
    return csv.writer(f)


def validate_run(out_dir):
    """Run tools/validate_format.py on the artifacts; returns parsed report."""
    script = os.path.join(REPO, "tools", "validate_format.py")
    if not os.path.exists(script):
        return {"ok": None, "error": "validate_format.py missing"}
    cmd = [sys.executable, script, "--out", out_dir,
           "--query", os.path.join(out_dir, "query.csv"),
           "--gallery", os.path.join(out_dir, "gallery.csv"), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    try:
        return json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": proc.stdout + proc.stderr}


def resolve_crop_cache(cfg: dict) -> str | None:
    """Path to the pre-cropped cache dir if it exists, else None."""
    explicit = cfg.get("crop_cache")
    if explicit:
        p = os.path.abspath(explicit)
        return p if os.path.isdir(p) else None
    size = int(cfg.get("cache_size", 320))
    p = os.path.abspath(os.path.join(REPO, "artifacts", "cache", f"train_{size}"))
    return p if os.path.isdir(p) else None


def evaluate_val(model, val_query, val_gallery, dataset_dir, cfg, device, out_dir,
                 amp_dtype=None, cache_dir=None, channels_last=False,
                 query_loader=None, gallery_loader=None):
    """Full val evaluation through the official evaluate.py. Returns report dict.

    ``query_loader``/``gallery_loader`` let the caller pass loaders that were
    built (and whose worker pool spawned) once for the whole run; when omitted
    they are built here.
    """
    size = int(cfg["image_size"])
    draft_factor = float(cfg.get("draft_factor", 2.0))
    channels_last = bool(channels_last or cfg.get("channels_last", False))
    if query_loader is None or gallery_loader is None:
        tf = build_eval_transform(size)
        eval_nw = int(cfg.get("eval_num_workers", 0))
        eval_bs = int(cfg.get("eval_batch_size", 128))
        prefetch = int(cfg.get("prefetch_factor", 6))
        if query_loader is None:
            query_loader = build_eval_loader(
                val_query, dataset_dir, tf, size, cache_dir=cache_dir,
                draft_factor=draft_factor, batch_size=eval_bs,
                num_workers=eval_nw, prefetch_factor=prefetch)
        if gallery_loader is None:
            gallery_loader = build_eval_loader(
                val_gallery, dataset_dir, tf, size, cache_dir=cache_dir,
                draft_factor=draft_factor, batch_size=eval_bs,
                num_workers=eval_nw, prefetch_factor=prefetch)
    emb_health = extract_val_artifacts(model, query_loader, gallery_loader,
                                       val_query, val_gallery, device, out_dir,
                                       amp_dtype=amp_dtype,
                                       channels_last=channels_last)
    report = run_val_official(out_dir)
    report["emb_health"] = emb_health
    return report


def extract_val_artifacts(model, query_loader, gallery_loader, val_query,
                          val_gallery, device, out_dir, amp_dtype=None,
                          channels_last=False):
    """GPU part of validation: embeddings + submission artifacts."""
    q_emb = extract_with_loader(model, query_loader, device, amp_dtype=amp_dtype,
                                channels_last=channels_last)
    g_emb = extract_with_loader(model, gallery_loader, device, amp_dtype=amp_dtype,
                                channels_last=channels_last)
    write_val_artifacts(out_dir, val_query, val_gallery, q_emb, g_emb)
    return embedding_health(q_emb)


def run_val_official(out_dir):
    """CPU-only part of validation: format check + official ``evaluate.py``.

    Kept separate so the training loop can hand it to a background thread and
    never leave the GPU idle while the official ranking runs.
    """
    fmt = validate_run(out_dir)
    report = run_official(
        gt_csv=os.path.join(out_dir, "gt.csv"),
        submission=os.path.join(out_dir, "submission.csv"),
        candidates=os.path.join(out_dir, "candidates.csv"),
        embeddings=os.path.join(out_dir, "embeddings.npy"),
        query=os.path.join(out_dir, "query.csv"),
        gallery=os.path.join(out_dir, "gallery.csv"),
        json_out=os.path.join(out_dir, "official_report.json"),
    )
    report["format_check"] = fmt
    return report


def embedding_health(emb: np.ndarray) -> dict:
    """Cheap collapse check: mean pairwise cosine of a random sample.

    Collinear (mode-collapsed) embeddings score near 1.0; healthy ReID
    embeddings are decorrelated and score well below 0.9.
    """
    if emb is None or len(emb) < 4:
        return {"mean_pairwise_cosine": None, "collinear": None}
    rng = np.random.default_rng(0)
    n = min(256, len(emb))
    idx = rng.choice(len(emb), size=n, replace=False)
    x = emb[idx].astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)
    sims = x @ x.T
    off = sims[np.triu_indices(n, k=1)]
    mean_cos = float(off.mean())
    return {"mean_pairwise_cosine": mean_cos, "collinear": bool(mean_cos > 0.98)}


def _flat_metrics(rep: dict) -> dict:
    rank = rep.get("ranking", {})
    full = rep.get("full_ranking", {})
    return {
        "mAP@10": rank.get("mAP@10"),
        "Rank-1": rank.get("Rank-1"),
        "Rank-5": rank.get("Rank-5"),
        "mAP_full": full.get("mAP_full"),
        "mINP": full.get("mINP"),
        "n_scored": rank.get("n_scored"),
        "n_openset_excluded": rank.get("n_openset_excluded"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Vehicle ReID training (W1-3)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="runs/<exp_id>")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="dataset dir with train.csv + images/")
    ap.add_argument("--epochs", type=int, default=None, help="override config")
    ap.add_argument("--init", default=None,
                    help="checkpoint to warm-start from (transfer init)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if not args.dataset:
        print("ERROR: dataset dir not given (--dataset or DATASET_DIR)", file=sys.stderr)
        return 2

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Backend knobs: let cuDNN autotune conv kernels (fixed input size) and use
    # the faster TF32 path for fp32 matmuls that AMP leaves in fp32.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"device={device}  out={out_dir}", flush=True)
    print(f"loading train.csv from {args.dataset}", flush=True)
    train_full = read_csv(os.path.join(args.dataset, "train.csv"),
                          required=TRAIN_COLUMNS)

    train_df, val_query, val_gallery = holdout_val(
        train_full,
        val_fraction=float(cfg.get("val_fraction", 0.2)),
        open_set_fraction=float(cfg.get("open_set_fraction", 0.2)),
        seed=seed,
    )
    train_df = train_df.reset_index(drop=True)
    print(f"split: train={len(train_df)} ({train_df.vehicle_id.nunique()} ids)  "
          f"val_query={len(val_query)}  val_gallery={len(val_gallery)}", flush=True)

    # ArcFace needs contiguous class indices 0..num_classes-1 (raw vehicle_id
    # values are sparse and would index the weight matrix out of bounds).
    uniq_ids = pd.unique(train_df["vehicle_id"])
    label_map = {int(v): i for i, v in enumerate(uniq_ids)}
    train_df = train_df.copy()
    train_df["label"] = train_df["vehicle_id"].map(label_map).astype("int64")
    labels = train_df["label"].to_numpy()
    num_classes = len(label_map)

    size = int(cfg["image_size"])
    cache_dir = resolve_crop_cache(cfg)
    print(f"crop cache: {cache_dir or 'none (on-the-fly draft decode)'}", flush=True)
    draft_factor = float(cfg.get("draft_factor", 2.0))
    train_tf = build_train_transform(size)
    train_ds = VehicleDataset(train_df, args.dataset, train_tf, size,
                              label_col="label", cache_dir=cache_dir,
                              draft_factor=draft_factor)
    camera_aware_pk = bool(cfg.get("camera_aware_pk", False))
    sampler = PKSampler(labels, p=int(cfg.get("p", 16)), k=int(cfg.get("k", 4)),
                        seed=seed, cameras=train_df["camera_id"].to_numpy(),
                        camera_aware=camera_aware_pk)
    if camera_aware_pk:
        print("PK sampling: camera-aware (frames per identity biased to "
              "distinct cameras)", flush=True)
    n_workers = int(cfg.get("num_workers", 8))
    prefetch_factor = int(cfg.get("prefetch_factor", 6))
    loader = DataLoader(train_ds, batch_sampler=sampler, drop_last=False,
                        **loader_kwargs(n_workers, prefetch_factor=prefetch_factor))
    # Spawn the persistent worker pool now, before the first epoch timer starts:
    # on Windows the spawn (re-importing torch in each worker) costs ~2 s/worker
    # and would otherwise be charged to epoch 0's wall time.
    if n_workers > 0:
        iter(loader)

    # Build the validation loaders ONCE and reuse them across epochs. With
    # ``persistent_workers`` the multiprocessing pool spawns a single time (the
    # ``iter`` calls below, outside any epoch timer) instead of per evaluation.
    val_query_loader = val_gallery_loader = None
    eval_nw = int(cfg.get("eval_num_workers", 0))
    eval_bs = int(cfg.get("eval_batch_size", 128))
    if bool(cfg.get("reuse_val_loader", True)):
        val_tf = build_eval_transform(size)
        if bool(cfg.get("eval_tensor_cache", True)):
            # Deterministic eval transform -> decode val once, then feed the GPU
            # from RAM (no per-epoch JPEG decode, no worker pool).
            t_cache = time.time()
            cache_nw = int(cfg.get("eval_cache_workers", 0))
            val_q_t = build_tensor_cache(val_query, args.dataset, val_tf, size,
                                         cache_dir=cache_dir, draft_factor=draft_factor,
                                         num_workers=cache_nw)
            val_g_t = build_tensor_cache(val_gallery, args.dataset, val_tf, size,
                                         cache_dir=cache_dir, draft_factor=draft_factor,
                                         num_workers=cache_nw)
            print(f"eval tensor cache: q={tuple(val_q_t.shape)} "
                  f"g={tuple(val_g_t.shape)} ({time.time() - t_cache:.1f}s)",
                  flush=True)
            val_query_loader = build_eval_loader(
                val_query, args.dataset, val_tf, size, batch_size=eval_bs,
                num_workers=0, tensors=val_q_t)
            val_gallery_loader = build_eval_loader(
                val_gallery, args.dataset, val_tf, size, batch_size=eval_bs,
                num_workers=0, tensors=val_g_t)
        else:
            val_query_loader = build_eval_loader(
                val_query, args.dataset, val_tf, size, cache_dir=cache_dir,
                draft_factor=draft_factor, batch_size=eval_bs,
                num_workers=eval_nw, prefetch_factor=prefetch_factor)
            val_gallery_loader = build_eval_loader(
                val_gallery, args.dataset, val_tf, size, cache_dir=cache_dir,
                draft_factor=draft_factor, batch_size=eval_bs,
                num_workers=eval_nw, prefetch_factor=prefetch_factor)
            if eval_nw > 0:  # pre-spawn persistent pools (startup, not epoch)
                for _ldr in (val_query_loader, val_gallery_loader):
                    iter(_ldr)

    model = build_model(
        backbone=cfg.get("backbone", "convnext_tiny"),
        num_classes=num_classes,
        emb_dim=int(cfg.get("emb_dim", 512)),
        pretrained=bool(cfg.get("pretrained", True)),
        margin=float(cfg.get("margin", 0.3)),
        scale=float(cfg.get("scale", 30.0)),
        gem_p=float(cfg.get("gem_p", 3.0)),
        image_size=size,
    ).to(device)
    init_ckpt = args.init or cfg.get("init")
    if init_ckpt:
        if not os.path.exists(init_ckpt):
            print(f"WARNING: --init checkpoint not found: {init_ckpt}", flush=True)
        else:
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            skipped = set(missing) | set(unexpected)
            print(f"transfer init from {init_ckpt}: loaded "
                  f"{len(sd) - len(skipped)}/{len(sd)} tensors"
                  + (f"; skipped={sorted(skipped)}" if skipped else ""),
                  flush=True)
    channels_last = bool(cfg.get("channels_last", False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("channels_last: enabled", flush=True)

    # AMP (bf16) — optional, speeds up ViT training on Ampere/Ada.
    amp_name = cfg.get("amp")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_name)
    if amp_dtype is not None:
        print(f"AMP enabled: {amp_name}", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.1)))
    base_lr = float(cfg.get("lr", 3e-4))
    bb_scale = float(cfg.get("backbone_lr_scale", 1.0))
    wd = float(cfg.get("weight_decay", 5e-4))
    if bb_scale != 1.0:
        bb_params = list(model.backbone.parameters())
        bb_ids = {id(p) for p in bb_params}
        head_params = [p for p in model.parameters() if id(p) not in bb_ids]
        optimizer = torch.optim.AdamW(
            [{"params": bb_params, "lr": base_lr * bb_scale},
             {"params": head_params, "lr": base_lr}], weight_decay=wd)
        print(f"optimizer: backbone lr={base_lr * bb_scale:.2e} (x{bb_scale}), "
              f"head lr={base_lr:.2e}", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    epochs = int(cfg.get("epochs", 5))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(epochs, int(cfg.get("warmup_epochs", 1))))

    use_ema = bool(cfg.get("ema", True))
    ema_decay = float(cfg.get("ema_decay", 0.999))
    ema_start = int(cfg.get("ema_start", 3))
    ema = None  # created/warm-started once the head has stopped being random
    clip_grad = cfg.get("clip_grad")
    clip_grad = float(clip_grad) if clip_grad else None
    triplet_weight = float(cfg.get("triplet_weight", 0.0) or 0.0)
    triplet_margin = float(cfg.get("triplet_margin", 0.3))
    if triplet_weight > 0.0:
        print(f"loss: ArcFace + {triplet_weight}*batch_hard_triplet"
              f"(m={triplet_margin})", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    except Exception:  # noqa: BLE001
        writer = None

    log_path = os.path.join(out_dir, "log.jsonl")
    manifest_sha = sha256_file(MANIFEST)
    best_map, best_epoch, best_metrics, best_source = -1.0, -1, {}, None
    val_every = int(cfg.get("eval_every", 1))

    def consider(metrics, source, epoch):
        nonlocal best_map, best_epoch, best_metrics, best_source
        if metrics.get("mAP@10") is None or metrics["mAP@10"] <= best_map:
            return
        best_map = metrics["mAP@10"]
        best_epoch = epoch
        best_metrics = metrics
        best_source = source
        # Called at the top of the FOLLOWING epoch, before any update, so the
        # live weights still equal the evaluated checkpoint.
        sd = (ema.module if source == "ema" else model).state_dict()
        tmp = os.path.join(out_dir, "best.pt.tmp")
        torch.save({
            "state_dict": sd,
            "config": cfg,
            "epoch": epoch,
            "metrics": metrics,
            "backbone": cfg.get("backbone"),
            "emb_dim": int(cfg.get("emb_dim", 512)),
            "weights": source,
        }, tmp)
        os.replace(tmp, os.path.join(out_dir, "best.pt"))

    # Official (CPU-only) evaluation runs in a background thread and is joined at
    # the top of the next epoch, so the GPU is never idle while the ranking runs.
    val_pool = ThreadPoolExecutor(max_workers=2)
    prev_record = None
    prev_futures = []  # (future, source, epoch, emb_health)

    def finalize_prev():
        nonlocal prev_record, prev_futures
        if prev_record is None:
            return
        record = prev_record
        for fut, source, ep, eh in prev_futures:
            try:
                rep = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  WARNING validation ({source}) failed: {e}", flush=True)
                record[f"val_{source}"] = {"mAP@10": None}
                continue
            metrics = _flat_metrics(rep)
            record[f"val_{source}"] = metrics
            record[f"emb_health_{source}"] = eh
            h = eh or {}
            if source == "ema" and h.get("collinear"):
                print("  WARNING: EMA embeddings collinear "
                      f"(mean cos={h.get('mean_pairwise_cosine')})", flush=True)
            print(f"           {source} mAP@10={metrics['mAP@10']:.4f}  "
                  f"Rank-1={metrics['Rank-1']:.4f}  "
                  f"Rank-5={metrics['Rank-5']:.4f}  mINP={metrics['mINP']:.4f}",
                  flush=True)
            consider(metrics, source, ep)
        if writer is not None:
            writer.add_scalar("train/loss", record["loss"], record["epoch"])
            writer.add_scalar("train/lr", record["lr"], record["epoch"])
            for tag in ("val_raw", "val_ema"):
                if tag in record and (record[tag] or {}).get("mAP@10") is not None:
                    writer.add_scalar(f"{tag}/mAP@10", record[tag]["mAP@10"],
                                      record["epoch"])
                    writer.add_scalar(f"{tag}/Rank-1", record[tag]["Rank-1"],
                                      record["epoch"])
                    writer.add_scalar(f"{tag}/mINP", record[tag]["mINP"],
                                      record["epoch"])
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        prev_record = None
        prev_futures = []

    for epoch in range(epochs):
        finalize_prev()
        sampler.set_epoch(epoch)
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, criterion, device,
                               log_every=max(1, len(loader) // 5),
                               amp_dtype=amp_dtype, ema=ema, clip_grad=clip_grad,
                               triplet_weight=triplet_weight,
                               triplet_margin=triplet_margin)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        record = {"epoch": epoch, "loss": loss, "lr": lr_now,
                  "sec": round(time.time() - t0, 1)}

        # Warm-start EMA only once the randomly-initialised head has trained a
        # few epochs; an EMA anchored at the random init poisons the average.
        if use_ema and ema is None and epoch >= ema_start:
            ema = EMA(model, decay=ema_decay)
            print(f"  EMA warm-started at epoch {epoch} (decay={ema_decay})", flush=True)

        print(f"  epoch {epoch}: loss={loss:.4f}  ({record['sec']}s)", flush=True)
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            # Evaluate only the source that is actually deployed: raw before the
            # EMA exists, the EMA afterwards ("auto"). "both" keeps the old
            # raw+ema comparison; "raw" pins the live weights.
            eval_sources = str(cfg.get("eval_sources", "auto"))
            sources = []
            if ema is not None and eval_sources != "raw":
                if eval_sources == "both":
                    sources.append(("raw", model))
                sources.append(("ema", ema.module))
            else:
                sources.append(("raw", model))

            for source, net in sources:
                vdir = os.path.join(out_dir, f"val_{source}")
                eh = extract_val_artifacts(
                    net, val_query_loader, val_gallery_loader, val_query,
                    val_gallery, device, vdir, amp_dtype=amp_dtype,
                    channels_last=channels_last)
                prev_futures.append(
                    (val_pool.submit(run_val_official, vdir), source, epoch, eh))
        prev_record = record

    finalize_prev()
    val_pool.shutdown(wait=True)
    if writer is not None:
        writer.close()

    try:
        import torch as _t
        env = {"torch": _t.__version__, "cuda": _t.version.cuda,
               "python": sys.version.split()[0],
               "gpu": _t.cuda.get_device_name(0) if _t.cuda.is_available() else None,
               "amp": amp_name, "ema": use_ema}
    except Exception:  # noqa: BLE001
        env = {}

    report = {
        "exp_name": cfg.get("exp_name"),
        "config": cfg,
        "seed": seed,
        "data_manifest_sha": manifest_sha,
        "git_sha": git_sha(),
        "env": env,
        "split": {"n_train": len(train_df), "n_val_query": len(val_query),
                  "n_val_gallery": len(val_gallery),
                  "n_train_ids": num_classes},
        "train": {"epochs": epochs, "best_epoch": best_epoch,
                  "batch_size": int(cfg.get("p", 16)) * int(cfg.get("k", 4)),
                  "weights": best_source,
                  "ema": use_ema, "ema_decay": ema_decay if use_ema else None,
                  "ema_start": ema_start if use_ema else None,
                  "amp": amp_name, "clip_grad": clip_grad,
                  "triplet_weight": triplet_weight,
                  "triplet_margin": triplet_margin if triplet_weight > 0 else None,
                  "init": init_ckpt},
        "val": best_metrics,
        "artifacts": {
            "checkpoint": os.path.join(os.path.relpath(out_dir, REPO), "best.pt"),
            "report": os.path.join(os.path.relpath(out_dir, REPO), "report.json"),
            "log": os.path.join(os.path.relpath(out_dir, REPO), "log.jsonl"),
            "tensorboard": os.path.join(os.path.relpath(out_dir, REPO), "tb"),
        },
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nFINAL best epoch", best_epoch, "metrics:", json.dumps(best_metrics,
                                                                  ensure_ascii=False))
    print("checkpoint:", os.path.join(out_dir, "best.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
