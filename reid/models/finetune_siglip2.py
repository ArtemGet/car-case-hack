"""Short fine-tune of the external SigLIP2 NaFlex vehicle-ReID backbone.

    python -m reid.models.finetune_siglip2 --bundle runs/external/...pth \
        --dataset "docs/<ds>/dataset" --out runs/exp-0033-siglip2-ft --epochs 12

Fine-tunes the public VeRi/VERI-Wild pretrained backbone (weights from
``runs/<exp>/external_sources.json``) on the train split with ArcFace + BNNeck,
PK sampling over ``vehicle_id``, cosine LR + warmup, label smoothing and bf16.
Best-by-val-mAP checkpoint, report.json, log.jsonl. Metrics only via the
OFFICIAL ``evaluate.py`` (reid.eval.harness). camera_id never enters the net.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from reid.data.crop import open_cropped
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.siglip2_reid import MAX_PATCHES, PATCH, build_siglip2_reid, patchify_batch
from reid.train import (PKSampler, _flat_metrics, embedding_health, git_sha,
                        loader_kwargs, make_lr_lambda, run_val_official, set_seed,
                        sha256_file, write_val_artifacts)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def build_train_tf():
    return T.Compose([
        T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.05),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def build_eval_tf():
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def nat_grid(w, h, max_patches=256, patch=16):
    """Patch grid (rows, cols) preserving the crop's natural aspect ratio."""
    aspect = float(w) / max(1.0, float(h))
    rows = max(1, min(16, int(round((max_patches / aspect) ** 0.5))))
    cols = max(1, int(round(max_patches / rows)))
    if rows * cols > max_patches:
        cols = max_patches // rows
    return rows, cols


class PatchDataset(Dataset):
    """Natural-aspect bbox crop -> NaFlex patch sequence (same grid as the ONNX)."""

    def __init__(self, df, dataset_dir, transform, size=None, label_col="label",
                 cache_dir=None, draft_factor=2.0):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.label_col = label_col
        self.cache_dir = cache_dir
        self.draft_factor = float(draft_factor)

    def __len__(self):
        return len(self.df)

    def _raw_crop(self, r):
        from reid.data.io import image_path
        with Image.open(image_path(self.dataset_dir, r.image_id)) as im:
            im = im.convert("RGB")
            ow, oh = im.size
            x0 = min(max(int(round(r.x)), 0), ow - 1)
            y0 = min(max(int(round(r.y)), 0), oh - 1)
            x1 = min(max(int(round(r.x + r.w)), x0 + 1), ow)
            y1 = min(max(int(round(r.y + r.h)), y0 + 1), oh)
            return im.crop((x0, y0, x1, y1))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._raw_crop(r)
        rows, cols = nat_grid(*img.size)
        img = img.resize((cols * PATCH, rows * PATCH), Image.BILINEAR)
        x = self.transform(img)  # (3, rows*16, cols*16)
        pv, mask, shapes = patchify_batch(x.unsqueeze(0))
        label = int(r[self.label_col]) if self.label_col in self.df.columns else -1
        return pv[0], mask[0], shapes[0], label


@torch.no_grad()
def extract(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for pv, mask, shapes, _ in loader:
        pv = pv.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        shapes = shapes.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            emb = model.embed(pv, mask, shapes)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros((0, 512), np.float32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune SigLIP2 vehicle ReID")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--backbone_lr_scale", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    args = parse_args(argv)
    if not args.dataset:
        print("ERROR: --dataset or DATASET_DIR required", file=sys.stderr)
        return 2
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    out_dir = os.path.abspath(args.out); os.makedirs(out_dir, exist_ok=True)
    amp_dtype = torch.bfloat16

    train_full = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    train_df, val_query, val_gallery = holdout_val(
        train_full, val_fraction=0.2, open_set_fraction=0.2, seed=args.seed)
    train_df = train_df.reset_index(drop=True)
    uniq = pd.unique(train_df["vehicle_id"])
    label_map = {int(v): i for i, v in enumerate(uniq)}
    train_df["label"] = train_df["vehicle_id"].map(label_map).astype("int64")
    num_classes = len(label_map)
    print(f"train={len(train_df)} ids={num_classes} val_q={len(val_query)} "
          f"val_g={len(val_gallery)}", flush=True)

    cache_dir = os.path.abspath(os.path.join(REPO, "artifacts", "cache", "train_320"))
    cache_dir = cache_dir if os.path.isdir(cache_dir) else None
    size = int(args.size)

    train_ds = PatchDataset(train_df, args.dataset, build_train_tf(),
                            label_col="label", cache_dir=cache_dir)
    sampler = PKSampler(train_df["label"].to_numpy(), p=args.p, k=args.k, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, drop_last=False,
                              **loader_kwargs(args.num_workers))
    if args.num_workers > 0:
        iter(train_loader)

    val_q_tf = build_eval_tf()
    val_q_ds = PatchDataset(val_query, args.dataset, val_q_tf, label_col="vehicle_id",
                            cache_dir=cache_dir)
    val_g_ds = PatchDataset(val_gallery, args.dataset, val_q_tf, label_col="vehicle_id",
                            cache_dir=cache_dir)
    val_q_loader = DataLoader(val_q_ds, batch_size=64, shuffle=False,
                              **loader_kwargs(min(4, args.num_workers)))
    val_g_loader = DataLoader(val_g_ds, batch_size=64, shuffle=False,
                              **loader_kwargs(min(4, args.num_workers)))
    if args.num_workers > 0:
        iter(val_q_loader); iter(val_g_loader)

    model = build_siglip2_reid(args.bundle, num_classes=num_classes, emb_dim=args.emb_dim,
                               margin=args.margin, scale=args.scale,
                               image_size=size).to(device)
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    print(f"model: backbone {n_bb/1e6:.1f}M params, emb {args.emb_dim}", flush=True)

    bb_params = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": args.lr * args.backbone_lr_scale},
         {"params": head_params, "lr": args.lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.epochs, args.warmup_epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    log_path = os.path.join(out_dir, "log.jsonl")
    best_map, best_epoch, best_metrics = -1.0, -1, {}
    manifest_sha = sha256_file(os.path.join(REPO, "artifacts", "data_manifest.json"))

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time(); loss_sum = 0.0; n = 0
        for step, (pv, mask, shapes, labels) in enumerate(train_loader):
            pv = pv.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            shapes = shapes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits, _ = model(pv, mask, shapes, labels)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * pv.size(0); n += pv.size(0)
            if not math.isfinite(loss_sum):
                raise RuntimeError("non-finite loss (sanity)")
            if (step + 1) % max(1, len(train_loader) // 4) == 0:
                print(f"    step {step+1}/{len(train_loader)} loss={loss_sum/n:.4f}", flush=True)
        lr_now = optimizer.param_groups[0]["lr"]; scheduler.step()
        rec = {"epoch": epoch, "loss": loss_sum / max(1, n), "lr": lr_now,
               "sec": round(time.time() - t0, 1)}
        print(f"  epoch {epoch}: loss={rec['loss']:.4f} ({rec['sec']}s)", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            q = extract(model, val_q_loader, device, amp_dtype)
            g = extract(model, val_g_loader, device, amp_dtype)
            vdir = os.path.join(out_dir, "val")
            eh = embedding_health(q)
            write_val_artifacts(vdir, val_query, val_gallery, q, g)
            rep = run_val_official(vdir)
            m = _flat_metrics(rep)
            rec["val"] = m; rec["emb_health"] = eh
            print(f"           mAP@10={m['mAP@10']:.4f} Rank-1={m['Rank-1']:.4f} "
                  f"Rank-5={m['Rank-5']:.4f} mINP={m['mINP']:.4f}", flush=True)
            if (m.get("mAP@10") or -1) > best_map:
                best_map = m["mAP@10"]; best_epoch = epoch; best_metrics = m
                torch.save({"state_dict": model.state_dict(), "config": vars(args),
                            "epoch": epoch, "metrics": m, "verdict": "candidate"},
                           os.path.join(out_dir, "best.pt"))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {"exp_name": os.path.basename(out_dir), "git_sha": git_sha(),
              "data_manifest_sha": manifest_sha, "config": vars(args),
              "seed": args.seed, "env": {"torch": torch.__version__,
                                         "cuda": torch.version.cuda},
              "train": {"epochs": args.epochs, "best_epoch": best_epoch},
              "val": best_metrics,
              "artifacts": {"checkpoint": os.path.join(os.path.relpath(out_dir, REPO), "best.pt"),
                            "report": os.path.join(os.path.relpath(out_dir, REPO), "report.json"),
                            "log": os.path.join(os.path.relpath(out_dir, REPO), "log.jsonl")}}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"BEST epoch {best_epoch}: {json.dumps(best_metrics, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
