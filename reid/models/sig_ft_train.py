"""In-domain PARTIAL fine-tune of the external SigLIP2 NaFlex vehicle-ReID model.

The champion's SigLIP2 branch is frozen (0.602 val mAP@10 alone / 0.6495 with
k-reciprocal). The only remaining lever is *in-domain* adaptation. Full
fine-tuning (exp-0034) regressed (0.5186) — and that run was additionally
crippled by dropping the released neck affine (see ``siglip2_reid.py``). Here we
freeze the backbone and adapt only:

    released backbone (frozen)  -> attention pooler (frozen)
      -> released neck affine (trainable)  -> residual Adapter (trainable)
      -> proj (trainable, init from release) -> BNNeck -> ArcFace
    + last N transformer blocks unfrozen (small lr)
    + optional LoRA on q/v projections

Recipe: PK sampling over vehicle_id, cosine LR + warmup, label smoothing, bf16,
EMA, best-by-val-mAP checkpoint. Metrics ONLY via the official ``evaluate.py``.
camera_id is never a network input (split / sampling only).

    python -m reid.models.sig_ft_train \
        --bundle runs/external/vehicle_reid_siglip2_naflex_512d.pth \
        --dataset "docs/<ds>/dataset" --out runs/exp-0072-sig-ft \
        --epochs 10 --unfreeze 4 --lora 8 --adapter 256
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
from torch.utils.data import DataLoader

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.finetune_siglip2 import PatchDataset, build_eval_tf, build_train_tf
from reid.models.siglip2_reid import build_siglip2_reid
from reid.train import (EMA, PKSampler, _flat_metrics, embedding_health, git_sha,
                        loader_kwargs, make_lr_lambda, run_val_official, set_seed,
                        sha256_file, write_val_artifacts)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SigLIP2 NaFlex in-domain partial FT")
    ap.add_argument("--bundle", default="runs/external/vehicle_reid_siglip2_naflex_512d.pth")
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--unfreeze", type=int, default=4, help="last N transformer blocks")
    ap.add_argument("--lora", type=int, default=8, help="LoRA rank on q/v (0=off)")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--adapter", type=int, default=256, help="adapter hidden (0=off)")
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5.0e-5, help="head lr")
    ap.add_argument("--backbone_lr", type=float, default=1.0e-5)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--ema_start", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoke", type=int, default=0)
    return ap.parse_args(argv)


def find_dataset(arg):
    if arg:
        return arg
    import glob
    cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
    if not cand:
        raise SystemExit("ERROR: --dataset required (auto-detect failed)")
    return os.path.dirname(cand[0])


@torch.no_grad()
def extract_bank(model, df, dataset_dir, transform, device, amp_dtype, bs=64):
    ds = PatchDataset(df, dataset_dir, transform, label_col="vehicle_id")
    loader = DataLoader(ds, batch_size=int(bs), shuffle=False, **loader_kwargs(0))
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


def eval_val(model, val_query, val_gallery, dataset_dir, device, amp_dtype,
             workdir):
    q = extract_bank(model, val_query, dataset_dir, build_eval_tf(), device, amp_dtype)
    g = extract_bank(model, val_gallery, dataset_dir, build_eval_tf(), device, amp_dtype)
    eh = embedding_health(q)
    write_val_artifacts(workdir, val_query, val_gallery, q, g)
    rep = run_val_official(workdir)
    return _flat_metrics(rep), eh, np.vstack([q, g]).astype(np.float32)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    args = parse_args(argv)
    args.dataset = find_dataset(args.dataset)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 3
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    amp_dtype = torch.bfloat16

    train_full = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    train_df, val_query, val_gallery = holdout_val(
        train_full, val_fraction=0.2, open_set_fraction=0.2, seed=args.seed)
    train_df = train_df.reset_index(drop=True)
    if args.smoke:
        train_df = train_df.head(args.smoke * args.p * args.k).reset_index(drop=True)
        val_query = val_query.head(32).reset_index(drop=True)
        val_gallery = val_gallery.head(128).reset_index(drop=True)
    uniq = pd.unique(train_df["vehicle_id"])
    label_map = {int(v): i for i, v in enumerate(uniq)}
    train_df["label"] = train_df["vehicle_id"].map(label_map).astype("int64")
    num_classes = len(label_map)
    print(f"train={len(train_df)} ids={num_classes} val_q={len(val_query)} "
          f"val_g={len(val_gallery)}", flush=True)

    model = build_siglip2_reid(
        os.path.join(REPO, args.bundle), num_classes=num_classes,
        emb_dim=args.emb_dim, margin=args.margin, scale=args.scale,
        image_size=args.image_size, adapter_hidden=args.adapter).to(device)
    info = model.configure_trainable(n_unfreeze=args.unfreeze, lora_rank=args.lora,
                                     lora_alpha=args.lora_alpha,
                                     train_adapter=True, train_proj=True)
    print(f"[model] trainable {info['trainable']/1e6:.2f}M / {info['total']/1e6:.1f}M "
          f"({info['trainable_pct']:.2f}%)  unfreeze={args.unfreeze} "
          f"lora={args.lora} adapter={args.adapter}", flush=True)

    # ---- sanity: pretrained (neck-fixed) val, expect ~0.602 ----
    t0 = time.time()
    m_pre, eh_pre, emb_pre = eval_val(model, val_query, val_gallery, args.dataset,
                                      device, amp_dtype, os.path.join(out_dir, "val_pretrained"))
    print(f"[pretrained] mAP@10={m_pre['mAP@10']:.4f} R1={m_pre['Rank-1']:.4f} "
          f"mINP={m_pre['mINP']:.4f} collinear={eh_pre['collinear']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(out_dir, "pretrained_embeddings.npy"), emb_pre)

    # ---- train ----
    train_ds = PatchDataset(train_df, args.dataset, build_train_tf(), label_col="label")
    sampler = PKSampler(train_df["label"].to_numpy(), p=args.p, k=args.k, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, drop_last=False,
                              **loader_kwargs(args.num_workers))
    if args.num_workers > 0:
        iter(train_loader)

    bb_ids = {id(p) for p in model.backbone.parameters()}
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) in bb_ids]
    hd_params = [p for p in model.parameters() if p.requires_grad and id(p) not in bb_ids]
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": args.backbone_lr},
         {"params": hd_params, "lr": args.lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(args.epochs, args.warmup_epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ema = None
    log_path = os.path.join(out_dir, "log.jsonl")
    best = {"map": -1.0, "epoch": -1, "source": None, "metrics": {}}
    manifest_sha = sha256_file(os.path.join(REPO, "artifacts", "data_manifest.json"))

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        loss_sum = None
        n = 0
        for pv, mask, shapes, labels in train_loader:
            pv = pv.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            shapes = shapes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits, _ = model(pv, mask, shapes, labels)
                loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.clip_grad)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            bs = pv.size(0)
            with torch.no_grad():
                batch_loss = loss.detach().float() * bs
            loss_sum = batch_loss if loss_sum is None else loss_sum + batch_loss
            n += bs
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        mean_loss = float(loss_sum) / max(1, n)
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"non-finite loss {mean_loss} — abort")
        rec = {"epoch": epoch, "loss": mean_loss, "lr": lr_now,
               "sec": round(time.time() - t0, 1)}
        print(f"  epoch {epoch}: loss={mean_loss:.4f} lr={lr_now:.2e} "
              f"({rec['sec']}s)", flush=True)

        if args.ema and ema is None and epoch >= args.ema_start:
            ema = EMA(model, decay=args.ema_decay)
            print(f"  EMA warm-started (decay={args.ema_decay})", flush=True)

        sources = []
        if ema is not None:
            sources.append(("ema", ema.module))
            if epoch == 0 or (epoch + 1) % 3 == 0:
                sources.append(("raw", model))
        else:
            sources.append(("raw", model))

        for source, net in sources:
            metrics, eh, _ = eval_val(net, val_query, val_gallery, args.dataset,
                                      device, amp_dtype,
                                      os.path.join(out_dir, f"val_{source}_e{epoch}"))
            rec[f"val_{source}"] = metrics
            print(f"           {source} mAP@10={metrics['mAP@10']:.4f} "
                  f"R1={metrics['Rank-1']:.4f} R5={metrics['Rank-5']:.4f} "
                  f"mINP={metrics['mINP']:.4f}", flush=True)
            if (metrics.get("mAP@10") or -1) > best["map"]:
                best = {"map": metrics["mAP@10"], "epoch": epoch,
                        "source": source, "metrics": metrics}
                sd = (ema.module if source == "ema" else model).state_dict()
                tmp = os.path.join(out_dir, "best.pt.tmp")
                torch.save({"state_dict": sd, "config": vars(args), "epoch": epoch,
                            "metrics": metrics, "backbone": "siglip2_naflex",
                            "emb_dim": args.emb_dim, "weights": source}, tmp)
                os.replace(tmp, os.path.join(out_dir, "best.pt"))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- final: best checkpoint -> val bank + report ----
    if os.path.exists(os.path.join(out_dir, "best.pt")):
        ck = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device)
    # rebuild trainable flags not needed for eval
    m_best, eh_best, emb_best = eval_val(model, val_query, val_gallery, args.dataset,
                                         device, amp_dtype, os.path.join(out_dir, "val"))
    np.save(os.path.join(out_dir, "embeddings.npy"), emb_best)
    print(f"BEST epoch={best['epoch']} ({best['source']}) "
          f"mAP@10={best['map']:.4f}", flush=True)

    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "amp": "bf16",
           "ema": bool(args.ema)}
    report = {"exp_name": os.path.basename(out_dir), "git_sha": git_sha(),
              "data_manifest_sha": manifest_sha, "config": vars(args),
              "seed": args.seed, "env": env,
              "model_info": info,
              "pretrained_val": m_pre,
              "train": {"epochs": args.epochs, "best_epoch": best["epoch"],
                        "best_source": best["source"]},
              "val": best["metrics"], "val_best_recheck": m_best,
              "emb_health": eh_best,
              "artifacts": {
                  "checkpoint": os.path.relpath(os.path.join(out_dir, "best.pt"), REPO),
                  "embeddings": os.path.relpath(os.path.join(out_dir, "embeddings.npy"), REPO),
                  "report": os.path.relpath(os.path.join(out_dir, "report.json"), REPO),
                  "log": os.path.relpath(log_path, REPO)}}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
