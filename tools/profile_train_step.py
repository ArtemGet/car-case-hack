"""Per-op profiler for one DINOv2-B training step (task W1-8).

Measures, per step, the wall-clock decomposition of ``train_one_epoch``:
dataloader wait, H2D copy, forward (+loss), backward, optimizer step, EMA
update — using CUDA events for the GPU regions (one sync per step) and
``perf_counter`` for CPU wait. Optionally also measures a pure-GPU ceiling
(fixed batch already resident on the device) and one validation pass.

    python tools/profile_train_step.py --config configs/dinov2_b.yaml \
        --dataset "docs/Датасет/dataset" --steps 30 --json reports/profile_before.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid.data.aug import build_train_transform, build_eval_transform  # noqa: E402
from reid.data.io import read_csv, TRAIN_COLUMNS  # noqa: E402
from reid.data.splits import holdout_val  # noqa: E402
from reid.models import build_model  # noqa: E402
from reid.train import (  # noqa: E402
    EMA, PKSampler, VehicleDataset, extract_embeddings, loader_kwargs, load_config,
    set_seed,
)


def _ev():
    return torch.cuda.Event(enable_timing=True)


def profile_training(cfg, dataset, steps, device, use_ema):
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    train_full = read_csv(os.path.join(dataset, "train.csv"), required=TRAIN_COLUMNS)
    train_df, val_q, val_g = holdout_val(
        train_full, val_fraction=float(cfg.get("val_fraction", 0.2)),
        open_set_fraction=float(cfg.get("open_set_fraction", 0.2)), seed=seed)
    train_df = train_df.reset_index(drop=True)

    uniq = list(dict.fromkeys(train_df["vehicle_id"].tolist()))
    lmap = {int(v): i for i, v in enumerate(uniq)}
    train_df["label"] = train_df["vehicle_id"].map(lmap).astype("int64")
    labels = train_df["label"].to_numpy()
    num_classes = len(lmap)

    size = int(cfg["image_size"])
    cache_dir = os.path.abspath(os.path.join("artifacts", "cache", f"train_{cfg.get('cache_size', 320)}"))
    cache_dir = cache_dir if os.path.isdir(cache_dir) else None
    print(f"cache_dir={cache_dir}", flush=True)

    train_tf = build_train_transform(size)
    ds = VehicleDataset(train_df, dataset, train_tf, size, label_col="label",
                        cache_dir=cache_dir, draft_factor=float(cfg.get("draft_factor", 1.2)))
    sampler = PKSampler(labels, p=int(cfg.get("p", 16)), k=int(cfg.get("k", 4)), seed=seed)
    nw = int(cfg.get("num_workers", 12))
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, drop_last=False,
                                         **loader_kwargs(nw))
    print(f"loader: num_workers={nw} kw={loader_kwargs(nw)} batch={sampler.p*sampler.k} "
          f"steps/epoch={len(loader)}", flush=True)

    model = build_model(
        backbone=cfg.get("backbone", "convnext_tiny"), num_classes=num_classes,
        emb_dim=int(cfg.get("emb_dim", 512)), pretrained=bool(cfg.get("pretrained", True)),
        margin=float(cfg.get("margin", 0.3)), scale=float(cfg.get("scale", 30.0)),
        gem_p=float(cfg.get("gem_p", 3.0)), image_size=size).to(device)

    amp_name = cfg.get("amp")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_name)

    opt = torch.optim.AdamW([{"params": list(model.backbone.parameters()),
                              "lr": float(cfg.get("lr", 3e-4)) * float(cfg.get("backbone_lr_scale", 1.0))},
                             {"params": [p for n, p in model.named_parameters()
                                         if not n.startswith("backbone.")],
                              "lr": float(cfg.get("lr", 3e-4))}],
                            weight_decay=float(cfg.get("weight_decay", 5e-4)))
    crit = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.1)))
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999))) if use_ema else None

    it = iter(loader)
    # warmup (also warms workers)
    for _ in range(3):
        imgs, lab, _ = next(iter(loader))
        imgs = imgs.to(device, non_blocking=True); lab = lab.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(imgs, lab); loss = crit(logits, lab)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ema:
            ema.update(model)
    torch.cuda.synchronize()

    rec = {k: [] for k in ("data", "h2d", "fwd", "bwd", "opt", "ema", "total")}
    img_total = 0
    it = iter(loader)
    for s in range(steps):
        t0 = time.perf_counter()
        imgs, lab, _ = next(it)
        t1 = time.perf_counter()
        imgs = imgs.to(device, non_blocking=True); lab = lab.to(device, non_blocking=True)
        e = [_ev() for _ in range(5)]
        e[0].record()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(imgs, lab)
            loss = crit(logits, lab)
        e[1].record()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        e[2].record()
        opt.step()
        e[3].record()
        if ema:
            ema.update(model)
        e[4].record()
        t2 = time.perf_counter()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        rec["data"].append((t1 - t0) * 1000)
        rec["h2d"].append(0.0)
        rec["fwd"].append(e[0].elapsed_time(e[1]))
        rec["bwd"].append(e[1].elapsed_time(e[2]))
        rec["opt"].append(e[2].elapsed_time(e[3]))
        rec["ema"].append(e[3].elapsed_time(e[4]))
        rec["total"].append((t3 - t0) * 1000)
        img_total += imgs.size(0)

    out = {k: {"mean_ms": st.mean(v), "median_ms": st.median(v),
               "p90_ms": sorted(v)[int(0.9 * (len(v) - 1))]} for k, v in rec.items()}
    total_s = sum(rec["total"])
    out["_meta"] = {"steps": steps, "imgs": img_total, "sec": total_s,
                    "img_s": img_total / total_s if total_s else 0.0}
    return out


def profile_validation(cfg, dataset, device, out_dir):
    """Time the pieces of one ``evaluate_val`` (extract raw + official CPU)."""
    seed = int(cfg.get("seed", 42))
    train_full = read_csv(os.path.join(dataset, "train.csv"), required=TRAIN_COLUMNS)
    train_df, val_q, val_g = holdout_val(
        train_full, val_fraction=float(cfg.get("val_fraction", 0.2)),
        open_set_fraction=float(cfg.get("open_set_fraction", 0.2)), seed=seed)
    uniq = list(dict.fromkeys(train_df["vehicle_id"].tolist()))
    size = int(cfg["image_size"])
    cache_dir = os.path.abspath(os.path.join("artifacts", "cache", f"train_{cfg.get('cache_size', 320)}"))
    cache_dir = cache_dir if os.path.isdir(cache_dir) else None
    model = build_model(backbone=cfg.get("backbone"), num_classes=len(uniq),
                        emb_dim=int(cfg.get("emb_dim", 512)), pretrained=False,
                        image_size=size).to(device)
    amp_name = cfg.get("amp")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_name)
    tf = build_eval_transform(size)
    nw = int(cfg.get("num_workers", 8))
    res = {}
    embs = {}
    for name, df in (("query", val_q), ("gallery", val_g)):
        t0 = time.perf_counter()
        emb = extract_embeddings(model, df, dataset, tf, size, device, batch_size=64,
                                 num_workers=nw, amp_dtype=amp_dtype,
                                 cache_dir=cache_dir, draft_factor=float(cfg.get("draft_factor", 1.2)))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        embs[name] = emb
        res[name] = {"n": len(df), "sec": dt, "img_s": len(df) / dt if dt else 0.0}
    from reid.eval.harness import run_official
    from reid.train import write_val_artifacts
    q_emb, g_emb = embs["query"], embs["gallery"]
    t0 = time.perf_counter()
    write_val_artifacts(out_dir, val_q, val_g, q_emb, g_emb)
    t_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    rep = run_official(
        gt_csv=os.path.join(out_dir, "gt.csv"), submission=os.path.join(out_dir, "submission.csv"),
        candidates=os.path.join(out_dir, "candidates.csv"),
        embeddings=os.path.join(out_dir, "embeddings.npy"),
        query=os.path.join(out_dir, "query.csv"), gallery=os.path.join(out_dir, "gallery.csv"),
        json_out=os.path.join(out_dir, "official_report.json"))
    t_official = time.perf_counter() - t0
    res["write_artifacts_sec"] = t_write
    res["official_cpu_sec"] = t_official
    res["total_sec"] = (res["query"]["sec"] + res["gallery"]["sec"] + t_official)
    res["mAP@10"] = rep.get("ranking", {}).get("mAP@10")
    return res


def profile_pure_gpu(cfg, num_classes, device, batch=64, iters=30):
    """GPU ceiling: forward+backward+step on a fixed device-resident batch."""
    amp_name = cfg.get("amp")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_name)
    model = build_model(backbone=cfg.get("backbone"), num_classes=int(num_classes),
                        emb_dim=int(cfg.get("emb_dim", 512)), pretrained=False,
                        margin=float(cfg.get("margin", 0.3)), scale=float(cfg.get("scale", 30.0)),
                        gem_p=float(cfg.get("gem_p", 3.0)), image_size=int(cfg["image_size"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, int(cfg["image_size"]), int(cfg["image_size"]), device=device)
    y = torch.randint(0, num_classes, (batch,), device=device)
    for _ in range(5):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out, _ = model(x, y); loss = crit(out, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out, _ = model(x, y); loss = crit(out, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"batch": batch, "iters": iters, "img_s": batch * iters / dt,
            "ms_step": dt / iters * 1000, "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--pure-gpu", action="store_true")
    ap.add_argument("--val", action="store_true", help="also profile one val extract")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    report = {"config": args.config, "device": str(device)}
    report["training"] = profile_training(cfg, args.dataset, args.steps, device,
                                          use_ema=bool(cfg.get("ema", True)))
    if args.val:
        vdir = os.path.join("artifacts", "perf_profile", "val")
        os.makedirs(vdir, exist_ok=True)
        report["validation"] = profile_validation(cfg, args.dataset, device, vdir)
    if args.pure_gpu:
        report["pure_gpu"] = profile_pure_gpu(cfg, 1233, device)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
