"""Head-only training on FROZEN in-house DINOv2-B features, then fusion of heads.

Mirrors ``head_frozen.py`` (frozen SigLIP2 head-only) for the second backbone:
freeze DINOv2-B (champion exp-0007), extract 512-d features for the train split
on CUDA, learn only ``Linear -> BNNeck -> ArcFace`` on top with PK-sampling, then:
  * eval the head-transformed DINOv2 alone (val, official evaluate.py);
  * fuse it with the frozen SigLIP2 (raw and head-transformed) at the best
    k-reciprocal setting, to test whether a sharper DINOv2 subspace lifts the
    deployed fusion.

Red lines: no test data; camera_id never enters the network; per-query ranking;
CUDA mandatory for extraction and head training (no CPU forward).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.models.head_frozen import (Head, l2, pk_batches, rerank_eval)
from service.infer.backends import DinoBackend

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@torch.no_grad()
def embed_head(head, x, device):
    head.eval()
    return F.normalize(head.embed(torch.from_numpy(x).to(device)), dim=1).cpu().numpy()


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-ckpt", default="runs/exp-0007/best.pt")
    ap.add_argument("--val-sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--sig-head", default="runs/exp-0047-head/head_best.pt")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr); return 2
        args.dataset = os.path.dirname(cand[0])
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    images_dir = os.path.join(args.dataset, "images")
    print(f"[dataset] {args.dataset}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for head-only DINOv2 run; refusing CPU")
    device = torch.device("cuda")
    print(f"[cuda] {torch.cuda.get_device_name(0)}", flush=True)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    tr, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    tr = tr.reset_index(drop=True); vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    print(f"[split] train={len(tr)} val_q={nq} val_g={len(vg)}", flush=True)

    be = DinoBackend(args.dino_ckpt, device, amp="bf16", batch_size=64,
                     num_workers=0)
    ctr = os.path.join(out, "dino_train.npy")
    cval = os.path.join(out, "dino_val.npy")
    if args.reuse_cache and os.path.exists(ctr):
        Xtr = np.load(ctr).astype(np.float32)
    else:
        print("[extract] dino train ...", flush=True)
        Xtr = be.extract(tr, images_dir, args.size)
        np.save(ctr, Xtr)
    if args.reuse_cache and os.path.exists(cval):
        Xval = np.load(cval).astype(np.float32)
    else:
        print("[extract] dino val ...", flush=True)
        Xval = be.extract(vq, images_dir, args.size)
        Xvg = be.extract(vg, images_dir, args.size)
        Xval = np.concatenate([Xval, Xvg], axis=0)
        np.save(cval, Xval)
    del be
    torch.cuda.empty_cache()
    assert Xtr.shape[0] == len(tr) and Xval.shape[0] == nq + len(vg)

    # gt / query / gallery for official eval
    import pandas as pd
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()
    evaldir = os.path.join(out, "eval")

    # raw DINOv2 (base size) reference with the champion rerank setting
    rr_best = {"k1": 8, "k2": 3, "lam": 0.5, "pool_size": 200}
    dv = l2(Xval)
    m_raw = rerank_eval(dv[:nq], dv[nq:], q_ids, g_ids, gt, qcsv, gcsv,
                        os.path.join(evaldir, "raw"), rr_best, "raw")
    print(f"[raw dino {args.size}] mAP@10={m_raw['mAP@10']:.4f} "
          f"R1={m_raw['Rank-1']:.4f}", flush=True)

    Xtr_n = l2(Xtr)
    ytr = tr["vehicle_id"].to_numpy().astype(np.int64)
    classes = sorted(set(ytr.tolist()))
    cls_map = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([cls_map[c] for c in ytr], dtype=np.int64)
    C = len(classes)

    Xt = torch.from_numpy(Xtr_n).to(device)
    yt = torch.from_numpy(ytr_i).to(device)
    head = Head(Xtr_n.shape[1], args.emb_dim, C, args.margin, args.scale).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr_n) // (args.P * args.K))
    total = max(1, args.epochs * steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total,
                                                       eta_min=args.lr * 0.1)
    rng = np.random.default_rng(args.seed)
    hist = []; best_m = -1.0; best_ep = 0; best_state = None
    for ep in range(1, args.epochs + 1):
        head.train(); run = 0.0; nb = 0
        for idx in pk_batches(ytr_i, args.P, args.K, steps, rng):
            xb = Xt[torch.from_numpy(idx).to(device)]
            yb = yt[torch.from_numpy(idx).to(device)]
            logits, _ = head(xb, yb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += float(loss); nb += 1
        hv = embed_head(head, Xval, device)
        m = rerank_eval(hv[:nq], hv[nq:], q_ids, g_ids, gt, qcsv, gcsv,
                        os.path.join(evaldir, f"ep{ep:02d}"), rr_best, f"ep{ep}")
        hist.append({"epoch": ep, "loss": run / max(1, nb), **m})
        print(f"[ep {ep:02d}] loss={run/max(1,nb):.4f} dino-head mAP@10={m['mAP@10']:.4f} "
              f"R1={m['Rank-1']:.4f}", flush=True)
        if (m["mAP@10"] or 0) > best_m:
            best_m = m["mAP@10"] or 0; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save({"state_dict": head.state_dict(), "args": vars(args),
                "in_dim": Xtr_n.shape[1], "emb_dim": args.emb_dim,
                "num_classes": C, "best_epoch": best_ep,
                "val_dino_mAP@10": best_m},
               os.path.join(out, "head_dino_best.pt"))
    best_hist = max(hist, key=lambda r: r["mAP@10"] or 0)
    print(f"[head-dino] best ep={best_ep}:",
          json.dumps(best_hist, ensure_ascii=False), flush=True)

    # ---- fusion of heads ----
    hv = l2(embed_head(head, Xval, device))
    sig = l2(np.load(os.path.join(args.val_sig_run, "embeddings.npy")).astype(np.float32))
    fus_rows = []
    wgrid = (0.6, 0.8, 1.0, 1.2, 1.5)
    for w in wgrid:
        fq = l2(np.concatenate([hv[:nq], w * sig[:nq]], axis=1))
        fg = l2(np.concatenate([hv[nq:], w * sig[nq:]], axis=1))
        m = rerank_eval(fq, fg, q_ids, g_ids, gt, qcsv, gcsv,
                        os.path.join(evaldir, "fus_head_dino_raw_sig"), rr_best,
                        f"fus_dinohead_sig{w}")
        fus_rows.append({"w_sig": w, "kind": "dino_head(+)raw_sig", **m})
        print(f"[fus] dino-head + {w}*raw-sig mAP@10={m['mAP@10']:.4f} "
              f"R1={m['Rank-1']:.4f}", flush=True)
    # fusion with the SigLIP2 head (exp-0047) if present
    sig_head_rows = []
    if os.path.exists(args.sig_head):
        from reid.models.head_frozen import Head as H2
        ck = torch.load(args.sig_head, map_location="cpu", weights_only=False)
        sh = H2(ck["in_dim"], ck["emb_dim"], ck["num_classes"],
                ck["args"]["margin"], ck["args"]["scale"]).to(device)
        sh.load_state_dict(ck["state_dict"]); sh.eval()
        sv = l2(embed_head(sh, l2(np.load(os.path.join(args.val_sig_run,
                     "embeddings.npy")).astype(np.float32)), device))
        for w in (0.8, 1.0, 1.2):
            fq = l2(np.concatenate([hv[:nq], w * sv[:nq]], axis=1))
            fg = l2(np.concatenate([hv[nq:], w * sv[nq:]], axis=1))
            m = rerank_eval(fq, fg, q_ids, g_ids, gt, qcsv, gcsv,
                            os.path.join(evaldir, "fus_heads"), rr_best,
                            f"fus_heads_w{w}")
            sig_head_rows.append({"w_sighead": w, "kind": "dino_head(+)sig_head", **m})
            print(f"[fus-heads] {w} mAP@10={m['mAP@10']:.4f} R1={m['Rank-1']:.4f}",
                  flush=True)

    report = {"agent": "ml-trainer", "task": "head-only frozen DINOv2 + fusion of heads",
              "config": vars(args), "num_classes": C, "n_train": len(tr),
              "raw_dino": m_raw, "history": hist, "best_dino_head": best_hist,
              "best_epoch": best_ep, "fusion_dino_head_raw_sig": fus_rows,
              "fusion_heads": sig_head_rows,
              "split": {"seed": 42, "n_query": nq, "n_gallery": len(vg)}}
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
