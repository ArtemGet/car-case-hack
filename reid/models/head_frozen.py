"""Head-only training on FROZEN SigLIP2 NaFlex features (priority-1 lever).

The external SigLIP2 NaFlex vehicle-ReID backbone is used exactly as shipped
(public weights, no fine-tuning): we freeze it, extract the 512-d embedding for
every training image, then learn ONLY a small discriminative head
(``Linear -> BNNeck -> ArcFace``) on top with PK-sampling by ``vehicle_id``.
This is the classic "linear probe / head-only" transfer, which often beats
end-to-end fine-tuning of a large frozen backbone (state H-0011 confirmed the
latter regresses).

The head is trained on the TRAIN identities only; evaluation uses the frozen
hold-out val split (seed 42) and the OFFICIAL ``evaluate.py``. The trained head
is then re-fused with the DINOv2 champion features to test whether a sharper
SigLIP2 subspace lifts the deployed fusion.

Red lines: no test data; camera_id never enters the network; per-query ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official
from reid.models.head import ArcFace, BNNeck
from reid.models.zeroshot_eval import SigLIP2NaFlex, _prep_raw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class Head(nn.Module):
    def __init__(self, in_dim, emb_dim, num_classes, margin=0.3, scale=30.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, emb_dim)
        self.bnneck = BNNeck(emb_dim)
        self.arc = ArcFace(emb_dim, num_classes, margin=margin, scale=scale)

    def embed(self, x):
        return self.bnneck(self.fc(x))

    def forward(self, x, labels=None):
        e = self.embed(x)
        if labels is None:
            return e
        return self.arc(e, labels), e


def build_siglip(weights, require_cuda=True):
    """Build the SigLIP2 ONNX backend. CUDA is MANDATORY (project rule).

    Raises (never silently degrades to CPU) when the CUDA EP or torch CUDA is
    missing, so a mis-configured environment fails loudly.
    """
    if require_cuda:
        import onnxruntime as ort
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for SigLIP2 forward but torch.cuda "
                               "is unavailable; refusing CPU forward")
        from reid.models.zeroshot_eval import cuda_providers
        prov = cuda_providers(no_fallback=False)
        be = SigLIP2NaFlex(weights, providers=prov, strict_cuda=False)
        got = be.sess.get_providers()
        print(f"[extract] device={torch.cuda.get_device_name(0)} "
              f"ort_avail={ort.get_available_providers()} "
              f"session={got}", flush=True)
        if not got or got[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDAExecutionProvider not active (got {got}); "
                "refusing silent CPU forward")
        return be
    return SigLIP2NaFlex(weights)


def extract(backend, df, dataset_dir, bs=64):
    imgs, out = [], []
    n = len(df)
    for i, r in enumerate(df.itertuples(index=False)):
        imgs.append(_prep_raw(dataset_dir, r.image_id, r.x, r.y, r.w, r.h))
        if len(imgs) == bs or i == n - 1:
            out.append(backend.embed_batch(imgs, None))
            imgs = []
            if (i + 1) % 512 == 0 or i == n - 1:
                print(f"    {i + 1}/{n}", flush=True)
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
def pk_batches(y, P, K, steps_per_epoch, rng):
    classes = np.unique(y)
    by = {int(c): np.where(y == c)[0] for c in classes}
    for _ in range(steps_per_epoch):
        chosen = rng.choice(classes, min(P, len(classes)), replace=False)
        idx = []
        for c in chosen:
            pool = by[int(c)]
            idx.extend(rng.choice(pool, K, replace=len(pool) < K).tolist())
        rng.shuffle(idx)
        yield np.asarray(idx, dtype=np.int64)


def flat(rep):
    r = rep.get("ranking", {})
    fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"), "mINP": fr.get("mINP")}


def write_sub(path, q_ids, g_ids, orders, scores, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])
    with open(os.path.join(os.path.dirname(path), "candidates.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def rerank_eval(q, g, q_ids, g_ids, gt, qcsv, gcsv, evaldir, rr_cfg, tag):
    os.makedirs(evaldir, exist_ok=True)
    sub = os.path.join(evaldir, "submission.csv")
    emb = os.path.join(evaldir, "embeddings.npy")
    np.save(emb, np.vstack([q, g]).astype(np.float32))
    nq = q.shape[0]
    preps = [rerank.prepare_query(q[i], g, pool_size=rr_cfg["pool_size"])
             for i in range(nq)]
    o = np.empty((nq, len(g)), np.int64); sc = np.empty((nq, len(g)), np.float32)
    for i in range(nq):
        o[i], sc[i] = rerank.rank_prepared(preps[i], k1=rr_cfg["k1"],
                                           k2=rr_cfg["k2"], lam=rr_cfg["lam"])
    write_sub(sub, q_ids, g_ids, o, sc)
    m = flat(run_official(gt_csv=gt, submission=sub,
                          candidates=os.path.join(evaldir, "candidates.csv"),
                          embeddings=emb, query=qcsv, gallery=gcsv,
                          json_out=os.path.join(evaldir, f"official_{tag}.json")))
    return m


# ---------------------------------------------------------------------------
def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig-weights",
                    default="runs/external/vehicle_reid_siglip2_naflex_512d.onnx")
    ap.add_argument("--val-sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--val-dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--emb-dim", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(argv)
    if not args.dataset:
        import glob
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        if not cand:
            print("ERROR: --dataset required", file=sys.stderr); return 2
        args.dataset = os.path.dirname(cand[0])
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    print(f"[dataset] {args.dataset}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # CUDA is mandatory for the head-only run (project rule). Fail loudly.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for head-only run; refusing CPU")
    print(f"[cuda] {torch.cuda.get_device_name(0)}", flush=True)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    tr, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2,
                             seed=42)
    tr = tr.reset_index(drop=True); vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    print(f"[split] train={len(tr)} val_q={nq} val_g={len(vg)}", flush=True)

    # ---- frozen SigLIP2 train features (cached) ----
    cache = os.path.join(out, "sig_train.npy")
    if args.reuse_cache and os.path.exists(cache):
        Xtr = np.load(cache).astype(np.float32)
    else:
        be = build_siglip(args.sig_weights, require_cuda=True)
        Xtr = extract(be, tr, args.dataset)
        np.save(cache, Xtr)
    print(f"[feat] train X={Xtr.shape} cos_mean={float((l2(Xtr) @ l2(Xtr).T).mean()):.3f}",
          flush=True)

    Xtr = l2(Xtr)
    ytr = tr["vehicle_id"].to_numpy().astype(np.int64)
    classes = sorted(set(ytr.tolist()))
    cls_map = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([cls_map[c] for c in ytr], dtype=np.int64)
    C = len(classes)

    # ---- val features (deployment distribution) ----
    sig_val = l2(np.load(os.path.join(args.val_sig_run, "embeddings.npy")).astype(np.float32))
    dino_val = l2(np.load(os.path.join(args.val_dino_run, "embeddings.npy")).astype(np.float32))
    assert sig_val.shape[0] == dino_val.shape[0] == nq + len(vg)

    # gt/query/gallery files for official eval
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device)
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr_i).to(device)

    head = Head(Xtr.shape[1], args.emb_dim, C, args.margin, args.scale).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(Xtr) // (args.P * args.K))
    total = max(1, args.epochs * steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total,
                                                       eta_min=args.lr * 0.1)
    rng = np.random.default_rng(args.seed)

    evaldir = os.path.join(out, "eval")
    rr_cfg = {"k1": 8, "k2": 2, "lam": 0.6, "pool_size": 200}
    hist = []
    best_m = -1.0
    best_state = None
    best_ep = 0

    @torch.no_grad()
    def val_embed(model, x):
        model.eval()
        return F.normalize(model.embed(torch.from_numpy(x).to(device)), dim=1).cpu().numpy()

    for ep in range(1, args.epochs + 1):
        head.train(); run = 0.0; nb = 0
        for idx in pk_batches(ytr_i, args.P, args.K, steps, rng):
            xb = Xt[torch.from_numpy(idx).to(device)]
            yb = yt[torch.from_numpy(idx).to(device)]
            logits, emb = head(xb, yb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += float(loss); nb += 1
        # val: sig-only with rerank
        hv = val_embed(head, sig_val)
        q, g = hv[:nq], hv[nq:]
        m = rerank_eval(q, g, q_ids, g_ids, gt, qcsv, gcsv,
                        os.path.join(evaldir, f"ep{ep:02d}"), rr_cfg, f"ep{ep}")
        hist.append({"epoch": ep, "loss": run / max(1, nb), **m})
        print(f"[ep {ep:02d}] loss={run / max(1, nb):.4f} sig+rr "
              f"mAP@10={m['mAP@10']:.4f} R1={m['Rank-1']:.4f}", flush=True)
        torch.save({"state_dict": head.state_dict(), "args": vars(args),
                    "in_dim": Xtr.shape[1], "emb_dim": args.emb_dim,
                    "num_classes": C}, os.path.join(out, "head_last.pt"))
        if (m["mAP@10"] or 0) > best_m:
            best_m = m["mAP@10"] or 0
            best_ep = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}

    best_hist = max(hist, key=lambda r: r["mAP@10"] or 0)
    print(f"[head] best sig-only ep={best_ep}:",
          json.dumps(best_hist, ensure_ascii=False), flush=True)

    # ---- fuse the head-transformed SigLIP2 with DINOv2 + rerank ----
    if best_state is not None:
        head.load_state_dict(best_state)
    torch.save({"state_dict": head.state_dict(), "args": vars(args),
                "in_dim": Xtr.shape[1], "emb_dim": args.emb_dim,
                "num_classes": C, "best_epoch": best_ep,
                "val_sig_mAP@10": best_m}, os.path.join(out, "head_best.pt"))
    hv = val_embed(head, sig_val)
    fus_rows = []
    for w in (0.6, 0.8, 1.0):
        fq = l2(np.concatenate([hv[:nq], w * dino_val[:nq]], axis=1))
        fg = l2(np.concatenate([hv[nq:], w * dino_val[nq:]], axis=1))
        for rc in ({"k1": 8, "k2": 3, "lam": 0.5, "pool_size": 200},
                   {"k1": 6, "k2": 3, "lam": 0.6, "pool_size": 200}):
            m = rerank_eval(fq, fg, q_ids, g_ids, gt, qcsv, gcsv,
                            os.path.join(evaldir, "fusion"), rc,
                            f"hash_{w}_{rc['k1']}{rc['k2']}{rc['lam']}")
            row = {"w": w, **rc, **m}
            fus_rows.append(row)
            print(f"[fusion] w={w} {rc} mAP@10={m['mAP@10']:.4f} "
                  f"R1={m['Rank-1']:.4f}", flush=True)

    report = {"agent": "ml-trainer", "task": "head-only frozen SigLIP2",
              "config": vars(args), "num_classes": C, "n_train": len(tr),
              "history": hist, "best_sig_only": best_hist, "best_epoch": best_ep,
              "fusion": fus_rows,
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
