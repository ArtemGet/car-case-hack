"""EX-3 · auxiliary ATTRIBUTE heads (colour / body-type) on frozen ReID features.

Goal (ТЗ п.4, 07_EXPERIMENT_PLAN.md EX-3): do cheap auxiliary attribute
classifiers on top of the FROZEN champion features (SigLIP2 NaFlex 512d,
DINOv2-B 512d) improve ReID ranking when fused with the champion embedding?

Pipeline
--------
1. Pseudo-labels for TRAIN crops from a deterministic, self-contained heuristic
   (no network / no auth): crop -> HSV -> colour family; body-type from the bbox
   aspect ratio. Source + sha256 of this heuristic are recorded.
2. Train tiny MLP heads (512 -> 256 -> K) on the FROZEN train features
   (CPU, 8 epochs); the backbone is never touched.
3. Val: apply heads to the frozen val features -> attribute probability vectors.
4. Fusion: ``L2(concat[ L2(sig), 0.8*L2(dino), a*L2(attr) ])`` + k-reciprocal
   pool=300 (champion post-proc) -> OFFICIAL evaluate.py.

Red lines: camera_id never enters the network; heads train on the TRAIN split
only; val is untouched for fitting.

    python -m reid.models.attr_head --out runs/exp-0077-attr --json reports/exp-0077-attr.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from reid.data.io import TRAIN_COLUMNS, image_path, read_csv
from reid.data.splits import holdout_val
from reid.models.dba_postproc import eval_dba
from reid.models.fusion_postproc import build_post

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1 · deterministic colour / type pseudo-labels
# ---------------------------------------------------------------------------
COLORS = ["black", "white", "silver", "red", "orange", "yellow", "green",
          "blue", "brown"]
TYPES = ["car", "tall", "long"]          # aspect-ratio bins (bbox only)

_THIS_FILE = os.path.abspath(__file__)


def heuristic_source() -> dict:
    with open(_THIS_FILE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "kind": "deterministic crop heuristic (no network, no auth)",
        "code": "reid/models/attr_head.py",
        "sha256": sha,
        "color_rule": "central 60% crop -> 48x48 -> gray-world WB on bright 60%; "
                      "HSV mask S>0.30 & 0.15<V<0.95; modal hue family (10deg "
                      "bins); achromatic by median V; brown = hue 10-45 && V<0.45",
        "type_rule": "bbox aspect w/h: <0.75 tall, >1.8 long, else car",
    }


def _rgb_to_hsv(a):
    """a: (N,3) in [0,1] -> h(deg), s, v."""
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    mx = a.max(1)
    mn = a.min(1)
    d = mx - mn
    s = np.where(mx > 0, d / np.clip(mx, 1e-6, None), 0.0)
    h = np.zeros_like(mx)
    nz = d > 1e-6
    idx = nz & (mx == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / d[idx]) + 360) % 360
    idx = nz & (mx == g)
    h[idx] = 60 * ((b[idx] - r[idx]) / d[idx]) + 120
    idx = nz & (mx == b)
    h[idx] = 60 * ((r[idx] - g[idx]) / d[idx]) + 240
    return h, s, mx


def _family_from_hue(hue, sat, val):
    if sat < 0.15:
        if val < 0.22:
            return 0   # black
        if val > 0.78:
            return 1   # white
        return 2       # silver
    if 10 <= hue < 45 and val < 0.45 and sat > 0.25:
        return 8       # brown
    if hue < 15 or hue >= 345:
        return 3
    if hue < 45:
        return 4
    if hue < 70:
        return 5
    if hue < 165:
        return 6
    if hue < 260:
        return 7
    return 3           # purple/magenta -> red


def color_label_crop(im, x, y, w, h):
    ow, oh = im.size
    x0 = min(max(int(round(x)), 0), ow - 1)
    y0 = min(max(int(round(y)), 0), oh - 1)
    x1 = min(max(int(round(x + w)), x0 + 1), ow)
    y1 = min(max(int(round(y + h)), y0 + 1), oh)
    c = im.crop((x0, y0, x1, y1)).convert("RGB")
    cw, ch = c.size
    ix, iy = int(0.2 * cw), int(0.2 * ch)
    if cw - 2 * ix < 4 or ch - 2 * iy < 4:
        ix = iy = 0
    c = c.crop((ix, iy, cw - ix, ch - iy)).resize((48, 48), Image.BILINEAR)
    a = np.asarray(c, dtype=np.float32).reshape(-1, 3) / 255.0
    # gray-world white balance on the brighter half (kills camera colour cast)
    vmax = a.max(1)
    sel = vmax > np.percentile(vmax, 60)
    scale = a[sel].mean(0) if sel.any() else np.ones(3, np.float32)
    scale = scale / max(float(scale.mean()), 1e-6)
    a = np.clip(a / scale, 0.0, 1.0)
    hue, sat, val = _rgb_to_hsv(a)
    keep = (sat > 0.30) & (val > 0.15) & (val < 0.95)
    if keep.sum() < 10:
        v = float(np.median(val))
        return 0 if v < 0.22 else (1 if v > 0.78 else 2)
    hb = (hue[keep] / 10.0).astype(np.int64) % 36
    counts = np.bincount(hb, minlength=36)
    mode = int(np.argmax(counts)) * 10 + 5
    m = keep & (np.abs(((hue - mode + 180) % 360) - 180) < 10)
    return _family_from_hue(mode, float(sat[m].mean()), float(val[m].mean()))


def type_label(w, h):
    ar = float(w) / max(1.0, float(h))
    if ar < 0.75:
        return 1
    if ar > 1.8:
        return 2
    return 0


def pseudo_labels(df, dataset_dir):
    cols = np.zeros(len(df), np.int64)
    typs = np.zeros(len(df), np.int64)
    for i, r in enumerate(df.itertuples(index=False)):
        with Image.open(image_path(dataset_dir, r.image_id)) as im:
            cols[i] = color_label_crop(im, r.x, r.y, r.w, r.h)
        typs[i] = type_label(r.w, r.h)
    return cols, typs


# ---------------------------------------------------------------------------
# 2 · tiny head on FROZEN features
# ---------------------------------------------------------------------------
class AttrHead(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def train_head(X, y, n_cls, epochs, seed, hidden=256, lr=1e-3, bs=256,
               weighted=True):
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    Xt = torch.from_numpy(l2(X)).to(dev)
    yt = torch.from_numpy(y).to(dev)
    m = AttrHead(X.shape[1], hidden, n_cls).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y, minlength=n_cls).astype(np.float32)
    w = (torch.from_numpy((counts.sum() / np.clip(counts, 1, None)).astype(np.float32))
         if weighted else None)
    rng = np.random.default_rng(seed)
    n = len(X)
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(n)
        tot = nb = 0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            xb = Xt[torch.from_numpy(j)]
            yb = yt[torch.from_numpy(j)]
            loss = F.cross_entropy(m(xb), yb, weight=w, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        m.eval()
        with torch.no_grad():
            acc = float((m(Xt).argmax(1) == yt).float().mean())
        print(f"    head ep{ep + 1}/{epochs} loss={tot / max(1, nb):.4f} acc={acc:.3f}",
              flush=True)
    m.eval()
    return m


@torch.no_grad()
def head_probs(m, X):
    return m(torch.from_numpy(l2(X))).softmax(1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--sig-train", default="runs/exp-0047-head/sig_train.npy")
    ap.add_argument("--dino-train", default="runs/exp-0048-dinohead/dino_train.npy")
    ap.add_argument("--sig-run", default="runs/W2-5-val-siglip")
    ap.add_argument("--dino-run", default="runs/W2-5-val-dino")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--w", type=float, default=0.8)
    ap.add_argument("--pool", type=int, default=300)
    ap.add_argument("--k1", type=int, default=8)
    ap.add_argument("--k2", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.5,0.8")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not args.dataset:
        cand = glob.glob(os.path.join(REPO, "docs", "*", "dataset", "train.csv"))
        args.dataset = os.path.dirname(cand[0])

    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    print(f"[dataset] {args.dataset}", flush=True)

    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    tr, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    tr = tr.reset_index(drop=True); vq = vq.reset_index(drop=True)
    vg = vg.reset_index(drop=True)
    nq = len(vq)
    print(f"[split] train={len(tr)} val_q={nq} val_g={len(vg)}", flush=True)

    # ---- labels (cached) ----
    lab_path = os.path.join(out, "attr_labels.npz")
    if os.path.exists(lab_path):
        z = np.load(lab_path)
        ctr, ttr = z["ctr"], z["ttr"]; cvq, tvq = z["cvq"], z["tvq"]; cvg, tvg = z["cvg"], z["tvg"]
    else:
        ctr, ttr = pseudo_labels(tr, args.dataset)
        cvq, tvq = pseudo_labels(vq, args.dataset)
        cvg, tvg = pseudo_labels(vg, args.dataset)
        np.savez(lab_path, ctr=ctr, ttr=ttr, cvq=cvq, tvq=tvq, cvg=cvg, tvg=tvg)
    print("[labels] color train", np.bincount(ctr, minlength=len(COLORS)).tolist(),
          flush=True)
    print("[labels] type  train", np.bincount(ttr, minlength=len(TYPES)).tolist(),
          flush=True)

    # ---- frozen features ----
    Xs = np.load(args.sig_train).astype(np.float32)
    Xd = np.load(args.dino_train).astype(np.float32)
    assert Xs.shape[0] == Xd.shape[0] == len(tr), (Xs.shape, Xd.shape, len(tr))
    sv = np.load(os.path.join(args.sig_run, "embeddings.npy")).astype(np.float32)
    dv = np.load(os.path.join(args.dino_run, "embeddings.npy")).astype(np.float32)
    assert sv.shape[0] == dv.shape[0] == nq + len(vg)
    Xtr_cat = np.concatenate([l2(Xs), l2(Xd)], axis=1)
    Xval_cat = np.concatenate([l2(sv), l2(dv)], axis=1)

    # ---- train heads (colour: all 3 banks; type: concat only) ----
    heads = {}
    heads["color_sig"] = train_head(Xs, ctr, len(COLORS), args.epochs, args.seed)
    heads["color_dino"] = train_head(Xd, ctr, len(COLORS), args.epochs, args.seed)
    heads["color_cat"] = train_head(Xtr_cat, ctr, len(COLORS), args.epochs, args.seed)
    heads["type_cat"] = train_head(Xtr_cat, ttr, len(TYPES), args.epochs,
                                   args.seed, weighted=False)

    # val attribute vectors
    a_color_cat = head_probs(heads["color_cat"], Xval_cat)
    a_color_sig = head_probs(heads["color_sig"], l2(sv))
    a_color_dino = head_probs(heads["color_dino"], l2(dv))
    a_type_cat = head_probs(heads["type_cat"], Xval_cat)
    print("[heads] color val acc (cat) =",
          float((a_color_cat.argmax(1)[:nq] == cvq).mean()), flush=True)

    # ---- champion baseline + fusion ----
    bq, bg = build_post(sv, dv, nq, {"w": args.w})
    rr_cfg = {"k1": args.k1, "k2": args.k2, "lam": args.lam, "pool_size": args.pool}
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    ids = (vq["image_id"].tolist(), vg["image_id"].tolist(), gt, qcsv, gcsv)

    rows = []
    base = eval_dba(bq, bg, ids, os.path.join(out, "eval", "baseline"), rr_cfg,
                    "champion baseline")
    rows.append({"variant": "champion baseline", **base})
    print(f"[baseline] mAP@10={base['mAP@10']:.5f} R1={base['Rank-1']:.4f}", flush=True)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    def fuse_and_eval(attr, tag, alpha):
        aq, ag = l2(attr[:nq]), l2(attr[nq:])
        fq = l2(np.concatenate([bq, alpha * aq], axis=1))
        fg = l2(np.concatenate([bg, alpha * ag], axis=1))
        r = eval_dba(fq, fg, ids, os.path.join(out, "eval", tag.replace(" ", "_")),
                     rr_cfg, tag)
        return {"variant": tag, "alpha": alpha, **r}

    # colour only (concat head) grid
    for al in alphas:
        rows.append(fuse_and_eval(np.concatenate([a_color_cat], axis=1),
                                  f"color(a={al})", al))
    # colour (sig+dino separate heads) at the best small weight
    for al in (0.2,):
        rows.append(fuse_and_eval(np.concatenate([a_color_sig, a_color_dino], axis=1),
                                  f"color2(a={al})", al))
    # colour + type
    for al in (0.2, 0.5):
        rows.append(fuse_and_eval(np.concatenate([a_color_cat, a_type_cat], axis=1),
                                  f"color+type(a={al})", al))
    # direct heuristic one-hot (no learned head)
    oh = np.zeros((nq + len(vg), len(COLORS)), np.float32)
    oh[np.arange(nq), cvq] = 1.0
    oh[nq + np.arange(len(vg)), cvg] = 1.0
    for al in (1.0,):
        rows.append(fuse_and_eval(oh, f"heuristic-onehot(a={al})", al))

    best = max(rows, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps({k: best[k] for k in ("variant", "mAP@10", "Rank-1")}),
          flush=True)
    delta = (best["mAP@10"] or 0) - (base["mAP@10"] or 0)
    report = {
        "agent": "ml-trainer", "task": "EX-3 attribute heads (colour/body-type)",
        "config": vars(args), "source": heuristic_source(),
        "n_train": len(tr), "n_query": nq, "n_gallery": len(vg),
        "label_dist": {"color": np.bincount(ctr, minlength=len(COLORS)).tolist(),
                       "type": np.bincount(ttr, minlength=len(TYPES)).tolist()},
        "baseline": base, "results": rows, "best": best, "delta_vs_champion": delta,
        "verdict": "accepted" if delta > 1e-4 else "rejected",
    }
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    # save head weights (tiny)
    torch.save({k: v.state_dict() for k, v in heads.items()},
               os.path.join(out, "attr_heads.pt"))
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
