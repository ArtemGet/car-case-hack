"""Zero-shot evaluation of EXTERNAL, publicly-released vehicle-ReID extractors.

These are models trained by others on vehicle ReID datasets (VeRi-776 / VERI-Wild
etc.) and released with permissive licences. We use them as frozen feature
extractors and score them through the OFFICIAL ``evaluate.py`` on our hold-out
val split — the same split / junk / open-set protocol as every in-house run, so
numbers are directly comparable with ``runs/exp-0007``.

Sources (url + sha256 recorded in runs/<exp>/external_sources.json):
  resnet34_veri   dgwon/resnet-34-veri776   resnet34_veri776_deploy.pt   (VeRi mAP .73)
  clipreid        occurra/vehicle_vit_clip_reid  vehicle_vit_clip_reid.onnx  (VeRi)
  siglip2         occurra/vehicle_reid_siglip2_naflex_512d  *.onnx

Red lines: camera_id is never a network input; no test data; per-query ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from reid.data.crop import crop_vehicle, open_cropped
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------
def _prep_square(dataset_dir, image_id, x, y, w, h, size):
    """Aspect-preserving square-pad crop (our pipeline contract)."""
    img = open_cropped(dataset_dir, image_id, x, y, w, h, target=size,
                       draft_factor=2.0)
    return img


def _prep_stretch(dataset_dir, image_id, x, y, w, h, size):
    """Plain bbox crop stretched to size x size (no pad)."""
    from reid.data.io import image_path
    with Image.open(image_path(dataset_dir, image_id)) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        x0 = min(max(int(round(x)), 0), ow - 1)
        y0 = min(max(int(round(y)), 0), oh - 1)
        x1 = min(max(int(round(x + w)), x0 + 1), ow)
        y1 = min(max(int(round(y + h)), y0 + 1), oh)
        crop = im.crop((x0, y0, x1, y1))
        return crop.resize((size, size), Image.BILINEAR)


def _prep_raw(dataset_dir, image_id, x, y, w, h):
    """Natural-aspect bbox crop, no resize (SigLIP2 NaFlex consumes patches)."""
    from reid.data.io import image_path
    im = Image.open(image_path(dataset_dir, image_id)).convert("RGB")
    ow, oh = im.size
    x0 = min(max(int(round(x)), 0), ow - 1)
    y0 = min(max(int(round(y)), 0), oh - 1)
    x1 = min(max(int(round(x + w)), x0 + 1), ow)
    y1 = min(max(int(round(y + h)), y0 + 1), oh)
    return im.crop((x0, y0, x1, y1))


def _to_chw(img, mean, std):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1).astype(np.float32))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class ResNet34Veri:
    """torchvision resnet34 + VeRi-776 classifier stripped (penultimate feats)."""

    name = "resnet34_veri"
    norm = "imagenet"
    prep = "square"
    default_size = 256

    def __init__(self, weights, device):
        from torchvision import models
        net = models.resnet34(weights=None)
        net.fc = nn.Identity()
        ck = torch.load(weights, map_location="cpu")
        sd = ck.get("state_dict", ck)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("fc.")]
        if missing:
            raise RuntimeError(f"resnet34 missing keys: {missing[:5]}")
        net.eval().to(device)
        self.net = net
        self.device = device
        self.dim = 512

    @torch.no_grad()
    def embed_batch(self, imgs, size):
        x = np.stack([_to_chw(im, IMAGENET_MEAN, IMAGENET_STD) for im in imgs])
        t = torch.from_numpy(x).to(self.device)
        feat = self.net(t)
        feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return feat.float().cpu().numpy().astype(np.float32)


def _add_ort_dll_dirs():
    """Put torch/lib and nvidia/*/lib on the DLL search path BEFORE ORT CUDA.

    On Windows the onnxruntime CUDA EP needs the CUDA/cuDNN DLLs that ship with
    the torch wheels; without this the provider fails to load and ORT silently
    falls back to CPUExecutionProvider. Returns the added directories.
    """
    import glob
    import site
    cands = []
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            cands.append(lib)
    except Exception:  # noqa: BLE001
        pass
    try:
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
        for sp in roots:
            cands += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    except Exception:  # noqa: BLE001
        pass
    added = []
    for d in cands:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                added.append(d)
            except Exception:  # noqa: BLE001
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return added


def cuda_providers(no_fallback: bool = False):
    """Return the ORT CUDA provider list, or raise if CUDA is unavailable.

    The NaFlex graph contains a handful of integer shape ops (Mod/Equal/Where)
    that no CUDA EP implements, so the CPU EP stays in the list as an explicit
    per-node fallback for those shape ops ONLY. All heavy compute (MatMul /
    LayerNorm / Softmax / attention) runs on CUDA. Callers MUST still assert
    ``session.get_providers()[0] == "CUDAExecutionProvider"`` — ORT reports the
    CPU EP first when it silently falls back, so this ordering is the check.
    ``no_fallback=True`` isolates the CUDA EP completely.
    """
    import onnxruntime as ort
    _add_ort_dll_dirs()
    avail = ort.get_available_providers()
    print(f"[ort] available providers: {avail}", flush=True)
    if "CUDAExecutionProvider" not in avail:
        raise RuntimeError(
            "CUDAExecutionProvider unavailable; CUDA is required for model "
            f"forward (providers={avail}). Refusing to fall back to CPU.")
    if no_fallback:
        base = [("CUDAExecutionProvider", {"device_id": 0})]
    else:
        base = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Optional env-gated arena cap; with no REID_ORT_* set this is a no-op.
    from reid.export.ort_env import apply_cuda_options
    return apply_cuda_options(base)


def require_cuda_session(sess, what="ORT session"):
    """Fail loudly unless the CUDA EP is the session's first/active provider."""
    got = list(sess.get_providers())
    print(f"[ort] {what} session providers={got}", flush=True)
    if not got or got[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"{what}: CUDAExecutionProvider is not the active provider "
            f"(got {got}); refusing silent CPU execution")
    return got


class OnnxBackend:
    """Generic ONNX Runtime backend. ``providers`` selects the ORT EPs.

    GPU / forward code MUST pass ``providers=cuda_providers()`` (or
    ``require_cuda=True``) so the forward never silently falls back to CPU.
    CPU is an explicit opt-in only, for the CPU-only post-processing grids.
    """

    def __init__(self, weights, input_name="input", providers=None,
                 strict_cuda: bool = False, require_cuda: bool = False):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0
        if providers is None:
            providers = cuda_providers() if require_cuda \
                else ["CPUExecutionProvider"]  # explicit CPU opt-in
        wants_cuda = any("CUDAExecutionProvider" in str(p) for p in providers)
        if wants_cuda:
            _add_ort_dll_dirs()
        if strict_cuda:
            # Forbid the implicit CPU fallback partition.
            try:
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            except Exception:  # noqa: BLE001 - older ORT
                pass
        self.sess = ort.InferenceSession(
            weights, sess_options=so, providers=providers)
        if wants_cuda:
            require_cuda_session(self.sess, what=os.path.basename(str(weights)))
        self.input_name = input_name
        # Honour a STATIC batch dim (0 = dynamic/unlimited). Some third-party
        # exports hard-code batch=1; the extraction loop must not exceed it.
        self.max_batch = 0
        try:
            shape = self.sess.get_inputs()[0].shape
            if isinstance(shape[0], int) and shape[0] > 0:
                self.max_batch = int(shape[0])
        except Exception:  # noqa: BLE001
            pass

    def _run(self, feed):
        return self.sess.run(None, feed)[0]


class ClipReID(OnnxBackend):
    name = "clipreid"
    default_size = 256

    def __init__(self, weights, norm="imagenet", require_cuda=False,
                 providers=None):
        super().__init__(weights, input_name="input", providers=providers,
                         require_cuda=require_cuda)
        self.norm = norm
        self.prep = "square"
        mean, std = (CLIP_MEAN, CLIP_STD) if norm == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
        self.mean, self.std = mean, std
        self.dim = 512

    def embed_batch(self, imgs, size):
        x = np.stack([_to_chw(im, self.mean, self.std) for im in imgs])
        out = self._run({self.input_name: x.astype(np.float32)})
        out = np.asarray(out, dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-12, None)


class SigLIP2NaFlex(OnnxBackend):
    """SigLIP2 NaFlex ViT: patch-16 tokenisation, natural aspect ratio.

    pixel_values (B, 256, 768): each row is a 16x16x3 patch flattened.
    pixel_attention_mask (B, 256) int64: 1=real patch.
    spatial_shapes (B, 2) int64: (rows, cols) per image.
    """

    name = "siglip2"
    default_size = 256          # = max patches
    prep = "raw"
    norm = "half"
    patch = 16

    def __init__(self, weights, max_patches=256, providers=None,
                 strict_cuda=False):
        super().__init__(weights, providers=providers, strict_cuda=strict_cuda)
        self.max_patches = int(max_patches)
        self.dim = 512

    def _patchify(self, img, rows, cols):
        w, h = img.size
        img = img.resize((cols * self.patch, rows * self.patch), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # mean/std 0.5
        p = self.patch
        # (rows, p, cols, p, 3) -> (rows*cols, p*p*3)
        patches = arr.reshape(rows, p, cols, p, 3).transpose(0, 2, 1, 3, 4)
        return patches.reshape(rows * cols, p * p * 3)

    def embed_batch(self, imgs, size):
        b = len(imgs)
        mp = self.max_patches if not size else min(self.max_patches, int(size))
        pv = np.zeros((b, self.max_patches, self.patch * self.patch * 3), np.float32)
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
        feed = {"pixel_values": pv, "pixel_attention_mask": mask,
                "spatial_shapes": shapes}
        out = np.asarray(self._run(feed), dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-12, None)


class FastReIDVeri:
    """fast-reid ResNet50-IBN (SBS, VeRi-776) -- reconstructed in fastreid_veri."""

    name = "fastreid_veri"
    norm = "imagenet"
    prep = "square"
    default_size = 256

    def __init__(self, weights, norm="imagenet"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU forward")
        from reid.models.fastreid_veri import FastReIDVeriExtractor
        self.device = torch.device("cuda")
        self.net = FastReIDVeriExtractor.from_checkpoint(weights, self.device)
        self.norm = norm
        self.bgr = norm == "bgr"     # fast-reid reads BGR (cv2) by default
        self.dim = 2048

    @torch.no_grad()
    def embed_batch(self, imgs, size):
        if self.bgr:
            x = np.stack([_to_chw(im, IMAGENET_MEAN, IMAGENET_STD)[::-1]
                          for im in imgs]).copy()
        else:
            x = np.stack([_to_chw(im, IMAGENET_MEAN, IMAGENET_STD) for im in imgs])
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        feat = torch.nn.functional.normalize(self.net(t).float(), dim=1)
        return feat.cpu().numpy().astype(np.float32)


def build_backend(kind, weights, norm):
    if kind == "fastreid_veri":
        return FastReIDVeri(weights, norm=norm)
    if kind == "resnet34_veri":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU forward")
        device = torch.device("cuda")
        return ResNet34Veri(weights, device)
    if kind == "clipreid":
        # GPU-only: assert the CUDA EP is the active provider.
        return ClipReID(weights, norm=norm, require_cuda=True)
    if kind == "siglip2":
        return SigLIP2NaFlex(weights, require_cuda=True)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract(backend, df, dataset_dir, size):
    prep = {"square": _prep_square, "stretch": _prep_stretch}.get(backend.prep)
    imgs = []
    out = []
    bs = getattr(backend, "max_batch", 0) or 32
    t0 = time.time()
    n = len(df)
    for i, r in enumerate(df.itertuples(index=False)):
        if backend.prep == "raw":
            img = _prep_raw(dataset_dir, r.image_id, r.x, r.y, r.w, r.h)
        else:
            img = prep(dataset_dir, r.image_id, r.x, r.y, r.w, r.h, size)
        imgs.append(img)
        if len(imgs) == bs or i == n - 1:
            out.append(backend.embed_batch(imgs, size))
            imgs = []
            if (i + 1) % 256 == 0 or i == n - 1:
                print(f"    {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out, axis=0).astype(np.float32)


def _write_submission(path, q_ids, g_ids, orders, top_k=10):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for i, qid in enumerate(q_ids):
            w.writerow([qid] + [g_ids[j] for j in orders[i, :top_k]])


def _write_candidates(path, q_ids, g_ids, orders, scores):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for i, qid in enumerate(q_ids):
            j = int(orders[i, 0])
            w.writerow([qid, g_ids[j], f"{float(scores[i, 0]):.6f}"])


def _write_gt(val_query, val_gallery, path):
    import pandas as pd
    q = val_query[["image_id", "vehicle_id", "camera_id"]].copy(); q["split"] = "query"
    g = val_gallery[["image_id", "vehicle_id", "camera_id"]].copy(); g["split"] = "gallery"
    pd.concat([q, g], ignore_index=True).to_csv(path, index=False)


def _flat(rep):
    r = rep.get("ranking", {}); fr = rep.get("full_ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"), "mAP_full": fr.get("mAP_full"),
            "mINP": fr.get("mINP"), "n_scored": r.get("n_scored"),
            "n_openset_excluded": r.get("n_openset_excluded")}


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="External vehicle-ReID zero-shot eval")
    ap.add_argument("--model", required=True,
                    choices=["resnet34_veri", "clipreid", "siglip2",
                             "fastreid_veri"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", default=os.environ.get("DATASET_DIR", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--norm", default="imagenet",
                    choices=["imagenet", "clip", "bgr"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--tta", default="", help="comma scales, e.g. 256,320")
    args = ap.parse_args(argv)
    if not args.dataset:
        print("ERROR: --dataset or DATASET_DIR required", file=sys.stderr)
        return 2

    torch.manual_seed(42); np.random.seed(42)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    backend = build_backend(args.model, args.weights, args.norm)
    size = int(args.size or backend.default_size)
    print(f"backend={backend.name} prep={backend.prep} size={size} dim={backend.dim}",
          flush=True)

    train_full = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, val_query, val_gallery = holdout_val(train_full, val_fraction=0.2,
                                            open_set_fraction=0.2, seed=42)
    val_query = val_query.reset_index(drop=True)
    val_gallery = val_gallery.reset_index(drop=True)
    print(f"val: query={len(val_query)} gallery={len(val_gallery)}", flush=True)

    t0 = time.time()
    q = extract(backend, val_query, args.dataset, size)
    g = extract(backend, val_gallery, args.dataset, size)
    print(f"extracted q={q.shape} g={g.shape} in {time.time() - t0:.0f}s", flush=True)

    q_ids = val_query["image_id"].tolist(); g_ids = val_gallery["image_id"].tolist()
    gt = os.path.join(out, "gt.csv"); _write_gt(val_query, val_gallery, gt)
    val_query.to_csv(os.path.join(out, "query.csv"), index=False)
    val_gallery.to_csv(os.path.join(out, "gallery.csv"), index=False)
    np.save(os.path.join(out, "raw_embeddings.npy"),
            np.vstack([q, g]).astype(np.float32))

    emb = os.path.join(out, "embeddings.npy"); sub = os.path.join(out, "submission.csv")
    cand = os.path.join(out, "candidates.csv")
    np.save(emb, np.vstack([q, g]).astype(np.float32))

    sims = q @ g.T
    orders = np.argsort(-sims, axis=1, kind="stable")
    _write_submission(sub, q_ids, g_ids, orders)
    _write_candidates(cand, q_ids, g_ids, orders, sims)
    rep = run_official(gt_csv=gt, submission=sub, candidates=cand, embeddings=emb,
                       query=os.path.join(out, "query.csv"),
                       gallery=os.path.join(out, "gallery.csv"),
                       json_out=os.path.join(out, "official_baseline.json"))
    baseline = _flat(rep)
    print("ZERO-SHOT baseline", json.dumps(baseline, ensure_ascii=False), flush=True)

    result = {"model": args.model, "weights": args.weights, "norm": args.norm,
              "size": size, "baseline": baseline}

    # ---- query-side TTA (multiple NaFlex patch budgets, no hflip) ----------
    q_tta = q
    if args.tta:
        from reid import rerank as rk
        scales = [int(s) for s in args.tta.split(",") if s.strip()]
        extra = {s: extract(backend, val_query, args.dataset, s) for s in scales}
        q_tta = np.stack([
            rk.fuse_embeddings([q[i]] + [extra[s][i] for s in scales])
            for i in range(len(q))]).astype(np.float32)
        sims_t = q_tta @ g.T
        ord_t = np.argsort(-sims_t, axis=1, kind="stable")
        _write_submission(sub, q_ids, g_ids, ord_t)
        _write_candidates(cand, q_ids, g_ids, ord_t, sims_t)
        np.save(emb, np.vstack([q_tta, g]).astype(np.float32))
        m_t = _flat(run_official(gt_csv=gt, submission=sub, candidates=cand, embeddings=emb,
                                 query=os.path.join(out, "query.csv"),
                                 gallery=os.path.join(out, "gallery.csv"),
                                 json_out=os.path.join(out, "official_tta_only.json")))
        print("TTA-only", json.dumps(m_t, ensure_ascii=False), flush=True)
        result["tta_scales"] = scales
        result["tta_only"] = m_t

    def rerank_best(qq, tag):
        from reid import rerank
        best = None
        for k1, k2, lam in [(5, 2, 0.7), (10, 2, 0.7), (5, 2, 0.5), (20, 3, 0.5)]:
            o = np.empty((len(qq), len(g)), np.int64)
            sc = np.empty((len(qq), len(g)), np.float32)
            for i in range(len(qq)):
                prep = rerank.prepare_query(qq[i], g, pool_size=100)
                o[i], sc[i] = rerank.rank_prepared(prep, k1=k1, k2=k2, lam=lam)
            _write_submission(sub, q_ids, g_ids, o)
            _write_candidates(cand, q_ids, g_ids, o, sc)
            np.save(emb, np.vstack([qq, g]).astype(np.float32))
            m = _flat(run_official(gt_csv=gt, submission=sub, candidates=cand, embeddings=emb,
                                   query=os.path.join(out, "query.csv"),
                                   gallery=os.path.join(out, "gallery.csv"),
                                   json_out=os.path.join(out, f"official_{tag}_{k1}_{k2}_{lam}.json")))
            print(f"  {tag} k1={k1} k2={k2} lam={lam}: mAP@10={m['mAP@10']:.4f} "
                  f"R1={m['Rank-1']:.4f}", flush=True)
            if best is None or (m["mAP@10"] or 0) > (best[1]["mAP@10"] or 0):
                best = ({"k1": k1, "k2": k2, "lam": lam, "tag": tag}, m)
        return best

    if args.rerank:
        best = rerank_best(q, "rr")
        result["rerank"] = best[0]; result["rerank_metrics"] = best[1]
        if args.tta:
            best_t = rerank_best(q_tta, "rrtta")
            result["rerank_tta"] = best_t[0]; result["rerank_tta_metrics"] = best_t[1]
            if (best_t[1]["mAP@10"] or 0) > (best[1]["mAP@10"] or 0):
                best = best_t
        # keep best submission on disk
        from reid import rerank
        o = np.empty((len(q), len(g)), np.int64); sc = np.empty((len(q), len(g)), np.float32)
        qq = q_tta if best[0].get("tag") == "rrtta" else q
        for i in range(len(qq)):
            prep = rerank.prepare_query(qq[i], g, pool_size=100)
            o[i], sc[i] = rerank.rank_prepared(prep, k1=best[0]["k1"], k2=best[0]["k2"],
                                               lam=best[0]["lam"])
        _write_submission(sub, q_ids, g_ids, o)
        _write_candidates(cand, q_ids, g_ids, o, sc)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"report -> {args.json}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
