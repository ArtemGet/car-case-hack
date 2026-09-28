"""tools/build_gallery.py — build a static SigLIP2 **CPU** gallery for the API.

The API facade (``service/api``) searches a *pre-built* static gallery supplied
via ``REID_API_GALLERY`` (``.npy`` ``(Ng, D)`` float32) + ``REID_API_GALLERY_IDS``
(``_ids.json``). This tool builds exactly those two files for a real CPU demo
(SigLIP2, dim=512) without touching the GPU:

* reads a dataset directory (``images/`` + a CSV with ``image_id,x,y,w,h``);
* embeds every bbox crop through the torch-free CPU path
  :class:`service.infer.cpu_backend.SiglipCpuBackend` (ONNX Runtime
  ``CPUExecutionProvider`` only, PIL/NumPy pre-process);
* writes ``<out>.npy`` + ``<out>_ids.json`` via
  :func:`service.api.gallery.save_gallery`.

Example
-------
    python tools/build_gallery.py \\
        --dataset "docs/Датасет/dataset" --csv test_gallery.csv \\
        --out artifacts/prototype_gallery/gallery.npy \\
        --variant siglip --threads 8

Then run the real CPU demo::

    set REID_API_ENGINE=infer
    set REID_API_DEVICE=cpu
    set REID_API_GALLERY=artifacts/prototype_gallery/gallery.npy
    uvicorn service.api.app:app --host 127.0.0.1 --port 8000

Fully offline: weights are the public local ONNX inside the image; no network.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REPO = _ROOT
DEFAULT_SIGLIP_ONNX = os.path.join("artifacts", "siglip2_fp16.onnx")
DEFAULT_CSV = "test_gallery.csv"
VARIANTS = ("siglip",)


def _log(msg: str) -> None:
    print(msg, flush=True)


def resolve_images_dir(dataset: str) -> str:
    """Accept either the ``images/`` folder itself or a dataset root."""
    p = os.path.abspath(dataset)
    if not os.path.isdir(p):
        raise SystemExit(f"--dataset: каталог не найден: {p}")
    if any(f.lower().endswith(".jpg") for f in os.listdir(p)):
        return p
    nested = os.path.join(p, "images")
    if os.path.isdir(nested):
        return nested
    return p


def resolve_csv(dataset: str, csv: str) -> str:
    """Resolve ``--csv`` against ``--dataset`` when it is a relative name."""
    if os.path.isabs(csv) or os.path.exists(csv):
        return os.path.abspath(csv)
    cand = os.path.join(os.path.abspath(dataset), csv)
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"--csv: файл не найден: {csv!r} (и {cand})")


def _resolve_weights(explicit: str | None) -> str:
    if explicit:
        p = os.path.abspath(explicit)
    else:
        p = os.path.join(REPO, DEFAULT_SIGLIP_ONNX)
    if not os.path.exists(p):
        raise SystemExit(f"не найден SigLIP2 ONNX: {p}")
    return p


def _check_coverage(df, images_dir: str) -> None:
    ids = df["image_id"].astype(str).tolist()
    missing = [i for i in ids if not os.path.exists(
        os.path.join(images_dir, f"{i}.jpg"))]
    if missing:
        raise SystemExit(
            f"отсутствуют изображения ({len(missing)}): {missing[:5]}")
    if len(set(ids)) != len(ids):
        raise SystemExit("дубликаты image_id в CSV (галерея должна быть уникальной)")


def _self_check(emb: np.ndarray, ids: list[str], top_k: int = 10) -> dict:
    """Self-search sanity: each gallery vector must rank itself top-1."""
    from service.api.gallery import NumpyGalleryIndex

    n = emb.shape[0]
    if n == 0:
        return {"checked": 0, "top1_self": 0, "top1_self_rate": 0.0,
                "median_top1": 0.0}
    g = NumpyGalleryIndex(emb, ids)
    hit = 0
    top1s = np.empty(n, dtype=np.float32)
    for i in range(n):
        got, sc = g.search(emb[i], top_k=min(top_k, n))
        top1s[i] = float(sc[0]) if len(sc) else 0.0
        if got and got[0] == ids[i]:
            hit += 1
    return {
        "checked": int(n),
        "top1_self": int(hit),
        "top1_self_rate": hit / n,
        "median_top1": float(np.median(top1s)),
    }


def build_gallery(dataset: str, csv: str = DEFAULT_CSV, out: str = "",
                  *, limit: int = 0, variant: str = "siglip", threads: int = 8,
                  batch_size: int = 32, max_ram_mb: int = 0,
                  siglip_weights: str | None = None, seed: int = 42,
                  report_path: str | None = None) -> dict:
    """Build ``<out>.npy`` + ``<out>_ids.json`` for ``dataset``/``csv``.

    Returns a JSON-serialisable report (time, size, RSS, self-search rate).
    """
    if variant not in VARIANTS:
        raise SystemExit(
            f"build_gallery поддерживает только --variant siglip "
            f"(CPU-прототип); получено {variant!r}")

    from reid.data.io import QUERY_COLUMNS, read_csv

    images_dir = resolve_images_dir(dataset)
    csv_path = resolve_csv(dataset, csv)
    _log(f"[gallery] dataset={os.path.abspath(dataset)} images_dir={images_dir}")
    _log(f"[gallery] csv={csv_path}")

    df = read_csv(csv_path, required=QUERY_COLUMNS)
    if int(limit or 0) > 0:
        df = df.head(int(limit)).reset_index(drop=True)
        _log(f"[gallery] --limit={limit}: беру первые {len(df)} строк")
    _check_coverage(df, images_dir)
    n = len(df)
    if n == 0:
        raise SystemExit("галерея пуста: CSV без строк")
    _log(f"[gallery] rows={n} (unique ids, coverage OK)")

    # CPU-only path: torch-free, ORT CPUExecutionProvider, PIL pre-process.
    from service.infer.cpu_backend import (SiglipCpuBackend, rss_mb,
                                           set_cpu_determinism)
    set_cpu_determinism(seed)

    weights = _resolve_weights(siglip_weights)
    _log(f"[gallery] SigLIP2 CPU backend weights={weights} "
         f"threads={threads} batch={batch_size} rss={rss_mb():.0f}MB")
    t0 = time.time()
    be = SiglipCpuBackend(weights, threads=int(threads), max_ram_mb=max_ram_mb)
    emb = be.extract(df, images_dir, batch_size=int(batch_size))
    elapsed = time.time() - t0

    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if emb.shape != (n, be.dim):
        raise SystemExit(f"неверная форма эмбеддингов: {emb.shape} != {(n, be.dim)}")
    if not np.isfinite(emb).all():
        raise SystemExit("эмбеддинги содержат NaN/Inf")
    norms = np.linalg.norm(emb, axis=1)
    ids = df["image_id"].astype(str).tolist()

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    from service.api.gallery import save_gallery
    npy_path, ids_path = save_gallery(out, ids, emb)
    _log(f"[gallery] saved {npy_path} ({emb.shape}) + {ids_path}")
    _log(f"[gallery] extract {elapsed:.1f}s "
         f"({elapsed / n * 1000.0:.0f} ms/frame), rss={rss_mb():.0f}MB")

    sc = _self_check(emb, ids)
    _log(f"[gallery] self-search top-1 == own id: "
         f"{sc['top1_self']}/{sc['checked']} ({sc['top1_self_rate'] * 100:.1f}%) "
         f"median_top1_cos={sc['median_top1']:.4f}")

    report = {
        "variant": variant,
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "weights": weights,
        "dataset": os.path.abspath(dataset),
        "images_dir": images_dir,
        "csv": csv_path,
        "n_gallery": int(n),
        "dim": int(emb.shape[1]),
        "threads": int(threads),
        "batch_size": int(batch_size),
        "limit": int(limit or 0),
        "time_s": elapsed,
        "ms_per_frame": elapsed / n * 1000.0,
        "rss_after_mb": float(rss_mb()),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "gallery_npy": os.path.abspath(npy_path),
        "gallery_npy_bytes": os.path.getsize(npy_path),
        "gallery_ids": os.path.abspath(ids_path),
        "gallery_ids_bytes": os.path.getsize(ids_path),
        "self_search": sc,
    }
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _log(f"[gallery] report -> {os.path.abspath(report_path)}")
    return report


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Сборка статической CPU-галереи (SigLIP2) для API-демо")
    ap.add_argument("--dataset", required=True,
                    help="каталог датасета (с images/ или самим каталогом jpg)")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="CSV с image_id,x,y,w,h (по умолчанию test_gallery.csv; "
                         "относительный путь — от --dataset)")
    ap.add_argument("--out", required=True,
                    help="путь к gallery.npy (рядом будет gallery_ids.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="взять первые N строк (0 = все; для быстрой пробы)")
    ap.add_argument("--variant", default="siglip", choices=list(VARIANTS),
                    help="только siglip (CPU-прототип)")
    ap.add_argument("--threads", type=int, default=8,
                    help="intra-op потоки ONNX Runtime (CPU)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-ram-mb", type=int, default=0,
                    help="лимит RSS (0 = без лимита)")
    ap.add_argument("--siglip-weights", default=None,
                    help="SigLIP2 ONNX (по умолчанию artifacts/siglip2_fp16.onnx)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None,
                    help="путь для JSON-отчёта сборки (опционально)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    report_path = args.report
    if report_path is None:
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "gallery_report.json")
    report = build_gallery(
        args.dataset, args.csv, args.out,
        limit=args.limit, variant=args.variant, threads=args.threads,
        batch_size=args.batch_size, max_ram_mb=args.max_ram_mb,
        siglip_weights=args.siglip_weights, seed=args.seed,
        report_path=report_path)
    _log(f"[done] gallery={report['n_gallery']} dim={report['dim']} "
         f"{report['time_s']:.1f}s -> {report['gallery_npy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
