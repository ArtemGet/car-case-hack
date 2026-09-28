"""Статическая галерея: векторный поиск с фиксированным интерфейсом.

Контракт (:class:`GalleryIndex`) не зависит от бэкенда. Доступны две
реализации:

* :class:`NumpyGalleryIndex` — in-memory cosine top-K на numpy (детерминирован,
  стабильный тай-брейк по порядку галереи, как требует ``evaluate.py``);
* :class:`FaissGalleryIndex` — тот же интерфейс поверх ``faiss.IndexFlatIP``
  (если ``faiss`` установлен; иначе фабрика молча выбирает numpy).

Галерея **статична** и строится заранее: эмбеддинги ``(Ng, D)`` + список id.
Эмбеддинги здесь не вычисляются (это делает inference-сервис) — индекс только
хранит и ищет, т.е. логика эмбеддинга не дублируется.

Файлы на диске: ``<name>.npy`` (float32 (Ng, D)) и ``<name>_ids.json``:
``{"ids": [...], "dim": D}`` (либо просто JSON-список id).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

try:  # optional acceleration; отсутствует — не ошибка
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _HAS_FAISS = False


def l2norm(x: np.ndarray, axis: int | None = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-12, None)


@runtime_checkable
class GalleryIndex(Protocol):
    """Фиксированный интерфейс поиска по статической галерее."""

    dim: int
    ids: list[str]
    backend: str

    def search(self, query: np.ndarray, top_k: int = 10) -> tuple[list[str], np.ndarray]:
        """Возвращает ``(gallery_ids, scores)`` top-K по убыванию близости."""
        ...


class NumpyGalleryIndex:
    """In-memory cosine top-K на L2-нормированных векторах."""

    backend = "numpy"

    def __init__(self, embeddings: np.ndarray, ids: Sequence[str]):
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"gallery embeddings must be 2-D, got {emb.shape}")
        ids = [str(i) for i in ids]
        if len(ids) != emb.shape[0]:
            raise ValueError(
                f"gallery ids ({len(ids)}) != embeddings rows ({emb.shape[0]})")
        if len(set(ids)) != len(ids):
            raise ValueError("gallery ids must be unique")
        self.embeddings = np.ascontiguousarray(l2norm(emb, axis=1))
        self.ids = ids
        self.dim = int(emb.shape[1])

    def search(self, query: np.ndarray, top_k: int = 10) -> tuple[list[str], np.ndarray]:
        q = l2norm(np.asarray(query, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != gallery dim {self.dim}")
        if not len(self.ids):
            return [], np.zeros((0,), dtype=np.float32)
        sims = self.embeddings @ q
        k = min(int(top_k), len(self.ids))
        # kind="stable": ничьи разрешаются порядком галереи (контракт evaluate.py)
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.ids[i] for i in order], sims[order].astype(np.float32)


class FaissGalleryIndex:
    """Тот же интерфейс поверх ``faiss.IndexFlatIP`` (cosine = IP на L2)."""

    backend = "faiss"

    def __init__(self, embeddings: np.ndarray, ids: Sequence[str]):
        if not _HAS_FAISS:
            raise RuntimeError("faiss недоступен")
        emb = np.asarray(embeddings, dtype=np.float32)
        ids = [str(i) for i in ids]
        if len(ids) != emb.shape[0]:
            raise ValueError(
                f"gallery ids ({len(ids)}) != embeddings rows ({emb.shape[0]})")
        self._emb = np.ascontiguousarray(l2norm(emb, axis=1))
        self.ids = ids
        self.dim = int(emb.shape[1])
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(self._emb)

    def search(self, query: np.ndarray, top_k: int = 10) -> tuple[list[str], np.ndarray]:
        q = l2norm(np.asarray(query, dtype=np.float32).reshape(-1))
        k = min(int(top_k), len(self.ids))
        if k == 0:
            return [], np.zeros((0,), dtype=np.float32)
        scores, idx = self._index.search(q.reshape(1, -1), k)
        idx = idx[0].tolist()
        return [self.ids[i] for i in idx if i >= 0], scores[0].astype(np.float32)


def build_gallery(embeddings: np.ndarray, ids: Sequence[str],
                  backend: str = "auto") -> GalleryIndex:
    """Фабрика: ``auto`` -> faiss если доступен, иначе numpy."""
    if backend == "auto":
        backend = "faiss" if _HAS_FAISS else "numpy"
    if backend == "faiss":
        if _HAS_FAISS:
            return FaissGalleryIndex(embeddings, ids)
        backend = "numpy"
    if backend == "numpy":
        return NumpyGalleryIndex(embeddings, ids)
    raise ValueError(f"unknown gallery backend: {backend!r}")


# ---------------------------------------------------------------------------
# Persistence: заранее построенная статическая галерея
# ---------------------------------------------------------------------------
def ids_path_for(embeddings_path: str) -> str:
    base, _ = os.path.splitext(embeddings_path)
    return base + "_ids.json"


def load_ids(ids_path: str) -> list[str]:
    with open(ids_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("ids", [])
    return [str(i) for i in data]


def load_gallery(embeddings_path: str, ids_path: str | None = None,
                 backend: str = "auto") -> GalleryIndex:
    """Загрузить готовую галерею с диска (``.npy`` + ``_ids.json``)."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(embeddings_path)
    emb = np.load(embeddings_path)
    ip = ids_path or ids_path_for(embeddings_path)
    if not os.path.exists(ip):
        raise FileNotFoundError(ip)
    return build_gallery(emb, load_ids(ip), backend=backend)


def save_gallery(embeddings_path: str, ids: Sequence[str], embeddings: np.ndarray,
                 ids_path: str | None = None) -> tuple[str, str]:
    """Сохранить галерею: ``<name>.npy`` + ``<name>_ids.json``."""
    os.makedirs(os.path.dirname(os.path.abspath(embeddings_path)), exist_ok=True)
    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    ip = ids_path or ids_path_for(embeddings_path)
    with open(ip, "w", encoding="utf-8") as f:
        json.dump({"ids": [str(i) for i in ids],
                   "dim": int(np.asarray(embeddings).shape[1])},
                  f, ensure_ascii=False)
    return embeddings_path, ip


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Инфо по статической галерее")
    ap.add_argument("--embeddings", required=True, help="<name>.npy (Ng, D)")
    ap.add_argument("--ids", default=None, help="<name>_ids.json (по умолчанию рядом)")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "faiss"])
    args = ap.parse_args(argv)
    g = load_gallery(args.embeddings, args.ids, backend=args.backend)
    print(f"gallery backend={g.backend} size={len(g.ids)} dim={g.dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GalleryIndex",
    "NumpyGalleryIndex",
    "FaissGalleryIndex",
    "build_gallery",
    "load_gallery",
    "save_gallery",
    "ids_path_for",
    "load_ids",
]
