"""Build ``artifacts/data_manifest.json`` - a reproducibility manifest.

Records the sha256 of every dataset CSV, the image count, and a digest over the
sorted ``(filename, size)`` list of the images directory, plus the dataset dir.

Usage:
    python tools/make_manifest.py [--dataset-dir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DATASET_DIR = os.path.join(ROOT, "docs", "Датасет", "dataset")
DEFAULT_OUT = os.path.join(ROOT, "artifacts", "data_manifest.json")

CSV_NAMES = ["train.csv", "test_query.csv", "test_gallery.csv"]


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def image_pairs(images_dir):
    pairs = []
    for name in os.listdir(images_dir):
        full = os.path.join(images_dir, name)
        if os.path.isfile(full):
            pairs.append((name, os.path.getsize(full)))
    pairs.sort(key=lambda p: p[0])
    return pairs


def build_manifest(dataset_dir):
    csv_sha = {}
    for name in CSV_NAMES:
        path = os.path.join(dataset_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        csv_sha[name] = sha256_file(path)

    images_dir = os.path.join(dataset_dir, "images")
    pairs = image_pairs(images_dir)

    digest = hashlib.sha256()
    total_bytes = 0
    for name, size in pairs:
        digest.update(f"{name}\t{size}\n".encode("utf-8"))
        total_bytes += size

    return {
        "dataset_dir": os.path.abspath(dataset_dir),
        "csv_sha256": csv_sha,
        "n_images": len(pairs),
        "images_total_bytes": total_bytes,
        "images_list_digest_sha256": digest.hexdigest(),
        "images_list_digest_algo": "sha256 over sorted '<filename>\\t<size>\\n' lines",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    manifest = build_manifest(args.dataset_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nwritten: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
