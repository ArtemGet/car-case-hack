"""Smoke test for tools/bench_perf.py in --dummy mode.

Synthetic 6 images + CSVs; runs a shortened protocol (small warmup/runs and
1-second throughput) on CPU and checks the report is written and metrics > 0.
"""
from __future__ import annotations

import importlib.util
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_PATH = os.path.join(ROOT, "tools", "bench_perf.py")


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_perf", BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_dataset(root, n=6):
    from PIL import Image

    images = os.path.join(root, "images")
    os.makedirs(images, exist_ok=True)
    ids = [f"img{i:02d}" for i in range(n)]
    for i, iid in enumerate(ids):
        arr = Image.new("RGB", (64 + i, 48 + i), (20 * i % 255, 90, 200 - i))
        arr.save(os.path.join(images, f"{iid}.jpg"), quality=90)

    def write_csv(path, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("image_id,x,y,w,h\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

    q = os.path.join(root, "test_query.csv")
    g = os.path.join(root, "test_gallery.csv")
    write_csv(q, [(ids[0], 0, 0, 40, 30), (ids[1], 2, 2, 40, 30)])
    write_csv(g, [(iid, 0, 0, 32, 24) for iid in ids])
    return images, q, g


def test_dummy_bench_writes_report(tmp_path):
    bench = _load_bench()
    images, q, g = _make_dataset(str(tmp_path), n=6)
    out = tmp_path / "bench_dummy.json"

    rc = bench.main([
        "--dummy",
        "--images", images,
        "--query", q,
        "--gallery", g,
        "--json", str(out),
        "--device", "cpu",
        "--warmup", "2",
        "--latency-runs", "5",
        "--throughput-seconds", "1",
        "--batches", "1,8",
    ])
    assert rc == 0
    assert out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))

    # 2 query rows + 6 gallery rows (same ids reused in this synthetic set)
    assert rep["num_images"] == 8
    assert rep["latency_b1_ms"] > 0
    assert rep["throughput_fps_best"] > 0
    assert rep["weights_mb"] > 0
    assert rep["backend"] == "torch"
    assert rep["model"] == "dummy:resnet18"
    assert len(rep["throughput"]["runs"]) == 2
    # median <= p90 ordering sanity
    assert rep["latency_b1"]["median_ms"] <= rep["latency_b1"]["p90_ms"]


def test_missing_images_returns_error(tmp_path):
    bench = _load_bench()
    images = os.path.join(str(tmp_path), "images")
    os.makedirs(images)
    q = tmp_path / "q.csv"
    q.write_text("image_id,x,y,w,h\nnone,0,0,10,10\n", encoding="utf-8")
    rc = bench.main([
        "--dummy", "--images", images, "--query", str(q),
        "--json", str(tmp_path / "r.json"), "--device", "cpu",
        "--warmup", "0", "--latency-runs", "1",
        "--throughput-seconds", "0.1", "--batches", "1",
    ])
    assert rc == 2
