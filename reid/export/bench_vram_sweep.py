#!/usr/bin/env python
"""Sweep ORT CUDA arena caps on the frozen fusion (224-only) deploy config.

Runs ``tools/bench_perf.py`` (organisers' full protocol) once per config, each in
a fresh subprocess so the CUDA context / ORT arena never carry over between
measurements.  Only environment variables change; the deploy default (no
``REID_ORT_*`` set) is one of the configs and is expected to reproduce the
~15.6 GB arena peak.

Usage (under the gpu lock)::

    python docs\\_workspace\\tools\\gpu_lock.py run --wait 3600 -- ^
        python -m reid.export.bench_vram_sweep

Writes ``reports/opt_vram_cap_raw.json`` (per-config bench report + status).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "tools", "bench_perf.py")
ART = os.path.join(REPO, "artifacts")
DINO = os.path.join(ART, "dinov2_b_fp16.onnx")
SIGLIP = os.path.join(ART, "siglip2_fp16.onnx")

# Config matrix: (name, env overrides). Unlisted REID_ORT_* are cleared.
CONFIGS = [
    ("baseline_noenv", {}),
    ("16384_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "16384", "REID_ORT_ARENA": "same"}),
    ("12288_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "12288", "REID_ORT_ARENA": "same"}),
    ("10240_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "10240", "REID_ORT_ARENA": "same"}),
    ("8192_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "8192", "REID_ORT_ARENA": "same"}),
    ("6144_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "6144", "REID_ORT_ARENA": "same"}),
    ("6656_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "6656", "REID_ORT_ARENA": "same"}),
    ("6784_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "6784", "REID_ORT_ARENA": "same"}),
    ("6912_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "6912", "REID_ORT_ARENA": "same"}),
    ("7040_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "7040", "REID_ORT_ARENA": "same"}),
    ("7168_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "7168", "REID_ORT_ARENA": "same"}),
    ("7680_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "7680", "REID_ORT_ARENA": "same"}),
    ("4096_same", {"REID_ORT_GPU_MEM_LIMIT_MB": "4096", "REID_ORT_ARENA": "same"}),
    ("12288_same_heur", {"REID_ORT_GPU_MEM_LIMIT_MB": "12288",
                         "REID_ORT_ARENA": "same", "REID_ORT_CUDNN_HEURISTIC": "1"}),
    ("8192_same_heur", {"REID_ORT_GPU_MEM_LIMIT_MB": "8192",
                        "REID_ORT_ARENA": "same", "REID_ORT_CUDNN_HEURISTIC": "1"}),
    ("8192_off", {"REID_ORT_GPU_MEM_LIMIT_MB": "8192", "REID_ORT_ARENA": "off"}),
    ("same_noenv", {"REID_ORT_ARENA": "same"}),
    ("8192_default_arena", {"REID_ORT_GPU_MEM_LIMIT_MB": "8192"}),
    ("12288_default_arena", {"REID_ORT_GPU_MEM_LIMIT_MB": "12288"}),
]


def _find_dataset() -> str:
    docs = os.path.join(REPO, "docs")
    for name in sorted(os.listdir(docs)):
        ds = os.path.join(docs, name, "dataset")
        if os.path.isfile(os.path.join(ds, "train.csv")) and \
                os.path.isdir(os.path.join(ds, "images")):
            return ds
    raise SystemExit("dataset dir with train.csv+images not found under docs/")


def main() -> int:
    dataset = os.environ.get("REID_VRAM_DATASET") or _find_dataset()
    images = os.path.join(dataset, "images")
    query = os.path.join(dataset, "test_query.csv")
    gallery = os.path.join(dataset, "test_gallery.csv")
    out_dir = os.path.join(REPO, "reports")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    only = set(sys.argv[1:])
    configs = [c for c in CONFIGS if not only or c[0] in only]
    for name, env_ovr in configs:
        env = dict(os.environ)
        for k in ("REID_ORT_GPU_MEM_LIMIT_MB", "REID_ORT_ARENA",
                  "REID_ORT_CUDNN_HEURISTIC"):
            env.pop(k, None)
        env.update(env_ovr)
        json_out = os.path.join(out_dir, f"opt_vram_{name}.json")
        cmd = [sys.executable, BENCH, "--variant", "fusion", "--preproc", "gpu",
               "--dino-tta", "224", "--dino-model", DINO,
               "--siglip-model", SIGLIP, "--fusion-w", "0.8",
               "--images", images, "--query", query, "--gallery", gallery,
               "--json", json_out, "--device", "cuda"]
        print(f"\n===== [{name}] env={env_ovr} =====", flush=True)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                               text=True, timeout=900,
                               encoding="utf-8", errors="replace")
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc = -999
            out = e.stdout if isinstance(e.stdout, str) else ""
            err = "TIMEOUT after 900s"
        dt = time.time() - t0
        rep = None
        if os.path.isfile(json_out):
            try:
                with open(json_out, "r", encoding="utf-8") as f:
                    rep = json.load(f)
            except Exception:  # noqa: BLE001
                rep = None
        tail = "\n".join((out or "").strip().splitlines()[-25:])
        print(tail, flush=True)
        if rc != 0:
            print(f"[{name}] FAILED rc={rc}\n{(err or '')[-2000:]}", flush=True)
        results.append({
            "name": name, "env": env_ovr, "returncode": rc, "seconds": dt,
            "ok": rc == 0 and rep is not None,
            "report": rep,
            "stdout_tail": tail,
            "stderr_tail": (err or "")[-2000:],
        })

    raw = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "device": "NVIDIA GeForce RTX 4090",
           "config": {"variant": "fusion", "tta_scales": [224], "fusion_w": 0.8,
                      "preproc": "gpu", "draft_factor": 1.0},
           "dataset": dataset.replace("\\", "/"),
           "weights": {"dino": DINO.replace("\\", "/"),
                       "siglip": SIGLIP.replace("\\", "/")},
           "results": results}
    raw_path = os.path.join(
        out_dir,
        "opt_vram_cap_raw.json" if not only
        else "opt_vram_cap_raw_%s.json" % "_".join(sorted(only)))
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"\n[sweep] raw -> {raw_path}", flush=True)
    for r in results:
        if r["ok"]:
            rr = r["report"]
            print(f"  {r['name']:<18} peak_smi={rr.get('peak_vram_smi_mb')} "
                  f"lat={rr['latency_b1_ms']:.1f} fps={rr['throughput_fps_best']:.1f}",
                  flush=True)
        else:
            print(f"  {r['name']:<18} FAILED rc={r['returncode']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
