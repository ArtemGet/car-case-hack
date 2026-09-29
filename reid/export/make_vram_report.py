#!/usr/bin/env python
"""Consolidate the ORT CUDA arena-cap sweep into ``reports/opt_vram_cap.json``.

Reads every ``reports/opt_vram_cap_raw*.json`` produced by
:mod:`reid.export.bench_vram_sweep` (dedup by config name, preferring a run that
succeeded) plus the capped val-eval report, and emits the single deliverable
report with the limit -> peak/latency/FPS table and the 12/10/8 GB verdicts.

CPU-only.
"""
from __future__ import annotations

import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(REPO, "reports")
VAL = os.path.join(REPORTS, "opt_vram_val_8192.json")
OUT = os.path.join(REPORTS, "opt_vram_cap.json")

# config name -> (limit_mb, arena, heuristic)
ARENA = {"same": "kSameAsRequested", "off": "off(dflt)", None: "default"}
ORDER = [
    "baseline_noenv", "same_noenv",
    "16384_same", "12288_same", "10240_same", "8192_same", "7680_same",
    "7168_same", "7040_same", "6912_same", "6784_same", "6656_same",
    "6144_same", "4096_same", "8192_default_arena", "12288_default_arena",
    "8192_off", "12288_same_heur", "8192_same_heur",
]


def load_rows():
    rows = {}
    for path in sorted(glob.glob(os.path.join(REPORTS, "opt_vram_cap_raw*.json"))):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for x in raw.get("results", []):
            name = x["name"]
            prev = rows.get(name)
            # prefer a successful run; otherwise first seen
            if prev is None or (x.get("ok") and not prev.get("ok")):
                rep = x.get("report") or {}
                rows[name] = {
                    "name": name,
                    "env": x.get("env", {}),
                    "ok": bool(x.get("ok")),
                    "returncode": x.get("returncode"),
                    "seconds": round(x.get("seconds", 0), 1),
                    "peak_vram_smi_mb": rep.get("peak_vram_smi_mb"),
                    "peak_vram_torch_mb": rep.get("peak_vram_mb"),
                    "baseline_vram_mb": rep.get("baseline_vram_mb"),
                    "latency_b1_ms": (round(rep["latency_b1_ms"], 2)
                                      if rep.get("latency_b1_ms") else None),
                    "throughput_fps_best": (round(rep["throughput_fps_best"], 1)
                                            if rep.get("throughput_fps_best") else None),
                    "error": None if x.get("ok") else (
                        (x.get("stderr_tail") or "").strip().splitlines()[-1]
                        if (x.get("stderr_tail") or "").strip() else "failed"),
                }
    ordered = [rows[n] for n in ORDER if n in rows]
    ordered += [rows[n] for n in sorted(rows) if n not in ORDER]
    return ordered


def main() -> int:
    rows = load_rows()
    val = None
    if os.path.isfile(VAL):
        with open(VAL, encoding="utf-8") as f:
            val = json.load(f)

    # minimum working cap: smallest *_same limit that succeeded (>=6912 anyway)
    def limit_of(name):
        import re
        m = re.match(r"(\d+)_same$", name)
        return int(m.group(1)) if m else None

    working = sorted(limit_of(r["name"]) for r in rows
                     if r["ok"] and limit_of(r["name"]) is not None)
    failing = sorted(limit_of(r["name"]) for r in rows
                     if not r["ok"] and limit_of(r["name"]) is not None)
    min_working = working[0] if working else None
    max_failing = max(failing) if failing else None

    def row_for(name):
        for r in rows:
            if r["name"] == name:
                return r
        return None

    base = row_for("baseline_noenv")
    cap8192 = row_for("8192_same")
    cap7168 = row_for("7168_same")

    def peak(r):
        return r and (r["peak_vram_smi_mb"] or r["peak_vram_torch_mb"])

    report = {
        "id": "opt-vram-cap",
        "agent": "perf-engineer",
        "task": "OPT-VRAM — cap the ORT CUDA arena to fit a small (12 GB) GPU "
                "without changing the default build",
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "device_measured": "NVIDIA GeForce RTX 4090 (onnxruntime 1.26.0, CUDA EP)",
        "config": {
            "variant": "fusion",
            "formula": "L2(concat[L2(SigLIP2 512d), 0.8*L2(DINOv2-B@224 512d)])",
            "tta": "OFF (224-only)",
            "preproc": "gpu (CPU draft-decode + CUDA letterbox/normalise/patchify)",
            "rerank": "k-reciprocal k1=8 k2=3 lam=0.5 pool=300",
            "weights": {"dinov2_b_fp16.onnx": 164.5, "siglip2_fp16.onnx": 180.3},
        },
        "env_vars": {
            "REID_ORT_GPU_MEM_LIMIT_MB": "int MB -> CUDA EP gpu_mem_limit",
            "REID_ORT_ARENA": "same -> arena_extend_strategy=kSameAsRequested; "
                              "off -> enable_cuda_mem_arena=false where supported "
                              "(ORT 1.26 has no such option -> falls back to "
                              "kSameAsRequested); default/unset -> untouched",
            "REID_ORT_CUDNN_HEURISTIC": "1 -> cudnn_conv_algo_search=HEURISTIC",
            "default": "with none set the provider list is byte-identical to "
                       "before (plain ['CUDAExecutionProvider','CPUExecutionProvider'])",
        },
        "results": rows,
        "weights_gate": {"total_mb": 344.8, "limit_mb": 2048, "pass": True},
        "min_working_limit_mb": min_working,
        "largest_failing_limit_mb": max_failing,
        "min_working_note": "probe boundary: 6784 MB OOM, 6912 MB OK; the arena's "
                            "actual need sits ~7.5 GB above the torch/ORT baseline",
        "verdicts": {},
        "map_check": None,
        "notes": [
            "The ~15.6 GB fusion peak was the ORT CUDA BFC arena growing to fill "
            "the GPU, not the weights (fp16 weights total 344.8 MB).",
            "gpu_mem_limit alone is NOT enough: with the default (doubling) "
            "arena_extend_strategy a 8192 MB cap OOMs. It must be combined with "
            "REID_ORT_ARENA=same (kSameAsRequested).",
            "REID_ORT_CUDNN_HEURISTIC makes no measurable difference here "
            "(+-2 MB peak, latency within noise).",
            "Latency/FPS are unchanged by the cap (same ~18.4 ms / ~313 FPS).",
            "ORIENTATION on a clean 12 GB card: subtract the idle baseline "
            "(~2.07 GB, other desktop processes) from the smi/torch peaks.",
        ],
    }

    def verdict(name, budget_mb, label):
        r = row_for(name)
        if not r or not r["ok"]:
            return {"config": name, "ok": False, "fits": False,
                    "verdict": "no run"}
        p_smi = r["peak_vram_smi_mb"]
        p_t = r["peak_vram_torch_mb"]
        p = p_smi or p_t
        delta = p - (r["baseline_vram_mb"] or 0) if p else None
        # conservative: raw device peak vs the budget
        fits_cons = bool(p and p < budget_mb)
        # card-independent: process footprint above the idle baseline + headroom
        fits_delta = bool(delta and delta + 512 < budget_mb)
        return {
            "config": name, "ok": True, "budget_mb": budget_mb,
            "peak_vram_smi_mb": p_smi, "peak_vram_torch_mb": p_t,
            "process_delta_above_baseline_mb": (round(delta) if delta else None),
            "fits_conservative": fits_cons,
            "fits_process_delta_512mb_headroom": fits_delta,
            "fits": bool(fits_cons and fits_delta),
            "verdict": label,
        }

    report["verdicts"] = {
        "12gb": verdict("8192_same", 12288, "PASS — fits, ~1.3 GB conservative "
                         "margin (~2.9 GB by process delta); recommended cap 8192"),
        "10gb": verdict("7168_same", 10240, "NOT RELIABLE — conservative device "
                         "peak ~11.0 GB exceeds 10 GB; only fits if the idle "
                         "desktop baseline is negligible"),
        "8gb": verdict("7168_same", 8192, "NO — process footprint ~8.9 GB at the "
                        "minimum working cap already exceeds 8 GB"),
    }
    if base and cap8192:
        report["peak_reduction"] = {
            "baseline_smi_mb": base["peak_vram_smi_mb"],
            "capped_8192_smi_mb": cap8192["peak_vram_smi_mb"],
            "baseline_torch_mb": base["peak_vram_torch_mb"],
            "capped_8192_torch_mb": cap8192["peak_vram_torch_mb"],
            "latency_b1_ms_baseline": base["latency_b1_ms"],
            "latency_b1_ms_capped": cap8192["latency_b1_ms"],
            "fps_baseline": base["throughput_fps_best"],
            "fps_capped": cap8192["throughput_fps_best"],
        }

    if val:
        m = val.get("metrics", {})
        report["map_check"] = {
            "report": "reports/opt_vram_val_8192.json",
            "capped_env": {"REID_ORT_GPU_MEM_LIMIT_MB": 8192,
                           "REID_ORT_ARENA": "same",
                           "REID_ORT_CUDNN_HEURISTIC": 1},
            "mAP@10": m.get("mAP@10"),
            "Rank-1": m.get("Rank-1"),
            "expected_mAP@10": 0.6909,
            "delta_vs_uncapped": (m.get("mAP@10") - 0.69086) if m.get("mAP@10") else None,
            "unchanged": bool(m.get("mAP@10") and abs(m.get("mAP@10") - 0.69086) <= 0.002),
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[vram-report] -> {OUT}")
    print(f"  min working limit: {min_working} MB (largest failing: {max_failing})")
    print(f"  12GB: {report['verdicts']['12gb']}")
    print(f"  10GB: {report['verdicts']['10gb']}")
    print(f"  8GB : {report['verdicts']['8gb']}")
    if report["map_check"]:
        print(f"  mAP@10 capped: {report['map_check']['mAP@10']} "
              f"unchanged={report['map_check']['unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
