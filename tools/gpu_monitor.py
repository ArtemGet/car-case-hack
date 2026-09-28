"""Run a training command while sampling ``nvidia-smi`` GPU utilisation (W1-8).

    python tools/gpu_monitor.py --interval 2 --json reports/util_before.json -- \
        python -m reid.train --config configs/dinov2_b.yaml --out runs/perf-before \
        --dataset "docs/Датасет/dataset" --epochs 4

Prints avg util, %samples >=70, p10/p50/p90 and the longest zero-run. Only
``nvidia-smi`` (a local, unauthenticated binary) is used.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time


def sample_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        parts = out.stdout.strip().splitlines()[0].split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:  # noqa: BLE001
        return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--warmup-sec", type=float, default=0.0,
                    help="exclude this many seconds of startup from the stats")
    ap.add_argument("--json", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command after --")
    args = ap.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("no command given", file=sys.stderr)
        return 2

    samples = []
    stop = threading.Event()

    def worker():
        t_start = time.perf_counter()
        while not stop.is_set():
            samples.append((time.perf_counter() - t_start, *sample_util()))
            stop.wait(args.interval)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    dt = time.perf_counter() - t0
    stop.set()
    th.join(timeout=5)

    def stats(rows):
        utils = [u for _, u, _ in rows if u is not None]
        mems = [m for _, _, m in rows if m is not None]
        longest_zero = 0
        cur = 0
        for u in utils:
            if u <= 1:
                cur += 1
                longest_zero = max(longest_zero, cur)
            else:
                cur = 0
        return {
            "samples": len(utils),
            "util_avg": st.mean(utils) if utils else None,
            "util_p10": sorted(utils)[int(0.1 * (len(utils) - 1))] if utils else None,
            "util_p50": st.median(utils) if utils else None,
            "util_p90": sorted(utils)[int(0.9 * (len(utils) - 1))] if utils else None,
            "samples_ge70_pct": 100.0 * sum(u >= 70 for u in utils) / len(utils) if utils else None,
            "samples_ge90_pct": 100.0 * sum(u >= 90 for u in utils) / len(utils) if utils else None,
            "samples_zero_pct": 100.0 * sum(u <= 1 for u in utils) / len(utils) if utils else None,
            "longest_zero_run_samples": longest_zero,
            "mem_used_avg_mb": st.mean(mems) if mems else None,
            "mem_used_max_mb": max(mems) if mems else None,
        }

    steady_rows = [r for r in samples if r[0] >= args.warmup_sec]
    report = {
        "cmd": " ".join(cmd),
        "wall_sec": dt,
        "interval_s": args.interval,
        "warmup_sec_excluded": args.warmup_sec,
        "full": stats(samples),
        "steady": stats(steady_rows),
        "raw_samples": [[round(t, 1), u] for t, u, _ in samples],
        "returncode": proc.returncode,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
