#!/bin/sh
# Offline runtime entrypoint for the vehicle-ReID image.
#
# WHY THIS EXISTS (critical, found by perf-engineer):
# ONNX Runtime's CUDAExecutionProvider needs cuBLAS/cuBLASLt/cuDNN on the dynamic
# library search path. PyTorch ships those exact libraries inside the wheel at
#   $SITE/torch/lib            (libcublas*, libcudnn*, ...)
#   $SITE/nvidia/*/lib         (pip nvidia-* wheels)
# On Windows the equivalent fix is tools/bench_perf.py::enable_ort_cuda()
# (os.add_dll_directory(torch/lib)); on Linux it is LD_LIBRARY_PATH.
#
# If these are missing, ORT does NOT raise — it SILENTLY drops the CUDA EP and
# runs on CPU, giving ~60 FPS instead of ~216 FPS for the fusion pipeline.
#
# Kept as a tiny shell wrapper (not a static ENV) so the site-packages path is
# discovered at container start and cannot drift when the interpreter changes.
set -eu

SITE="/opt/venv/lib/python3.12/site-packages"
extra=""
for d in "$SITE/torch/lib" "$SITE"/nvidia/*/lib; do
    if [ -d "$d" ]; then
        if [ -n "$extra" ]; then
            extra="$extra:$d"
        else
            extra="$d"
        fi
    fi
done

if [ -n "$extra" ]; then
    export LD_LIBRARY_PATH="$extra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$@"
