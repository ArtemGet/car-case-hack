# NOTE: no `# syntax=docker/dockerfile:1` directive on purpose — this file uses
# only built-in Dockerfile instructions, so the build needs no extra frontend
# image pull (one less network dependency for the offline-stand build).
#
# Multi-stage build: build WITH network, run OFFLINE with weights baked in.
#
#   Stage 1 (builder)  install Python deps into an isolated venv (needs pip/network).
#   Stage 2 (runtime)  copy only the venv + code + baked weights. No pip, no
#                      network at inference time. ENTRYPOINT is the batch runner.
#
# GPU runtime (stand: CUDA 12.2 driver, A5000):
#   * base is the small `python:3.12-slim` (~50 MB pull) — deliberately NOT
#     `nvidia/cuda:*` (a 2.5 GB pull once hung the build); pip wheels bring the
#     CUDA libs instead.
#   * `torch==2.6.0` from PyPI is the cu124 build and pulls the matching
#     `nvidia-*-cu12` wheels (cuBLAS/cuDNN/cuFFT...); `onnxruntime-gpu==1.26.0`
#     provides the CUDAExecutionProvider.
#   * run with `docker run --gpus all ...` / compose `gpus: all`; the host NVIDIA
#     driver (>= 525 for CUDA 12.x) is injected by the container toolkit, so the
#     image does not need the host's exact CUDA toolkit.
#
# CRITICAL — ONNX Runtime CUDA vs torch libs (see docker/entrypoint.sh):
#   ORT's CUDAExecutionProvider needs cuBLAS/cuDNN on the library search path,
#   and PyTorch already ships/receives them in torch/lib and nvidia/*/lib.
#   Without that path ORT SILENTLY falls back to CPU (~60 FPS instead of
#   ~216 FPS on fusion, and `get_providers()[0]` becomes CPU). We register those
#   dirs system-wide with ldconfig AND export LD_LIBRARY_PATH from the
#   entrypoint, so the CUDA EP loads either way.

# ---------------------------------------------------------------- builder ----
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain for any sdist-only dependency; kept out of the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# CR-18/CR-19: install ONLY the minimal runtime deps. The full training stack
# (optuna/tensorboard/matplotlib/scikit-learn/timm/transformers/onnx) lives in
# requirements.txt and is deliberately NOT installed here — pulling it made the
# build balloon to gigabytes and hang (see requirements-runtime.txt).
COPY requirements-runtime.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-runtime.txt

# Solution source needed to build/export (kept in builder only).
COPY reid/ ./reid/
COPY service/ ./service/
COPY tools/ ./tools/
COPY configs/ ./configs/

# Weights are pre-exported and versioned under artifacts/ (deliverables):
#   ROVNO the two deploy ONNX (variant A, 224-only, TTA OFF):
#     artifacts/siglip2_fp16.onnx + artifacts/dinov2_b_fp16.onnx  ~= 344.8 MiB
#   (< 2 GiB weight gate). The TTA@280 graph is NOT shipped.
# Re-export here only when regenerating (needs runs/<id>/best.pt, not shipped):
# RUN /opt/venv/bin/python -m reid.export.export_dino_onnx \
#         --out artifacts/dinov2_b_fp16.onnx --fp16

# ---------------------------------------------------------------- runtime ----
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime OS deps only: libGL/libglib for Pillow/OpenCV, libgomp for torch.
# No build chain, no pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Register torch's bundled CUDA/cuDNN libs (and any nvidia-* pip libs) with the
# dynamic linker so ONNX Runtime can load its CUDAExecutionProvider. ldconfig is
# the robust, process-wide fix; the entrypoint repeats it via LD_LIBRARY_PATH.
RUN printf '%s\n' \
        /opt/venv/lib/python3.12/site-packages/torch/lib \
        > /etc/ld.so.conf.d/torch-cuda.conf \
    && find /opt/venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -type d -name lib \
        -exec sh -c 'echo "$1" >> /etc/ld.so.conf.d/torch-cuda.conf' _ {} \; \
    && ldconfig

WORKDIR /app
COPY reid/ ./reid/
COPY service/ ./service/
COPY tools/ ./tools/
COPY configs/ ./configs/

# Bake the deliverable weights inside the image (offline inference): EXACTLY the
# two deploy ONNX (SigLIP2 fp16 + DINOv2-B fp16 @224) + the manifest. The
# TTA@280 graph and any other dev weights are deliberately NOT copied.
COPY artifacts/siglip2_fp16.onnx artifacts/dinov2_b_fp16.onnx artifacts/data_manifest.json ./artifacts/

# Entry point: lib-path shim -> the offline batch runner (images + csv -> 3 files).
COPY docker/entrypoint.sh /usr/local/bin/vehicle-reid-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/vehicle-reid-entrypoint \
    && chmod 0755 /usr/local/bin/vehicle-reid-entrypoint

# One step: images + test_query.csv + test_gallery.csv -> submission.csv,
# embeddings.npy, candidates.csv.
# Default = champion FUSION variant A: SigLIP2 fp16 + DINOv2-B fp16 @224, TTA OFF
# (224-only). Both ONNX are baked in and wired into service.infer (ONNX-DINO
# backend, run.py default variant="fusion"). Override --variant siglip|dino for
# a single-backbone run; --dino-tta '224,280' re-enables the non-deploy TTA
# ablation (needs a 280 graph that is NOT shipped).
ENTRYPOINT ["/usr/local/bin/vehicle-reid-entrypoint", "python", "-m", "service.infer.run"]
CMD ["--images", "/in/images", \
     "--query", "/in/test_query.csv", \
     "--gallery", "/in/test_gallery.csv", \
     "--out", "/out", \
     "--variant", "fusion", \
     "--siglip-weights", "/app/artifacts/siglip2_fp16.onnx", \
     "--dino-onnx", "/app/artifacts/dinov2_b_fp16.onnx"]
