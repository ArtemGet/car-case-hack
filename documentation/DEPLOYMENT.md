# Deployment / offline runtime

How the solution is packaged and run: `docker compose up` starts the full service
(batch inference + HTTP API + web UI + pgvector store) from one command, with the
model weights baked into the image and **no network access at inference time**.

## Image layout (`Dockerfile`, multi-stage)

| Stage | Base | What it does |
|-------|------|--------------|
| `builder` | `python:3.12-slim` | Creates `/opt/venv`, `pip install -r requirements-runtime.txt` (network allowed) — **minimal runtime deps only**, NOT the full train stack (`requirements.txt`). Copies source. Weights are pre-exported and only copied in, never downloaded. |
| `runtime` | `python:3.12-slim` | Copies `/opt/venv` + code + the baked weights. **No pip, no network.** |

Base is deliberately `python:3.12-slim` (~50 MB pull), **not** `nvidia/cuda:*` — a
2.5 GB CUDA base pull once hung the offline build. The CUDA runtime libraries are
brought in by the pip wheels instead: `torch==2.6.0` (cu124) + `onnxruntime-gpu==1.26.0`.

The runtime image contains exactly:

- `/opt/venv` — runtime Python env (`torch==2.6.0`, `onnxruntime-gpu==1.26.0`, imaging/API deps);
- `/app/reid`, `/app/service`, `/app/tools`, `/app/configs` — the solution source;
- `/app/artifacts/` — the baked deliverables: the **two** deploy ONNX
  (`siglip2_fp16.onnx`, `dinov2_b_fp16.onnx`) + `data_manifest.json`.

Baked deliverables (also the weight-gate subject):

```
artifacts/dinov2_b_fp16.onnx      164.5 MB   (DINOv2-B @224, deploy)
artifacts/siglip2_fp16.onnx       180.3 MB
--------------------------------------------
total                             344.8 MB   < 2 GiB  -> weight-gate PASS
```

`ENTRYPOINT` is the offline batch runner, one step `images + csv -> 3 files`.
Default is the champion **fusion** variant (224-only, TTA OFF) with both baked weights:

```
python -m service.infer.run --images /in/images \
    --query /in/test_query.csv --gallery /in/test_gallery.csv \
    --out /out \
    --variant fusion \
    --siglip-weights /app/artifacts/siglip2_fp16.onnx \
    --dino-onnx      /app/artifacts/dinov2_b_fp16.onnx
```

## Prototype / demo image (CPU, no GPU) — NOT the graded path

`Dockerfile` above is the **graded (оценка)** GPU build. For a small demo box
(2 vCPU / 2 GB RAM / **no GPU**) there is a separate, self-contained prototype:

| | Graded | Prototype |
|---|---|---|
| Dockerfile | `Dockerfile` | `Dockerfile.prototype` |
| Compose | `docker-compose.yml` (`gpus: all`, pgvector) | `docker-compose.prototype.yml` |
| Deps | `requirements-runtime.txt` (torch + onnxruntime-gpu) | `requirements-prototype.txt` (onnxruntime CPU, no torch) |
| Weights | `artifacts/{siglip2_fp16,dinov2_b_fp16}.onnx` | only `artifacts/siglip2_fp16.onnx` |
| Process | batch runner (`service.infer.run`) | HTTP API (`uvicorn service.api.app:app`) |
| Ports | api `8000`, web `8080` | api `80:8000` (`mem_limit 1900m`), web `8080` |

The prototype serves the **SigLIP2-only** CPU engine
(`REID_API_ENGINE=infer`, `REID_API_DEVICE=cpu`; DINOv2/fusion need CUDA). Static
UI is served from the committed `web/dist` bundle (`npm ci && npm run build`).

```bash
docker compose -f docker-compose.prototype.yml up --build
# API/health: http://localhost/health   Swagger: http://localhost/docs
# UI:         http://localhost:8080
```

## GPU

- Run with `--gpus all`; compose uses `gpus: all`.
- Base image is `python:3.12-slim`; `torch==2.6.0` (cu124 wheels) + `onnxruntime-gpu`
  embed their own CUDA 12.x runtime libs, so the host needs only the NVIDIA driver
  (`>= 525` for CUDA 12.x) and does not need a matching CUDA toolkit.

## OnnxRuntime CUDA needs torch's libs (important)

ONNX Runtime's `CUDAExecutionProvider` requires `cuBLAS`/`cuBLASLt`/`cuDNN` on the
dynamic-library search path. PyTorch ships exactly those inside the wheel at
`site-packages/torch/lib` (plus `site-packages/nvidia/*/lib`). If they are not
discoverable, ORT **does not error** — it silently drops the CUDA provider and runs
on CPU, i.e. **~60 FPS instead of ~314 FPS** for the fusion pipeline.

The image fixes this two ways (belt and suspenders):

1. `ldconfig` — the Dockerfile writes `torch/lib` and every `nvidia/*/lib` into
   `/etc/ld.so.conf.d/torch-cuda.conf` and runs `ldconfig` (process-wide).
2. `docker/entrypoint.sh` — prepends the same dirs to `LD_LIBRARY_PATH` at start.

Windows equivalent lives in `tools/bench_perf.py::enable_ort_cuda()`
(`os.add_dll_directory(torch/lib)`).

> **Current behavior (CUDA required, no silent CPU fallback):**
> `service/infer/backends.py` builds its providers via `_cuda_ort_providers(
> require_cuda=True)`, which returns `["CUDAExecutionProvider",
> "CPUExecutionProvider"]` **and raises** `RuntimeError` when the CUDA EP is not
> available instead of quietly running on CPU. So a missing lib path fails loudly
> (misconfiguration is caught) rather than silently costing ~314 → ~60 FPS. Use
> `--device cpu` only for an intentional CPU run.

## Weight gate — submission-only

The organizers scan **the solution directory** (the checked-out repository) and
sum every weight-extension file under it; only the files that are actually part
of the submission count. Dev training checkpoints are **not** shipped and must
not count. The `.gitignore` enforces that policy: it ignores `artifacts/*` and
re-includes only the deployed champion weights — the **two** deploy ONNX plus the
manifest: `artifacts/siglip2_fp16.onnx`, `artifacts/dinov2_b_fp16.onnx` and
`artifacts/data_manifest.json`. Everything else under `artifacts/` stays ignored.

The two ONNX are ~165–180 MB each (> GitHub's 100 MB hard limit), so they are
tracked through **Git LFS** — see `.gitattributes` and §Publishing below.

Three ways to run the gate, all excluding dev checkpoints:

```
# git-aware (most faithful): only git-tracked files == the actual submission
python tools/weight_gate.py --dir . --git                 # -> PASS (344.8 MB)

# explicit: only the baked deliverable directory
python tools/weight_gate.py --include-dir artifacts       # -> PASS (344.8 MB)

# whole tree with dev dirs (raw data, checkpoints, caches) pruned
python tools/weight_gate.py
```

`--git` fails loudly (exit 2) when the tree is not a git work tree, so a
mis-scoped run can never silently under-count. Both `--git` and `--include-dir`
count only the champion ONNX files; `weight_gate` is green at **344.8 MB**
(`< 2 GiB`).

CI (`.github/workflows/ci.yml`) runs lint, tests, and `weight_gate.py --git`; the
image build step is added at release time (heavy, deferred on the shared GPU box).

## Publishing the repository (open code, weights via LFS)

The submission is an **open, checked-out repository** that must contain the code,
docs and the two deliverable ONNX. The ONNX are >100 MB, so they cannot be
committed as plain blobs. Two supported layouts; pick one and document the link.

### Option A — Git LFS (default, already wired)

`.gitattributes` marks every `artifacts/*.onnx` as `filter=lfs diff=lfs
merge=lfs -text`, so a normal `git add` stores a pointer and the bytes in LFS.

Install the `git-lfs` client once per machine (all OSes):

```bash
# Windows
winget install GitHub.GitLFS    # or: it ships with Git for Windows
# macOS
brew install git-lfs
# Linux
sudo apt-get install git-lfs
```

```bash
git lfs install                 # once per machine
git init                        # already done in the working tree
git add .gitattributes .gitignore
git add artifacts/*.onnx artifacts/data_manifest.json
git commit -m "submission: code + LFS weights"
git remote add origin <repo-url>
git push origin main            # uploads ONNX to LFS

# after cloning on any machine:
git lfs pull                    # materialize the ONNX before `docker build`
```

The `Dockerfile` `COPY artifacts/siglip2_fp16.onnx artifacts/dinov2_b_fp16.onnx
artifacts/data_manifest.json ./artifacts/` then finds both files on a clean
clone. GitHub serves LFS to public repos within the free quota; verify with
`git lfs ls-files` (must list the two deploy ONNX, **~180 / ~165 MB** each).
If a downloaded `.onnx` is only **~134 bytes** it is an LFS pointer, not the
weights — run `git lfs pull` before `docker build`.

### Option B — no LFS: fetch weights by URL + sha256

Keep the ONNX out of git and fetch them in the builder stage **before** they are
needed (still offline at runtime — the runtime stage copies the built/copied
files, never the network):

```dockerfile
# in the builder stage, before COPY artifacts/*.onnx into runtime:
RUN set -eux; \
    mkdir -p /artifacts; \
    fetch() { url="$1"; dst="$2"; sha="$3"; \
      curl -fsSL "$url" -o "$dst"; \
      echo "$sha  $dst" | sha256sum -c -; }; \
    fetch "<url-siglip2>"  /artifacts/siglip2_fp16.onnx      <sha256>; \
    fetch "<url-dinov2>"   /artifacts/dinov2_b_fp16.onnx     <sha256>
```

Record the real URLs and the `sha256` of each file (e.g. in
`artifacts/data_manifest.json`); the `sha256sum -c` check makes the download
verifiable and tamper-evident. Docker build needs network; the runtime image does
not.

## Сборка и запуск

```bash
# 1. Сборка runtime-образа (pip/сеть нужны только в builder-стадии, см. выше)
docker build --progress=plain -t vehicle-reid:runtime .

# 2. Полный контур одной командой: batch-infer + API + web + pgvector
docker compose up
```

Офлайн-запуск батча одной командой (веса уже внутри образа, сеть не нужна):

```bash
docker compose run --rm infer
# вход : /in/images + /in/test_query.csv + /in/test_gallery.csv
# выход: /out/submission.csv, /out/embeddings.npy, /out/candidates.csv  (ровно 3 файла)
```

Ожидаемый результат — ровно **3 файла** в `/out`; на тестовом наборе это те же
**984 ответа / 126 отказов**, что и у референсного прогона.

## Commands

```bash
# Validate compose without building
docker compose config

# Full bring-up (builds the runtime image the first time)
docker compose up --build

# Batch run only (weights are inside the image)
docker compose run --rm infer

# Direct run
docker run --rm --gpus all -v "$PWD/data:/in:ro" -v "$PWD/out:/out" vehicle-reid:runtime
```
