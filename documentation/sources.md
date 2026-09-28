# Внешние источники

Все внешние веса, датасеты и библиотеки — **публичные, без аутентификации**, с зафиксированными
URL/тегом и sha256. В рантайме инференса сеть не используется вовсе; сеть нужна только для
`pip install` при сборке образа и единоразовой загрузки публичных весов.

## 1. Данные задачи

| Артефакт | Источник | sha256 / digest |
|---|---|---|
| `train.csv` | Датасет ЛЦТ 2026, задача №7 «Фалькон Тех» | `bd1df45b052ae9aabb7fd356898e244875f8cad2f111a3f76977c1b90bce9268` |
| `test_query.csv` | ↑ | `97e1ed21942bae9c95b1ce2e5d339d9f635bf49fa484367e6b19349789bb9b4c` |
| `test_gallery.csv` | ↑ | `a64ed21fa39bbfd8c7a172468d043415800500b6ed695cdb8162188cf9005496` |
| images (11416 JPEG) | ↑ | digest `cb82fcd86f49355fbcf231ff4de8bbca471f3a32fc349356a6233db206395663` (sha256 по строкам `filename\tsize`) |

Манифест: `artifacts/data_manifest.json`.
Раскладка: `train` 9556 (1541 ID, 96 камер), `test_query` 1110, `test_gallery` 750; open-set ~20 %.

## 2. Предобученные веса

| Модель / файл | URL (публичный) | Лицензия | sha256 |
|---|---|---|---|
| DINOv2 ViT-B/14 `vit_base_patch14_dinov2.lvd142m` (init backbone) | `https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m` (rev `4685c99dabffe5affac90bd99dbffd25801ae58d`, `model.safetensors`) | Apache-2.0 | `55cbb5d887b336d430e649c277b85a1429e724871f9d02ac16203235886d8c7b` |
| SigLIP2 NaFlex 512d — ONNX (frozen, ключевой компонент fusion) | `https://huggingface.co/occurra/vehicle_reid_siglip2_naflex_512d/resolve/main/vehicle_reid_siglip2_naflex_512d.onnx` | apache-2.0 | `201a17f50eb4f4d7fd7b16fd01adac0e56444570a86ad52532b1d4bf5b0a956b` |
| SigLIP2 NaFlex 512d — torch-исходник | `.../resolve/main/vehicle_reid_siglip2_naflex_512d.pth` | apache-2.0 | `6733ac3a8343130d0c7db40bc227b38e1c638cdf89e2ef897fd6f90e821e184a` |

### 2.1 Собственные обученные / экспортированные веса (сдача)

| Файл | Роль | Размер | sha256 |
|---|---|---|---|
| `artifacts/dinov2_b_fp16.onnx` | fp16 ONNX-экспорт нашего чемпиона @224 (static, деплой) | 164.5 МБ | `84db06d9a0b06254f6b4dadc22897865e34c7a081143eebe9c1b97b6f6557d58` |
| `artifacts/siglip2_fp16.onnx` | fp16 ONNX из внешнего SigLIP2 | 180.3 МБ | `32bdebebee52a67f02e7580cec58550bfda3e55ac4e173c4e95e9e5b2712a647` |

Провенанс сдаваемого чемпиона: `artifacts/dinov2_b_fp16.onnx` — экспорт нашего финального
дообученного чекпоинта DINOv2-B (GeM + BNNeck + ArcFace, обучение с EMA, лучшая эпоха 21).

Итого сдаваемый `artifacts/` = **344.8 МБ, ровно 2 ONNX** (< 2 ГБ,
`tools/weight_gate.py --include-dir artifacts` → PASS). Официальный `evaluate.py` независимо
подтвердил: деплой (224-only, TTA OFF) **mAP@10 0.6909** (Δ = 0) и порог **0.6970806** (Δ = 0).

> **Оговорка про GitHub (рекомендация).** Каждый `.onnx` (160–180 МБ) превышает лимит GitHub
> в 100 МБ на файл для обычного репозитория. Для сдачи весов нужен **git-lfs** (`git lfs track
> "artifacts/*.onnx"`) либо размещение по публичному **URL + sha256** (таблица выше), либо сборка
> весов на месте из публичных источников (§2.1). Обычный `git push` больших файлов будет отклонён.

## 3. Программные зависимости (пины)

`requirements.txt` (полный train-стек) / `requirements-runtime.txt` (минимальный
runtime-образ) / `pyproject.toml` (без `>=`):

| Пакет | Версия | Назначение |
|---|---|---|
| torch / torchvision | 2.6.0 / 0.21.0 | обучение, DINOv2 (через timm) |
| onnxruntime-gpu | 1.26.0 | инференс ONNX fp16 (CUDA EP) в образе; CPU-цели → `onnxruntime` |
| timm | 1.0.11 | `vit_base_patch14_dinov2.lvd142m` |
| transformers | 5.6.2 | вспомогательно (конвертация SigLIP2) |
| numpy / pandas / pillow / opencv-python | 2.1.0 / 2.2.3 / 12.2.0 / 5.0.0.93 | данные, кроп, декод |
| fastapi / uvicorn / pydantic | 0.136.1 / 0.52.1 / 2.13.4 | API + OpenAPI |
| faiss-cpu | 1.9.0 | ANN-поиск (масштабирование) |
| pytest | 9.1.1 | тесты |
| optuna, pyyaml, tensorboard, matplotlib, scikit-learn | 4.1.0 / 6.0.2 / 2.18.0 / 3.10.0 / 1.6.0 | поиск гиперпараметров, конфиги, логи, графики |

## 4. Инфраструктура

| Компонент | Версия | Источник |
|---|---|---|
| Python | 3.12 (образ `python:3.12-slim`; локально 3.12.2) | python.org |
| База runtime-образа | `python:3.12-slim` (обе стадии `Dockerfile`) | Docker Hub |
| CUDA runtime | из pip-wheels `torch==2.6.0` (cu124) + `onnxruntime-gpu==1.26.0` — base-образ `nvidia/cuda:*` не используется | PyPI / NVIDIA |
| pgvector | `pgvector/pgvector:pg16` | Docker Hub |
| nginx | `nginx:1.27-alpine` | Docker Hub |
| Node/npm | 18+ | nodejs.org (сборка UI) |

## 5. Проверяемость

- Каждый sha256 можно пересчитать: `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>`.
- Источники весов были загружены один раз (публично, без токенов); повторная загрузка при
  инференсе не требуется — веса внутри образа.
- Никакие ключи/токены/приватные API/внутренние серверы не использовались и не хранятся в репозитории.
