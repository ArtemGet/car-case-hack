# Vehicle ReID (open-set) — цифровой признак автомобиля

Решение задачи **№7 «Фалькон Тех»** (ЛЦТ 2026): по кропу автомобиля (BBox задан) строится
эмбеддинг для **кросс-камерного сопоставления** в **open-set** режиме — **без госномера**, без
детекции и трекинга. Артефакты сдачи: `submission.csv` (top-10 галереи на каждый query),
`embeddings.npy` (`(query+gallery, D)` float32), `candidates.csv` (top-1 + `confidence`;
отказ = отсутствие строки). Плюс микросервисный прототип (API + UI + векторный поиск).

---

## 1. Требования

- **Python 3.12** (как в Docker-образе; локально совместимо с 3.10+). Пины — `requirements.txt` / `pyproject.toml`.
- **Полный пайплайн:** NVIDIA GPU + CUDA 12.x + Docker (`--gpus all`).
- **Демо и тесты:** работают на CPU (SigLIP2 ONNX, без torch).

## 2. Быстрый старт — инференс одной командой (офлайн)

`service/infer/run.py` читает каталог JPEG и два CSV и **без доступа в сеть** пишет ровно три
файла в `--out`:

```bash
python -m service.infer.run \
    --images  /in/images \
    --query   /in/test_query.csv \
    --gallery /in/test_gallery.csv \
    --out     /out \
    --variant fusion
# → /out/submission.csv, /out/embeddings.npy, /out/candidates.csv
```

Веса `artifacts/*.onnx` подхватываются автоматически (либо `--dino-onnx` / `--siglip-weights`).
Варианты: `fusion` (по умолчанию), `siglip`, `dino`. Детерминизм: seed=42 + `cudnn.deterministic`;
два полных прогона дают байт-идентичные файлы.

Проверка формата:

```bash
python tools/validate_format.py --out /out --query /in/test_query.csv --gallery /in/test_gallery.csv
```

## 3. Тесты

```bash
pytest -q
```

## 4. Обучение (воспроизведение)

Данные — датасет ЛЦТ 2026 (задача №7): JPEG + `train.csv` / `test_query.csv` / `test_gallery.csv`
(манифест `artifacts/data_manifest.json`). Val-сплит — `reid/data/splits.py::holdout_val`
(seed=42, open-set ≈ 0.216).

```bash
python -m reid.train --config configs/dinov2_b.yaml --out <run_dir> --dataset <dataset_dir>
```

Конфиги — `configs/*.yaml`; метрика — обёртка `reid/eval` над официальным `evaluate.py`;
калибровка порога — `reid/calibrate.py`; ре-ранжирование — `reid/rerank.py`.

## 5. Демо-сервис (CPU, без GPU)

```bash
uvicorn service.api.app:app --host 0.0.0.0 --port 8000
# → / (UI), /docs (Swagger/OpenAPI)
```

CPU-образ (2 vCPU / 2 ГБ, только SigLIP2 ONNX, без torch), UI под nginx:

```bash
docker compose -f docker-compose.prototype.yml up --build   # web на http://localhost:8080
```

## 6. Результаты

- **Качество (hold-out val, seed=42):** mAP@10 **0.6909**, Rank-1 **0.6769**, Rank-5 0.8197,
  mINP **0.5982** (fusion 224-only, TTA OFF, k-reciprocal pool=300). Проверено официальным `evaluate.py`.
- **Порог отказа 0.6970806:** F1 **0.9702**, TNR **1.0**, балл `0.7·F1+0.3·TNR` **0.9792**
  (PR-AUC 0.9922). Выбран на val, тест не использовался; скор = cosine top-1.
- **Веса:** 2 ONNX fp16, **344.8 МБ** (< 2 ГБ).
- **Перф (4090, протокол организаторов):** **17.9 мс / 314 FPS**; на стенде A5000 оценочно
  ~29–40 мс / 106–157 FPS.

## 7. Документация

- [`documentation/architecture.md`](documentation/architecture.md) — архитектура и обоснование технологий.
- [`documentation/metrics.md`](documentation/metrics.md) — метрики на val и логика порога.
- [`documentation/errors.md`](documentation/errors.md) — честный анализ ошибок.
- [`documentation/sources.md`](documentation/sources.md) — внешние источники (веса/датасеты/библиотеки, URL + sha256).
- [`documentation/DEPLOYMENT.md`](documentation/DEPLOYMENT.md) — деплой и прототип.

> Презентация решения сдаётся **отдельной ссылкой** (PDF/PPTX) и в репозиторий не входит.

## 8. Структура репозитория

```
reid/          данные, модель, обучение, метрики, экспорт, ре-ранк, калибровка
service/infer/ офлайн batch-раннер (3 файла)
service/api/   FastAPI + OpenAPI
web/           React/TS UI
tools/         validate_format.py, bench_perf.py, weight_gate.py
tests/         unit + acceptance
configs/       YAML конфиги экспериментов
artifacts/     2 fp16 ONNX (344.8 МБ) + data_manifest.json
documentation/ архитектура, метрики, ошибки, источники, деплой
```

## 9. Соответствие ТЗ

- Офлайн-рантайм: `docker build` — с сетью, инференс — без.
- Docker (multi-stage) + `docker-compose`; API на OpenAPI/Swagger.
- Потоковость: каждый query обрабатывается **независимо** (per-query); k-reciprocal — внутри query.
- Веса ≤ 2 ГБ; обучение и подбор гиперпараметров — только на своём val, не на закрытых ответах.
