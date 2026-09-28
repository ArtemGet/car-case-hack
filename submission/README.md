# `submission/` — тестовый вывод раннера (3 файла сдачи)

Артефакты, которые офлайн-раннер `python -m service.infer.run` записывает в `--out`.
Папка сдаётся вместе с кодом (ТЗ п.8) и **не** исключена `.gitignore`.

Источник: прогон **`runs/W2-5-test-final`** (final variant A: fusion `224-only`, **TTA OFF**,
GPU-preproc, ONNX fp16 CUDA EP, k-reciprocal `k1=8,k2=3,λ=0.5,pool=300`; порог отказа
**0.6970806**). Канонический деплой-прогон, верифицирован `eval-guardian` (exp-0080/0081/0082,
Δ = 0.0, два прогона байт-в-байт).

| Файл | Формат | Размер |
|------|--------|--------|
| `submission.csv` | без заголовка; `query_id,g1..g10` — ровно 10 `gallery_id` по убыванию уверенности | 404 040 Б |
| `embeddings.npy` | `float32` `(1860, 1024)`; сначала query (в порядке файла), затем gallery | 7 618 688 Б |
| `candidates.csv` | заголовок `query_id,gallery_id,confidence`; только top-1 при `confidence >= 0.6970806` (отказ = строки нет) | 74 816 Б |

Детали: 1110 query-строк в `submission.csv`, **984 принятых / 126 отказов (11.35 %)** в
`candidates.csv`; `gallery_id` без дублей в каждой строке; `validate_format` = OK.

Воспроизвести (в контейнере, веса внутри образа):

```bash
docker compose run --rm infer
# /in/{images,test_query.csv,test_gallery.csv} -> /out/{3 файла}
```

Полный конфиг и хэши входных CSV — в `artifacts/data_manifest.json`.
