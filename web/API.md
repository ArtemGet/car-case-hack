# API-контракт Web UI ↔ `service/api`

Реальный источник: `service/api/schemas.py`, `service/api/app.py` (авто-OpenAPI —
`http://127.0.0.1:8000/docs`, схема `openapi.json`). UI обращается **только** к
same-origin `/api/*` и `/health` (dev — прокси Vite, prod — `web/nginx.conf`);
внешних хостов и CORS нет.

Изображение передаётся **base64** (можно с префиксом `data:image/...;base64,`),
поэтому `python-multipart` не нужен. UI кодирует выбранный файл через
`FileReader.readAsDataURL` и шлёт строку как есть.

## `GET /health`

```json
{ "status": "ok", "version": "0.1.0", "engine": "stub",
  "gallery_size": 0, "gallery_dim": null, "models_available": 1 }
```

UI показывает `engine` / `gallery_size`; при недоступности — явный оффлайн-баннер.

## `GET /api/v1/models`

```json
{ "default": "stub", "models": [
  { "name": "stub", "variant": "stub", "dim": 512,
    "description": "…", "available": true, "is_default": true, "weights": null }
] }
```

UI строит селектор модели; недоступные варианты (`available=false`, нет весов)
показываются, но заблокированы.

> Примечание: скелет `service/api` пока принимает `model` в запросе, но `search`
> отвечает движком из конфигурации (`REID_API_ENGINE`); селектор начнёт влиять
> после доработки фасада (не блокирует UI).

## `POST /api/v1/search`

`application/json`:

```json
{
  "image_base64": "/9j/4AAQSkZJRg...",
  "bbox": { "x": 120, "y": 210, "w": 180, "h": 140 },
  "top_k": 10,
  "model": "stub"
}
```

`bbox` — пиксели **исходного** кадра, `w,h >= 1`, должен лежать внутри кадра
(иначе `422`). `model` опционален (по умолчанию — модель деплоя).

Ответ:

```json
{
  "query_id": null,
  "model": "stub",
  "bbox": { "x": 120, "y": 210, "w": 180, "h": 140 },
  "top_k": 10,
  "candidates": [
    { "rank": 1, "gallery_id": "g_000045", "score": 0.9310, "confidence": 0.9310 }
  ],
  "confidence": 0.9310,
  "threshold": 0.4623,
  "accepted": true,
  "refused": false,
  "latency_ms": 21.0
}
```

- `refused = true` → UI показывает статус **«ОТКАЗ»**; экспорт `candidates.csv`
  даёт один заголовок и ни одной строки (контракт сдачи: отказ = нет строк).
- Порог (`threshold`) сервер берёт из `reid.calibrate`; шкалы `stub`/`dino` и
  `fusion` **разные** (0.4623 vs 0.7464), поэтому модель выбирают осознанно.

## `POST /api/v1/explain` (W2-6)

`application/json` — те же `image_base64` + `bbox` (+ опц. `model`).

```json
{
  "model": "stub",
  "bbox": { "x": 120, "y": 210, "w": 180, "h": 140 },
  "method": "input-gradient (fallback; Grad-CAM — W2-6)",
  "heatmap_png_base64": "data:image/png;base64,...",
  "overlay_png_base64": "data:image/png;base64,..."
}
```

- `overlay_png_base64` — уже наложенная на кроп карта (`crop+map`, размер кропа);
  UI показывает её поверх плоского кропа с регулятором прозрачности.
- `heatmap_png_base64` — «чистая» карта; используется, если композита нет.
- Сейчас это честный фолбэк `input-gradient`; настоящий Grad-CAM по активациям —
  задача W2-6 (backend).

## Превью галереи

Отдельного image-эндпоинта пока нет, поэтому превью кандидатов — аккуратный
плейсхолдер с рангом. Если backend начнёт возвращать `image_url` в кандидате,
UI использует его автоматически (см. `ResultsTable.tsx`).

## Асинхронные задачи

`POST /api/v1/jobs` + `GET /api/v1/jobs/{id}` существуют в API; UI их пока не
использует (демо-сценарий укладывается в синхронный `/search`).
