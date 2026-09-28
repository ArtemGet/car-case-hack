# Vehicle ReID — Web UI

Демонстрационный интерфейс к HTTP API сервиса (`service/api`, FastAPI). React + TypeScript +
Vite, минимальные зависимости (только `react` / `react-dom`), без UI-китов и без обращений во
внешнюю сеть в рантайме.

## Что умеет

1. **Экран поиска** — загрузка изображения (файл / drag-and-drop), задание BBox интерактивно
   (протянуть мышью по кузову) или численно (`X/Y/W/H`), кнопка «Весь кадр».
   `POST /api/v1/search` (JSON + base64) → таблица top-N с превью, `score`, `confidence`,
   `модель`, `latency_ms` и **явным статусом «отказ»**.
2. **BBox + Grad-CAM** — рамка BBox рисуется поверх кадра запроса; панель показывает кроп
   запроса и накладывает карту из `POST /api/v1/explain` (композит `overlay_png_base64` или
   «чистая» `heatmap_png_base64`; переключатель показа и прозрачность).
3. **Выбор модели** — селектор из `GET /api/v1/models` (`stub` / `dino` / `siglip` / `fusion`);
   варианты без локальных весов задизейблены.
4. **Экспорт** — `candidates.csv` (`query_id,gallery_id,confidence`, отказ = ни одной строки,
   как в контракте сдачи) и `results.json` (полный ответ, включая `refused`, `threshold`).

> UI **не содержит логики модели** и **не подменяет данные заглушками**: если backend
> недоступен, показывается явная ошибка с подсказкой, а не фейковый результат.

## Запуск (разработка)

```bash
cd web
npm install          # один раз, требует сеть (публичный npm registry)
npm run dev          # http://127.0.0.1:5173
```

Поднимите API рядом (демо-стенд работает на заглушке, без весов и GPU):

```bash
# из корня репозитория
uvicorn service.api.app:app --port 8000
```

Dev-сервер проксирует `/api/*` и `/health` на backend (`vite.config.ts`). По умолчанию target —
`http://127.0.0.1:8000`; переопределяется без правки файла:

```powershell
$env:VITE_API_TARGET="http://127.0.0.1:8000"; npm run dev
```

## Сборка

```bash
npm run build        # → web/dist (статические файлы, офлайн)
npm run preview      # локальный предпросмотр сборки
npm run typecheck    # tsc --noEmit
```

`dist/` отдаётся любым статическим сервером. Для продакшена в контейнере — `web/nginx.conf`
(SPA-fallback + прокси `/api` и `/health` на сервис `api:8000`); подключение файла в
`docker-compose.yml` — зона devops.

## Контракт API

Реальный контракт — `service/api/schemas.py` (+ авто-Swagger `/docs`); краткая выжимка и
ожидания UI — [`API.md`](./API.md).

## Структура

```
web/
  index.html
  vite.config.ts          прокси /api и /health, сборка
  src/
    main.tsx              точка входа
    App.tsx               компоновка, состояние, вызовы API
    api.ts                клиент service/api + нормализация ответов
    types.ts              типы контракта
    styles.css            тёмная тема
    components/
      ImageCanvas.tsx     кадр + рисование/ввод BBox
      CropPreview.tsx     кроп запроса + наложение Grad-CAM
      ResultsTable.tsx    таблица top-N
      ExportBar.tsx       экспорт CSV/JSON
  API.md                  контракт и заметки по интеграции
  nginx.conf              прод-раздача (опционально)
```

## Ограничения

- Превью кандидатов — плейсхолдер: в API пока нет image-эндпоинта галереи. UI автоматически
  использует `image_url`, если backend начнёт его возвращать.
- `npm install` не выполнялся в рамках задачи (offline-кэш npm неполон) — нужен доступ к
  npm registry один раз.
