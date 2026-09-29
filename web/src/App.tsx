import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, fileToDataUrl, humanError } from "./api";
import type { BBox, ExplainResponse, ModelInfo, SearchResponse } from "./types";
import ImageCanvas, { type NaturalSize } from "./components/ImageCanvas";
import CropPreview from "./components/CropPreview";
import ResultsTable from "./components/ResultsTable";
import ExportBar from "./components/ExportBar";

type HealthState =
  | { kind: "checking" }
  | { kind: "ok"; engine: string | null; gallerySize: number | null; models: number | null }
  | { kind: "down"; message: string };

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<NaturalSize | null>(null);
  const [bbox, setBbox] = useState<BBox | null>(null);
  const [topK, setTopK] = useState(10);

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<string>("");

  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [explaining, setExplaining] = useState(false);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthState>({ kind: "checking" });
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // -- backend discovery -----------------------------------------------------
  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((h) => {
        if (!alive) return;
        setHealth({
          kind: "ok",
          engine: h.engine,
          gallerySize: h.gallerySize,
          models: h.modelsAvailable,
        });
      })
      .catch((err) => {
        if (alive) setHealth({ kind: "down", message: humanError(err) });
      });

    api
      .models()
      .then((res) => {
        if (!alive) return;
        setModels(res.models);
        // Default to the server default when it has weights; otherwise the
        // first actually available model. Falls back to the raw default.
        const available = res.models.filter((m) => m.available);
        setModel(
          (current) =>
            current ||
            (available.some((m) => m.name === res.def) ? res.def : available[0]?.name ?? res.def),
        );
      })
      .catch(() => {
        /* model list is optional — search still falls back to the server default */
      });
    return () => {
      alive = false;
    };
  }, []);

  // -- local object URL + base64 payload (fully offline) ---------------------
  useEffect(() => {
    if (!file) {
      setObjectUrl(null);
      setImageBase64(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    let alive = true;
    fileToDataUrl(file)
      .then((b64) => {
        if (alive) setImageBase64(b64);
      })
      .catch(() => {
        if (alive) setImageBase64(null);
      });
    return () => {
      alive = false;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const resetResults = useCallback(() => {
    setResult(null);
    setSearchError(null);
    setExplain(null);
    setExplainError(null);
  }, []);

  const selectFile = useCallback(
    (next: File | null) => {
      setFile(next);
      setNaturalSize(null);
      setBbox(null);
      resetResults();
    },
    [resetResults],
  );

  const fullFrame = useCallback(() => {
    if (!naturalSize) return;
    setBbox({ x: 0, y: 0, w: naturalSize.w, h: naturalSize.h });
  }, [naturalSize]);

  // Selecting a frame should be immediately searchable: default the BBox to
  // the whole image, while a drag can still override it.
  const handleNaturalSize = useCallback((size: NaturalSize) => {
    setNaturalSize(size);
    setBbox((current) => current ?? { x: 0, y: 0, w: size.w, h: size.h });
  }, []);

  const runExplain = useCallback(
    async (payload: string, currentBbox: BBox) => {
      setExplaining(true);
      setExplainError(null);
      try {
        setExplain(
          await api.explain({ imageBase64: payload, bbox: currentBbox, model: model || null }),
        );
      } catch (err) {
        setExplainError(humanError(err));
      } finally {
        setExplaining(false);
      }
    },
    [model],
  );

  const runSearch = useCallback(async () => {
    if (!imageBase64 || !bbox) return;
    setSearching(true);
    setSearchError(null);
    setResult(null);
    setExplain(null);
    setExplainError(null);
    try {
      const response = await api.search({
        imageBase64,
        bbox,
        topK,
        model: model || null,
      });
      setResult(response);
      // Best-effort: a missing /explain must not hide the ranking.
      void runExplain(imageBase64, bbox);
    } catch (err) {
      setSearchError(humanError(err));
    } finally {
      setSearching(false);
    }
  }, [imageBase64, bbox, topK, model, runExplain]);

  const canSearch = Boolean(imageBase64 && bbox && !searching);
  const explainMethod = useMemo(() => explain?.method ?? null, [explain]);

  // Only models with weights are selectable; if the server reports none as
  // available, keep the previous behaviour (show all, disabled).
  const visibleModels = useMemo(() => {
    const available = models.filter((m) => m.available);
    return available.length > 0 ? available : models;
  }, [models]);

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Vehicle ReID · открытый поиск по кропу</h1>
          <p className="subtitle">
            Загрузите кадр, задайте BBox кузова — сервис вернёт top-N галереи,
            confidence и признак отказа.
          </p>
        </div>
        <HealthBadge health={health} />
      </header>

      {health.kind === "down" && (
        <div className="banner banner--error">
          <strong>Backend недоступен.</strong> {health.message}
          <div className="banner__hint">
            Поднимите API (<code>uvicorn service.api.app:app --port 8000</code>) и
            проверьте прокси <code>/api</code>. Интерфейс не подменяет данные заглушками.
          </div>
        </div>
      )}

      <main className="layout">
        <section className="panel panel--query">
          <h2>1. Запрос</h2>

          <div
            className="dropzone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f && f.type.startsWith("image/")) selectFile(f);
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => selectFile(e.target.files?.[0] ?? null)}
            />
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              {file ? "Заменить изображение" : "Выбрать изображение"}
            </button>
            <span className="dropzone__hint">
              {file ? file.name : "или перетащите файл сюда"}
            </span>
          </div>

          {objectUrl && (
            <ImageCanvas
              src={objectUrl}
              naturalSize={naturalSize}
              onNaturalSize={handleNaturalSize}
              bbox={bbox}
              onBboxChange={(b) => {
                setBbox(b);
                if (result) resetResults();
              }}
              disabled={searching}
            />
          )}

          <div className="query-controls">
            <button type="button" className="ghost" onClick={fullFrame} disabled={!naturalSize}>
              Весь кадр
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => setBbox(null)}
              disabled={!bbox}
            >
              Сбросить BBox
            </button>
            <label className="topk">
              top-N
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) =>
                  setTopK(Math.max(1, Math.min(100, Number(e.target.value) || 10)))
                }
              />
            </label>
            <label className="topk">
              модель
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={visibleModels.length === 0}
              >
                {visibleModels.length === 0 && <option value="">default</option>}
                {visibleModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                    {m.available ? ` · ${m.dim}d` : " (нет весов)"}
                  </option>
                ))}
              </select>
            </label>
            <button type="button" className="primary" onClick={runSearch} disabled={!canSearch}>
              {searching ? "Поиск…" : "Найти"}
            </button>
          </div>

          {!bbox && file && (
            <p className="hint">Задайте BBox: протяните мышью или нажмите «Весь кадр».</p>
          )}
          {searchError && <p className="error">Ошибка запроса: {searchError}</p>}
        </section>

        <section className="panel panel--crop">
          <h2>
            2. Кроп запроса и Grad-CAM{" "}
            {explaining && <span className="spinner">анализ…</span>}
          </h2>
          {objectUrl && bbox && naturalSize ? (
            <CropPreview
              src={objectUrl}
              naturalSize={naturalSize}
              bbox={bbox}
              overlayUrl={explain?.overlayUrl ?? null}
              heatmapUrl={explain?.heatmapUrl ?? null}
              method={explainMethod}
            />
          ) : (
            <p className="empty">Выберите изображение и BBox запроса.</p>
          )}
          {explainError && (
            <p className="warn">
              Explain недоступен: {explainError}. Ранжирование ниже получено.
            </p>
          )}
        </section>

        <section className="panel panel--results">
          <h2>3. Результат</h2>
          {result ? (
            <>
              <div
                className={`status-pill ${
                  result.refused ? "status-pill--refused" : "status-pill--ok"
                }`}
              >
                {result.refused
                  ? `ОТКАЗ: ${result.model} не находит совпадение выше порога ${result.threshold.toFixed(4)}`
                  : "Совпадение принято"}
              </div>
              <ResultsTable
                candidates={result.candidates}
                threshold={result.threshold}
                confidence={result.confidence}
                model={result.model}
                latencyMs={result.latencyMs}
              />
              <ExportBar result={result} />
            </>
          ) : (
            <p className="empty">
              {searching ? "Идёт поиск…" : "Здесь появятся кандидаты после запроса."}
            </p>
          )}
        </section>
      </main>
    </div>
  );
}

function HealthBadge({ health }: { health: HealthState }) {
  if (health.kind === "checking") {
    return <span className="health health--checking">проверка API…</span>;
  }
  if (health.kind === "down") {
    return <span className="health health--down">API offline</span>;
  }
  const parts = ["API online"];
  if (health.engine) parts.push(health.engine);
  if (health.gallerySize !== null) parts.push(`gallery ${health.gallerySize}`);
  return <span className="health health--ok">{parts.join(" · ")}</span>;
}
