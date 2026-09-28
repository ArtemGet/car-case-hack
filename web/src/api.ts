// Thin, dependency-free client for service/api. All calls go through the
// same-origin `/api` and `/health` proxy (vite.config.ts in dev, nginx.conf in
// prod): no CORS, no direct internet access at runtime.
//
// The backend speaks JSON with the image sent as base64 (optionally a data URL),
// so the browser encodes the picked file and posts it — no python-multipart.

import type {
  ApiError,
  BBox,
  Candidate,
  ExplainResponse,
  HealthResponse,
  ModelInfo,
  SearchResponse,
} from "./types";

const API_BASE = "/api/v1";

/** Health lives at the server root (not under /api/v1). */
const HEALTH_PATH = "/health";

type Json = Record<string, unknown>;

function asObject(value: unknown): Json {
  return value && typeof value === "object" ? (value as Json) : {};
}

function pick(obj: Json, keys: string[]): unknown {
  for (const key of keys) {
    if (obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return undefined;
}

function toNumber(value: unknown, fallback = 0): number {
  const n = typeof value === "string" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
}

function toStringOrNull(value: unknown): string | null {
  return value === undefined || value === null ? null : String(value);
}

function normalizeBBox(value: unknown): BBox | null {
  const obj = asObject(value);
  if (Array.isArray(value) && value.length >= 4) {
    const [x, y, w, h] = value;
    return { x: toNumber(x), y: toNumber(y), w: toNumber(w), h: toNumber(h) };
  }
  if (
    obj.x !== undefined &&
    obj.y !== undefined &&
    obj.w !== undefined &&
    obj.h !== undefined
  ) {
    return {
      x: toNumber(obj.x),
      y: toNumber(obj.y),
      w: toNumber(obj.w),
      h: toNumber(obj.h),
    };
  }
  return null;
}

function normalizeCandidate(raw: unknown, index: number): Candidate {
  const obj = asObject(raw);
  return {
    rank: Math.round(toNumber(pick(obj, ["rank"]), index + 1)),
    galleryId: toStringOrNull(pick(obj, ["gallery_id", "galleryId", "id"])) ?? `#${index + 1}`,
    score: toNumber(pick(obj, ["score", "similarity"])),
    confidence: toNumber(pick(obj, ["confidence", "score"])),
    imageUrl: toStringOrNull(pick(obj, ["image_url", "imageUrl", "preview_url"])),
  };
}

function normalizeSearch(raw: unknown, fallbackBBox: BBox): SearchResponse {
  const obj = asObject(raw);
  const list = pick(obj, ["candidates", "results", "hits"]);
  const candidates = Array.isArray(list)
    ? list.map((c, i) => normalizeCandidate(c, i)).sort((a, b) => a.rank - b.rank)
    : [];
  const confidence = toNumber(pick(obj, ["confidence"]));
  const threshold = toNumber(pick(obj, ["threshold"]));
  const acceptedRaw = pick(obj, ["accepted"]);
  const refusedRaw = pick(obj, ["refused", "is_refusal"]);
  const accepted =
    typeof acceptedRaw === "boolean" ? acceptedRaw : !(refusedRaw === true);
  return {
    queryId: toStringOrNull(pick(obj, ["query_id", "queryId"])),
    model: toStringOrNull(pick(obj, ["model", "backend", "variant"])) ?? "unknown",
    bbox: normalizeBBox(pick(obj, ["bbox"])) ?? fallbackBBox,
    topK: Math.round(toNumber(pick(obj, ["top_k"]), candidates.length)),
    candidates,
    confidence,
    threshold,
    accepted,
    refused: typeof refusedRaw === "boolean" ? refusedRaw : !accepted,
    latencyMs: toNumber(pick(obj, ["latency_ms", "latencyMs"])),
    raw,
  };
}

function normalizeExplain(raw: unknown): ExplainResponse {
  const obj = asObject(raw);
  return {
    model: toStringOrNull(pick(obj, ["model", "backend"])) ?? "unknown",
    bbox: normalizeBBox(pick(obj, ["bbox"])) ?? { x: 0, y: 0, w: 0, h: 0 },
    method: toStringOrNull(pick(obj, ["method"])) ?? "explain",
    overlayUrl: toStringOrNull(
      pick(obj, ["overlay_png_base64", "overlay_png", "overlay_url"]),
    ),
    heatmapUrl: toStringOrNull(
      pick(obj, ["heatmap_png_base64", "heatmap_png", "heatmap_url"]),
    ),
    raw,
  };
}

function normalizeModels(raw: unknown): { def: string; models: ModelInfo[] } {
  const obj = asObject(raw);
  const list = pick(obj, ["models"]);
  const models: ModelInfo[] = Array.isArray(list)
    ? list.map((m) => {
        const o = asObject(m);
        return {
          name: toStringOrNull(o.name) ?? "unknown",
          variant: toStringOrNull(o.variant) ?? "unknown",
          dim: Math.round(toNumber(o.dim)),
          description: toStringOrNull(o.description) ?? "",
          available: o.available === true,
          isDefault: o.is_default === true || o.isDefault === true,
        };
      })
    : [];
  const def =
    toStringOrNull(pick(obj, ["default"])) ??
    models.find((m) => m.isDefault)?.name ??
    models[0]?.name ??
    "stub";
  return { def, models };
}

async function fail(response: Response): Promise<never> {
  let detail: unknown;
  try {
    detail = await response.json();
  } catch {
    try {
      detail = await response.text();
    } catch {
      detail = undefined;
    }
  }
  const message =
    (detail && typeof detail === "object" && "detail" in detail
      ? String((detail as { detail: unknown }).detail)
      : null) || `HTTP ${response.status} ${response.statusText}`;
  const err: ApiError = { message, status: response.status, detail };
  throw err;
}

/** Encode a picked file as a data URL the API accepts (prefix is stripped server-side). */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("не удалось прочитать файл"));
    reader.readAsDataURL(file);
  });
}

export function humanError(error: unknown): string {
  if (error && typeof error === "object" && "message" in error) {
    const { message, status } = error as ApiError;
    return status ? `${message} (HTTP ${status})` : String(message);
  }
  return error instanceof Error ? error.message : String(error);
}

export const api = {
  async health(): Promise<HealthResponse> {
    const response = await fetch(HEALTH_PATH, { headers: { Accept: "application/json" } });
    if (!response.ok) await fail(response);
    const obj = asObject(await response.json());
    const gallerySize = pick(obj, ["gallery_size"]);
    const galleryDim = pick(obj, ["gallery_dim"]);
    const modelsAvailable = pick(obj, ["models_available"]);
    return {
      status: toStringOrNull(pick(obj, ["status"])) ?? "ok",
      version: toStringOrNull(pick(obj, ["version"])),
      engine: toStringOrNull(pick(obj, ["engine", "backend"])),
      gallerySize: gallerySize === undefined ? null : toNumber(gallerySize),
      galleryDim: galleryDim === undefined ? null : toNumber(galleryDim),
      modelsAvailable: modelsAvailable === undefined ? null : toNumber(modelsAvailable),
    };
  },

  async models(): Promise<{ def: string; models: ModelInfo[] }> {
    const response = await fetch(`${API_BASE}/models`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) await fail(response);
    return normalizeModels(await response.json());
  },

  async search(params: {
    imageBase64: string;
    bbox: BBox;
    topK: number;
    model?: string | null;
  }): Promise<SearchResponse> {
    const response = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        image_base64: params.imageBase64,
        bbox: params.bbox,
        top_k: params.topK,
        model: params.model ?? null,
      }),
    });
    if (!response.ok) await fail(response);
    return normalizeSearch(await response.json(), params.bbox);
  },

  async explain(params: {
    imageBase64: string;
    bbox: BBox;
    model?: string | null;
  }): Promise<ExplainResponse> {
    const response = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        image_base64: params.imageBase64,
        bbox: params.bbox,
        model: params.model ?? null,
      }),
    });
    if (!response.ok) await fail(response);
    return normalizeExplain(await response.json());
  },
};
