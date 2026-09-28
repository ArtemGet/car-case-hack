// Wire types for service/api (FastAPI, auto-OpenAPI). The UI only talks to the
// same-origin `/api` and `/health` proxy configured in vite.config.ts — it never
// runs model logic and never contacts an external host.

/** Pixel BBox in the ORIGINAL frame (see service/api/schemas.py:BBox). */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** One ranked gallery candidate (schemas.py:Candidate). */
export interface Candidate {
  rank: number;
  galleryId: string;
  score: number;
  confidence: number;
  /**
   * Preview URL of the gallery crop, e.g. `/api/v1/gallery/{id}/image`
   * (same-origin, resolved through the `/api` proxy). Null when unavailable.
   */
  imageUrl: string | null;
}

/** POST /api/v1/search (schemas.py:SearchResponse). */
export interface SearchResponse {
  queryId: string | null;
  model: string;
  bbox: BBox;
  topK: number;
  candidates: Candidate[];
  confidence: number;
  threshold: number;
  accepted: boolean;
  refused: boolean;
  latencyMs: number;
  raw: unknown;
}

/** POST /api/v1/explain (schemas.py:ExplainResponse). Both are data: URLs. */
export interface ExplainResponse {
  model: string;
  bbox: BBox;
  method: string;
  heatmapUrl: string | null;
  overlayUrl: string | null;
  raw: unknown;
}

/** GET /health (schemas.py:HealthResponse). */
export interface HealthResponse {
  status: string;
  version: string | null;
  engine: string | null;
  gallerySize: number | null;
  galleryDim: number | null;
  modelsAvailable: number | null;
}

/** GET /api/v1/models (schemas.py:ModelInfo). */
export interface ModelInfo {
  name: string;
  variant: string;
  dim: number;
  description: string;
  available: boolean;
  isDefault: boolean;
}

export interface ApiError {
  message: string;
  status?: number;
  detail?: unknown;
}
