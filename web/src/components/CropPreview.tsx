import { useMemo, useState } from "react";
import type { BBox } from "../types";
import type { NaturalSize } from "./ImageCanvas";

interface Props {
  /** Full query frame (local object URL). */
  src: string;
  naturalSize: NaturalSize | null;
  bbox: BBox | null;
  /** Backend-composited crop + Grad-CAM (data URL from /api/v1/explain). */
  overlayUrl: string | null;
  /** Raw heat-map, crop-sized (data URL). */
  heatmapUrl: string | null;
  method: string | null;
  maxWidth?: number;
}

/**
 * Crop of the query as the model sees it (BBox applied locally) with the
 * backend Grad-CAM/attention overlay shown on top. If the backend already
 * composited crop+map (overlay_png_base64), that image is used directly; a raw
 * heat-map is blended in the browser as a fallback.
 */
export default function CropPreview({
  src,
  naturalSize,
  bbox,
  overlayUrl,
  heatmapUrl,
  method,
  maxWidth = 300,
}: Props) {
  const [showMap, setShowMap] = useState(true);
  const [opacity, setOpacity] = useState(0.75);

  const geometry = useMemo(() => {
    if (!bbox || !naturalSize || bbox.w <= 0 || bbox.h <= 0) return null;
    const maxHeight = 360;
    const scale = Math.min(maxWidth / bbox.w, maxHeight / bbox.h);
    return {
      outWidth: Math.max(1, Math.round(bbox.w * scale)),
      height: Math.max(1, Math.round(bbox.h * scale)),
      innerWidth: naturalSize.w * scale,
      left: -bbox.x * scale,
      top: -bbox.y * scale,
    };
  }, [bbox, naturalSize, maxWidth]);

  if (!geometry || !bbox || !naturalSize) {
    return (
      <div className="crop-preview crop-preview--empty">
        Кроп появится после выбора BBox
      </div>
    );
  }

  // Prefer the server-composited overlay (already crop-sized); else blend raw map.
  const composite = overlayUrl;
  const rawHeat = !overlayUrl ? heatmapUrl : null;
  const hasMap = Boolean(composite || rawHeat);

  return (
    <div className="crop-preview">
      <div
        className="crop-preview__frame"
        style={{ width: geometry.outWidth, height: geometry.height }}
      >
        <img
          className="crop-preview__source"
          src={src}
          alt="Кроп запроса"
          draggable={false}
          style={{
            width: geometry.innerWidth,
            left: geometry.left,
            top: geometry.top,
          }}
        />
        {hasMap && showMap && (
          <img
            className="crop-preview__overlay"
            src={(composite || rawHeat) as string}
            alt="Grad-CAM"
            draggable={false}
            style={
              composite
                ? { left: 0, top: 0, width: "100%", height: "100%", opacity }
                : {
                    left: 0,
                    top: 0,
                    width: "100%",
                    height: "100%",
                    opacity,
                    mixBlendMode: "screen",
                  }
            }
          />
        )}
        <span className="crop-preview__tag">{method ? `кроп · ${method}` : "кроп запроса"}</span>
      </div>

      {hasMap && (
        <div className="crop-preview__controls">
          <label className="switch">
            <input
              type="checkbox"
              checked={showMap}
              onChange={(e) => setShowMap(e.target.checked)}
            />
            heat-map
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            aria-label="Прозрачность карты"
          />
        </div>
      )}
      {!hasMap && (
        <p className="crop-preview__note">
          Карта не пришла от <code>/api/v1/explain</code> — показан только кроп.
        </p>
      )}
    </div>
  );
}
