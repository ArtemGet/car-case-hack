import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import type { BBox } from "../types";

export interface NaturalSize {
  w: number;
  h: number;
}

interface Props {
  src: string;
  naturalSize: NaturalSize | null;
  onNaturalSize: (size: NaturalSize) => void;
  bbox: BBox | null;
  onBboxChange: (bbox: BBox) => void;
  disabled?: boolean;
}

interface Point {
  x: number;
  y: number;
}

interface DragState {
  start: Point;
  current: Point;
}

/** Clamp a rect to the image bounds and drop degenerate selections. */
function clampRect(rect: BBox, size: NaturalSize): BBox | null {
  const x1 = Math.max(0, Math.min(rect.x, size.w));
  const y1 = Math.max(0, Math.min(rect.y, size.h));
  const x2 = Math.max(0, Math.min(rect.x + rect.w, size.w));
  const y2 = Math.max(0, Math.min(rect.y + rect.h, size.h));
  const w = Math.round(x2 - x1);
  const h = Math.round(y2 - y1);
  if (w < 4 || h < 4) return null;
  return { x: Math.round(x1), y: Math.round(y1), w, h };
}

function rectFromDrag(drag: DragState): BBox {
  return {
    x: Math.min(drag.start.x, drag.current.x),
    y: Math.min(drag.start.y, drag.current.y),
    w: Math.abs(drag.current.x - drag.start.x),
    h: Math.abs(drag.current.y - drag.start.y),
  };
}

/**
 * Interactive frame: draws the query BBox over the uploaded image.
 *
 * The main interaction is a mouse/touch drag over the frame (a dashed live
 * rectangle follows the pointer, then commits on pointer-up). Numeric x/y/w/h
 * fields stay available as a secondary, exact method.
 *
 * Coordinates are mapped against the *rendered image element* rect (not the
 * padded stage), so letterboxing never offsets the selection; the drawn
 * rectangle is positioned in the same frame element, keeping them aligned.
 */
export default function ImageCanvas({
  src,
  naturalSize,
  onNaturalSize,
  bbox,
  onBboxChange,
  disabled = false,
}: Props) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [display, setDisplay] = useState<NaturalSize>({ w: 0, h: 0 });

  // Track the on-screen size of the image frame so image pixels map 1:1 with
  // the rendered rectangle (and survive responsive resizes).
  useEffect(() => {
    const el = frameRef.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      setDisplay({ w: rect.width, h: rect.height });
    };
    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [src, naturalSize]);

  const toImagePoint = useCallback(
    (clientX: number, clientY: number): Point | null => {
      const el = frameRef.current;
      if (!el || !naturalSize) return null;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return null;
      return {
        x: ((clientX - rect.left) / rect.width) * naturalSize.w,
        y: ((clientY - rect.top) / rect.height) * naturalSize.h,
      };
    },
    [naturalSize],
  );

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled || !naturalSize) return;
    const p = toImagePoint(event.clientX, event.clientY);
    if (!p) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ start: p, current: p });
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag) return;
    const p = toImagePoint(event.clientX, event.clientY);
    if (!p) return;
    setDrag((d) => (d ? { ...d, current: p } : d));
  };

  const commitDrag = useCallback(
    (d: DragState | null) => {
      if (!d) return;
      if (naturalSize) {
        // A tiny click (no real drag) keeps the current selection untouched.
        const clamped = clampRect(rectFromDrag(d), naturalSize);
        if (clamped) onBboxChange(clamped);
      }
      setDrag(null);
    },
    [naturalSize, onBboxChange],
  );

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    commitDrag(drag);
  };

  const onPointerCancel = () => commitDrag(drag);

  const draft = useMemo(() => (drag ? rectFromDrag(drag) : null), [drag]);
  const shown = draft ?? bbox;

  const scale =
    naturalSize && display.w > 0 ? display.w / naturalSize.w : 1;
  const overlayReady = Boolean(shown && naturalSize && display.w > 0);

  const update = (key: keyof BBox, value: number) => {
    if (!bbox || !naturalSize) return;
    const next = { ...bbox, [key]: Math.max(0, Math.round(value || 0)) };
    const clamped = clampRect(next, naturalSize);
    onBboxChange(clamped ?? next);
  };

  return (
    <div className="canvas-block">
      <div className="canvas-stage">
        <div
          ref={frameRef}
          className={`canvas-frame${disabled ? " canvas-frame--disabled" : ""}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerCancel}
          role="application"
          aria-label="Область изображения: протяните мышью или пальцем, чтобы задать BBox"
        >
          <img
            src={src}
            alt="Запрос"
            draggable={false}
            onLoad={(e) =>
              onNaturalSize({
                w: e.currentTarget.naturalWidth,
                h: e.currentTarget.naturalHeight,
              })
            }
          />
          {overlayReady && (
            <div
              className={`bbox-rect${draft ? " bbox-rect--draft" : ""}`}
              style={{
                left: shown!.x * scale,
                top: shown!.y * scale,
                width: shown!.w * scale,
                height: shown!.h * scale,
              }}
            />
          )}
        </div>
        <div className="canvas-hint">
          {naturalSize
            ? "Протяните мышью по кузову, чтобы задать BBox"
            : "Загрузка изображения…"}
        </div>
      </div>

      {naturalSize && (
        <div className="bbox-inputs">
          <span className="bbox-inputs__title">BBox (px)</span>
          {(["x", "y", "w", "h"] as const).map((key) => (
            <label key={key}>
              {key.toUpperCase()}
              <input
                type="number"
                min={0}
                value={bbox ? bbox[key] : 0}
                disabled={disabled || !bbox}
                onChange={(e) => update(key, Number(e.target.value))}
              />
            </label>
          ))}
          <span className="bbox-inputs__dim">
            кадр {naturalSize.w}×{naturalSize.h}
          </span>
        </div>
      )}
    </div>
  );
}
