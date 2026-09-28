import { useCallback, useEffect, useState } from "react";
import type { Candidate } from "../types";

interface Props {
  candidates: Candidate[];
  threshold: number;
  confidence: number;
  model: string;
  latencyMs: number;
}

type PreviewState = "idle" | "failed";

/** Large (160x110) clickable thumbnail for one candidate. */
function CardPreview({
  candidate,
  onOpen,
}: {
  candidate: Candidate;
  onOpen: (c: Candidate) => void;
}) {
  const [failed, setFailed] = useState<PreviewState>("idle");
  useEffect(() => setFailed("idle"), [candidate.imageUrl]);

  const clickable = Boolean(candidate.imageUrl) && failed === "idle";

  if (!clickable) {
    // No `image_url` from the backend, or the gallery frame failed to load —
    // show an honest placeholder with the rank instead of a broken image.
    return (
      <div className="cand-card__preview cand-card__preview--placeholder" title="Превью недоступно">
        <span className="cand-card__rank-big">#{candidate.rank}</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="cand-card__preview"
      onClick={() => onOpen(candidate)}
      title={`Открыть ${candidate.galleryId} в полном размере`}
      aria-label={`Увеличить превью ${candidate.galleryId}`}
    >
      <img
        src={candidate.imageUrl ?? undefined}
        alt={candidate.galleryId}
        loading="lazy"
        onError={() => setFailed("failed")}
      />
      <span className="cand-card__zoom" aria-hidden="true">
        Увеличить
      </span>
    </button>
  );
}

/** Full-size viewer over a dimmed backdrop. Esc / backdrop / ✕ close it. */
function Lightbox({ candidate, onClose }: { candidate: Candidate; onClose: () => void }) {
  const onKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );
  useEffect(() => {
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onKey]);

  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={`Превью ${candidate.galleryId}`}
      onClick={onClose}
    >
      <figure className="lightbox__frame" onClick={(e) => e.stopPropagation()}>
        <img
          className="lightbox__img"
          src={candidate.imageUrl ?? undefined}
          alt={candidate.galleryId}
        />
        <figcaption className="lightbox__meta">
          <span className="lightbox__rank">#{candidate.rank}</span>
          <span className="lightbox__gid" title={candidate.galleryId}>
            {candidate.galleryId}
          </span>
          <span className="lightbox__stat">score {candidate.score.toFixed(4)}</span>
          <span className="lightbox__stat">confidence {candidate.confidence.toFixed(4)}</span>
        </figcaption>
      </figure>
      <button type="button" className="lightbox__close" onClick={onClose} aria-label="Закрыть">
        ✕
      </button>
    </div>
  );
}

export default function ResultsTable({
  candidates,
  threshold,
  confidence,
  model,
  latencyMs,
}: Props) {
  const max = candidates.reduce((acc, c) => Math.max(acc, c.score), 0);
  const [lightbox, setLightbox] = useState<Candidate | null>(null);

  return (
    <section className="panel">
      <header className="panel__head">
        <h2>Ранжирование top-N</h2>
        <div className="meta-chips">
          <span className="chip">модель: {model}</span>
          <span className="chip">confidence: {confidence.toFixed(4)}</span>
          <span className="chip">порог отказа: {threshold.toFixed(4)}</span>
          <span className="chip">{latencyMs.toFixed(1)} мс</span>
        </div>
      </header>

      {candidates.length === 0 ? (
        <p className="empty">Кандидаты не возвращены.</p>
      ) : (
        <>
          <div className="cand-grid">
            {candidates.map((c) => {
              const pass = c.score >= threshold;
              const pct = max > 0 ? Math.max(2, Math.min(100, (c.score / max) * 100)) : 0;
              return (
                <article
                  key={`${c.galleryId}-${c.rank}`}
                  className={`cand-card${pass ? "" : " cand-card--low"}`}
                >
                  <div className="cand-card__head">
                    <span className="cand-card__rank">#{c.rank}</span>
                    <span
                      className={`cand-card__flag${
                        pass ? " cand-card__flag--ok" : " cand-card__flag--low"
                      }`}
                    >
                      {pass ? "выше порога" : "ниже порога"}
                    </span>
                  </div>

                  <CardPreview candidate={c} onOpen={setLightbox} />

                  <div className="cand-card__gid" title={c.galleryId}>
                    {c.galleryId}
                  </div>

                  <dl className="cand-card__stats">
                    <div className="cand-card__row">
                      <dt>score</dt>
                      <dd>{c.score.toFixed(4)}</dd>
                    </div>
                    <div className="cand-card__row">
                      <dt>confidence</dt>
                      <dd>{c.confidence.toFixed(4)}</dd>
                    </div>
                  </dl>

                  <div className="score__bar score__bar--wide">
                    <span
                      className={`score__fill${pass ? "" : " score__fill--low"}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </article>
              );
            })}
          </div>

          <div className="table-wrap table-wrap--compact">
            <table className="results">
              <thead>
                <tr>
                  <th>#</th>
                  <th>gallery_id</th>
                  <th>score</th>
                  <th>confidence</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => {
                  const pass = c.score >= threshold;
                  return (
                    <tr key={`row-${c.galleryId}-${c.rank}`}>
                      <td className="rank">{c.rank}</td>
                      <td className="gid" title={c.galleryId}>
                        {c.galleryId}
                      </td>
                      <td className={pass ? "num" : "num num--low"}>{c.score.toFixed(4)}</td>
                      <td className="num">{c.confidence.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {lightbox && lightbox.imageUrl && (
        <Lightbox candidate={lightbox} onClose={() => setLightbox(null)} />
      )}
    </section>
  );
}
