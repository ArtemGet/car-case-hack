import type { SearchResponse } from "../types";

interface Props {
  result: SearchResponse;
}

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function csvField(value: string): string {
  return /[",\n;]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/** candidates.csv contract: query_id,gallery_id,confidence; refusal = no rows. */
function toCandidatesCsv(result: SearchResponse): string {
  const lines = ["query_id,gallery_id,confidence"];
  if (!result.refused) {
    const queryId = result.queryId ?? "query";
    for (const c of result.candidates) {
      lines.push(
        [csvField(queryId), csvField(c.galleryId), c.confidence.toString()].join(","),
      );
    }
  }
  return lines.join("\n") + "\n";
}

function baseName(result: SearchResponse): string {
  const id = (result.queryId ?? "query").replace(/[^\w.-]+/g, "_");
  return `${id}_${result.model}_top${result.candidates.length || 0}`;
}

export default function ExportBar({ result }: Props) {
  const onCsv = () =>
    download(
      `${baseName(result)}_candidates.csv`,
      toCandidatesCsv(result),
      "text/csv;charset=utf-8",
    );

  const onJson = () =>
    download(
      `${baseName(result)}_result.json`,
      JSON.stringify(
        {
          query_id: result.queryId,
          model: result.model,
          bbox: [result.bbox.x, result.bbox.y, result.bbox.w, result.bbox.h],
          accepted: result.accepted,
          refused: result.refused,
          confidence: result.confidence,
          threshold: result.threshold,
          latency_ms: result.latencyMs,
          candidates: result.candidates.map((c) => ({
            rank: c.rank,
            gallery_id: c.galleryId,
            score: c.score,
            confidence: c.confidence,
          })),
        },
        null,
        2,
      ),
      "application/json",
    );

  return (
    <div className="export-bar">
      <button
        type="button"
        onClick={onCsv}
        disabled={result.refused || result.candidates.length === 0}
        title={result.refused ? "Отказ: строк для query нет (по контракту)" : undefined}
      >
        Экспорт CSV (candidates)
      </button>
      <button type="button" onClick={onJson}>
        Экспорт JSON
      </button>
    </div>
  );
}
