"""Валидатор формата трёх файлов сдачи (контракт INTERFACES.md §2).

Проверяет артефакты в каталоге `--out`:
  submission.csv   — все query, ровно 10 кандидатов (если галерея >= 10),
                     нет дублей, все gallery_id из галереи, нет пустых;
  embeddings.npy   — 2D, float32, ровно len(query)+len(gallery) строк;
  candidates.csv   — заголовок query_id,gallery_id,confidence, confidence
                     парсится во float, пустые gallery_id недопустимы.
                     Отказ кодируется ОТСУТСТВИЕМ строк (не ошибка).

Exit code: 1 при любой ошибке (error), иначе 0. Warnings не влияют на код.

Запуск:
    python tools/validate_format.py --out <dir> --query <test_query.csv> \
        --gallery <test_gallery.csv>
    python tools/validate_format.py ... --json      # машинный отчёт в stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

CAND_HEADER = ["query_id", "gallery_id", "confidence"]
TOP_K = 10


def read_ids(path):
    """image_id из CSV (первая колонка), заголовок 'image_id' пропускается."""
    ids = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            v = row[0].strip()
            if not v or v.lower() == "image_id":
                continue
            ids.append(v)
    return ids


def _submission_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for vals in csv.reader(f):
            vals = [v.strip() for v in vals]
            if not vals or not any(vals):
                continue
            if vals[0].lower() == "query_id":
                continue
            yield vals


def validate(out_dir, query_csv, gallery_csv, top_k=TOP_K):
    errors, warnings, checks = [], [], []

    def ok(name, detail=""):
        checks.append({"level": "ok", "name": name, "detail": detail})

    def warn(name, detail=""):
        warnings.append(f"{name}: {detail}")
        checks.append({"level": "warning", "name": name, "detail": detail})

    def err(name, detail=""):
        errors.append(f"{name}: {detail}")
        checks.append({"level": "error", "name": name, "detail": detail})

    def add(level, name, detail):
        return err(name, detail) if level == "error" else warn(name, detail)

    if not os.path.isdir(out_dir):
        err("out_dir", f"каталог не найден: {out_dir}")
        return _report(errors, warnings, checks)

    query_ids = read_ids(query_csv)
    gallery_ids = read_ids(gallery_csv)
    q_set, g_set = set(query_ids), set(gallery_ids)

    # ---------------- submission.csv ----------------
    sub_path = os.path.join(out_dir, "submission.csv")
    if not os.path.exists(sub_path):
        err("submission.csv", "файл отсутствует")
    else:
        seen_q = {}
        present_q = set()
        for vals in _submission_rows(sub_path):
            qid, preds = vals[0], vals[1:]
            present_q.add(qid)
            if qid in seen_q:
                err("submission.csv", f"query '{qid}' встречается несколько раз")
            seen_q[qid] = preds

            clean = [p for p in preds if p]
            if len(clean) != len(preds):
                err("submission.csv", f"query '{qid}': пустые gallery_id")
            dups = {p for p in clean if clean.count(p) > 1}
            if dups:
                err("submission.csv", f"query '{qid}': дубли {sorted(dups)}")
            unknown = [p for p in clean if p not in g_set]
            if unknown:
                err("submission.csv", f"query '{qid}': неизвестные gallery_id {unknown}")

            expected = top_k if len(gallery_ids) >= top_k else len(gallery_ids)
            if len(gallery_ids) >= top_k:
                if len(clean) != top_k:
                    err("submission.csv",
                        f"query '{qid}': {len(clean)} кандидатов, ожидалось {top_k}")
            else:
                if len(clean) != len(gallery_ids):
                    warn("submission.csv",
                         f"query '{qid}': {len(clean)} кандидатов при галерее "
                         f"{len(gallery_ids)} (<{top_k})")
            _ = expected

        if not seen_q:
            err("submission.csv", "нет ни одной строки")
        missing = q_set - present_q
        if missing:
            err("submission.csv",
                f"отсутствуют query: {len(missing)} шт (например {sorted(missing)[:3]})")
        extra = present_q - q_set
        if extra:
            warn("submission.csv",
                 f"лишние query_id, которых нет в test_query: {len(extra)} шт")
        if not missing and seen_q:
            ok("submission.csv", f"все {len(query_ids)} query, "
                                 f"{len(gallery_ids)} gallery")

    # ---------------- embeddings.npy ----------------
    emb_path = os.path.join(out_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        err("embeddings.npy", "файл отсутствует")
    else:
        try:
            emb = np.load(emb_path)
        except Exception as exc:  # noqa: BLE001
            err("embeddings.npy", f"не читается: {exc}")
        else:
            n_expected = len(query_ids) + len(gallery_ids)
            if emb.ndim != 2:
                err("embeddings.npy", f"должен быть 2D, получено shape={emb.shape}")
            if emb.dtype != np.float32:
                warn("embeddings.npy", f"dtype={emb.dtype}, ожидался float32")
            if emb.shape[0] != n_expected:
                err("embeddings.npy",
                    f"{emb.shape[0]} строк, ожидалось {len(query_ids)} (query) + "
                    f"{len(gallery_ids)} (gallery) = {n_expected}")
            elif emb.ndim == 2 and emb.dtype == np.float32:
                ok("embeddings.npy", f"shape={tuple(emb.shape)} float32")

    # ---------------- candidates.csv ----------------
    cand_path = os.path.join(out_dir, "candidates.csv")
    if not os.path.exists(cand_path):
        err("candidates.csv", "файл отсутствует")
    else:
        with open(cand_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        header = [c.strip() for c in rows[0]] if rows else []
        if [c.lower() for c in header] != CAND_HEADER:
            err("candidates.csv", f"заголовок {header}, ожидался {CAND_HEADER}")
        n_rows = 0
        n_bad_conf = 0
        n_empty_gid = 0
        cand_q = set()
        for vals in rows[1:]:
            vals = [v.strip() for v in vals]
            if not vals or not any(vals):
                continue
            n_rows += 1
            if len(vals) < 2 or not vals[1]:
                n_empty_gid += 1
                continue
            cand_q.add(vals[0])
            if len(vals) < 3:
                n_bad_conf += 1
                continue
            try:
                float(vals[2])
            except ValueError:
                n_bad_conf += 1
        if n_empty_gid:
            err("candidates.csv", f"пустых gallery_id: {n_empty_gid} шт")
        if n_bad_conf:
            err("candidates.csv", f"confidence не парсится во float: {n_bad_conf} шт")
        if n_rows and not n_empty_gid and not n_bad_conf:
            refused = q_set - cand_q
            ok("candidates.csv",
               f"{n_rows} строк, {len(cand_q)} query с ответом, "
               f"{len(refused)} отказ(ов)")

    return _report(errors, warnings, checks)


def _report(errors, warnings, checks):
    return {
        "ok": not errors,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _print_human(rep, out_dir):
    print(f"validate_format: {out_dir}")
    for c in rep["checks"]:
        mark = {"ok": "[ok]  ", "warning": "[warn]", "error": "[ERR] "}[c["level"]]
        print(f"  {mark} {c['name']}: {c['detail']}")
    print(f"\nошибок: {rep['n_errors']}, предупреждений: {rep['n_warnings']}")
    print("РЕЗУЛЬТАТ:", "OK" if rep["ok"] else "FAIL")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Валидатор формата файлов сдачи (задача 7)")
    ap.add_argument("--out", required=True, help="каталог с 3 файлами сдачи")
    ap.add_argument("--query", required=True, help="test_query.csv")
    ap.add_argument("--gallery", required=True, help="test_gallery.csv")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--json", action="store_true", help="машинный отчёт в stdout")
    args = ap.parse_args(argv)

    for p in (args.query, args.gallery):
        if not os.path.exists(p):
            msg = f"входной файл не найден: {p}"
            if args.json:
                print(json.dumps({"ok": False, "errors": [msg]}, ensure_ascii=False))
            else:
                print(f"[ERR] {msg}")
            return 1

    rep = validate(args.out, args.query, args.gallery, top_k=args.top_k)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_human(rep, args.out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
