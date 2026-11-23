"""Enriquece registros Open Library com descrição e excertos da obra.

Destinado a pilotos pela API pública. Para milhares de obras, prefira o dump
de obras da Open Library; o cache permite interromper e retomar sem repetir
requisições.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests


BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "output" / "runs" / "2026-09-21-openlibrary-five-cdd" / "prepared" / "dataset_limpo.csv"


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("value") or "").strip()
    return ""


def work_summary(work: dict[str, Any]) -> tuple[str, str]:
    """Prioriza description; usa os excerpts como alternativa rastreável."""
    description = _text(work.get("description"))
    if description:
        return description, "openlibrary_work_description"

    excerpts = work.get("excerpts")
    if isinstance(excerpts, list):
        parts = [_text(item.get("excerpt")) for item in excerpts if isinstance(item, dict)]
        text = "\n".join(part for part in parts if part)
        if text:
            return text, "openlibrary_work_excerpts"
    return "", ""


def load_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, str]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        work_id = item.get("work_id")
        if isinstance(work_id, str):
            cache[work_id] = item
    return cache


def append_cache(cache_path: Path, item: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_file:
        cache_file.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_work(work_id: str, session: requests.Session, timeout: float) -> dict[str, str]:
    url = f"https://openlibrary.org{work_id}.json"
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code == 404:
            return {"work_id": work_id, "status": "not_found", "resumo": "", "resumo_source": ""}
        response.raise_for_status()
        summary, source = work_summary(response.json())
        return {"work_id": work_id, "status": "ok", "resumo": summary, "resumo_source": source}
    except requests.RequestException as error:
        return {"work_id": work_id, "status": f"error:{type(error).__name__}", "resumo": "", "resumo_source": ""}


def enrich(
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    max_works: int | None = 300,
    delay_seconds: float = 0.2,
    timeout: float = 20.0,
) -> dict[str, int | str | None]:
    with input_path.open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file))
        fieldnames = list(rows[0].keys()) if rows else []
    if "work_id" not in fieldnames:
        raise ValueError("CSV de entrada não tem coluna work_id")

    for column in ("resumo_source",):
        if column not in fieldnames:
            fieldnames.append(column)

    cache = load_cache(cache_path)
    work_ids = list(dict.fromkeys(row.get("work_id", "").strip() for row in rows if row.get("work_id", "").strip()))
    pending = [work_id for work_id in work_ids if work_id not in cache]
    if max_works is not None:
        pending = pending[:max_works]

    with requests.Session() as session:
        session.headers.update({"User-Agent": "tcc-cdd-research/1.0 (Open Library enrichment pilot)"})
        for index, work_id in enumerate(pending, start=1):
            item = fetch_work(work_id, session, timeout)
            cache[work_id] = item
            append_cache(cache_path, item)
            if delay_seconds and index < len(pending):
                time.sleep(delay_seconds)

    enriched = 0
    for row in rows:
        item = cache.get(row.get("work_id", "").strip(), {})
        summary = item.get("resumo", "")
        if summary and not row.get("resumo", "").strip():
            row["resumo"] = summary
            row["resumo_source"] = item.get("resumo_source", "")
            enriched += 1
        else:
            row.setdefault("resumo_source", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "cache": str(cache_path),
        "obras_unicas": len(work_ids),
        "consultadas_nesta_execucao": len(pending),
        "linhas_enriquecidas": enriched,
        "max_works": max_works,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-works", type=int, default=300, help="padrão seguro para a API pública; use 0 para nenhuma")
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    max_works = None if args.max_works < 0 else args.max_works
    report = enrich(args.input, args.output, args.cache, max_works, args.delay_seconds, args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
