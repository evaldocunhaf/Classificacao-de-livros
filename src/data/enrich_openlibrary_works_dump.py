"""Enriquece um recorte CSV com os ``subjects`` do dump de obras Open Library.

O dump de Works é grande, portanto é percorrido uma única vez em streaming.
Somente os ``work_id`` que aparecem no CSV de entrada são mantidos em memória.
Os subjects não são tratados como gabarito: entram apenas como texto descritivo
de entrada para os classificadores e ficam identificados pela sua procedência.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


BASE = Path(__file__).resolve().parents[2]
DEFAULT_DUMP = BASE / "data" / "ol_dump_works_2026-09-30.txt.gz"


def _subjects(value: object) -> list[str]:
    """Converte a lista de subjects em termos únicos, preservando a ordem."""
    if not isinstance(value, list):
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("name") or item.get("value") or "").strip()
        else:
            text = ""
        normalized = text.casefold()
        if text and normalized not in seen:
            terms.append(text)
            seen.add(normalized)
    return terms


def iter_matching_subjects(dump_path: Path, work_ids: set[str]) -> Iterable[tuple[str, list[str]]]:
    """Produz ``(work_id, subjects)`` das obras pedidas sem carregar o dump."""
    found: set[str] = set()
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as dump:
        for line in dump:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5 or parts[0] != "/type/work":
                continue
            work_id = parts[1].strip()
            if work_id not in work_ids or work_id in found:
                continue
            try:
                record: dict[str, Any] = json.loads(parts[4])
            except json.JSONDecodeError:
                continue
            found.add(work_id)
            yield work_id, _subjects(record.get("subjects"))
            if len(found) == len(work_ids):
                return


def enrich_from_dump(input_path: Path, dump_path: Path, output_path: Path) -> dict[str, int | str]:
    if not dump_path.is_file():
        raise FileNotFoundError(f"dump de Works não encontrado: {dump_path}")
    with input_path.open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file))
        fieldnames = list(rows[0].keys()) if rows else []
    if "work_id" not in fieldnames:
        raise ValueError("CSV de entrada não tem coluna work_id")

    for column in ("subjects", "resumo_source"):
        if column not in fieldnames:
            fieldnames.append(column)

    work_ids = {row.get("work_id", "").strip() for row in rows if row.get("work_id", "").strip()}
    subjects_by_work = dict(iter_matching_subjects(dump_path, work_ids))
    enriched = 0
    works_with_subjects = 0
    for row in rows:
        subjects = subjects_by_work.get(row.get("work_id", "").strip(), [])
        joined = "\n".join(subjects)
        row["subjects"] = json.dumps(subjects, ensure_ascii=False)
        if subjects:
            works_with_subjects += 1
        # O campo resumo é o texto descritivo que o preparador já conhece. Não
        # apagamos uma descrição preexistente; se houver, acrescentamos os
        # subjects como contexto adicional e registramos ambas as fontes.
        original = row.get("resumo", "").strip()
        if joined:
            row["resumo"] = f"{original}\n{joined}".strip() if original else joined
            source = row.get("resumo_source", "").strip()
            row["resumo_source"] = "+".join(filter(None, (source, "openlibrary_work_subjects")))
            enriched += 1
        else:
            row.setdefault("resumo", original)
            row.setdefault("resumo_source", row.get("resumo_source", ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "input": str(input_path),
        "dump": str(dump_path),
        "output": str(output_path),
        "linhas": len(rows),
        "obras_solicitadas": len(work_ids),
        "obras_encontradas": len(subjects_by_work),
        "obras_com_subjects": works_with_subjects,
        "linhas_enriquecidas": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(enrich_from_dump(args.input, args.dump, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
