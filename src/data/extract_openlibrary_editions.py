"""Extrai um CSV de classificação CDD a partir do dump de edições Open Library.

O dump é um TSV de várias dezenas de GB. Este módulo o lê em streaming, uma
linha por vez, e nunca o carrega integralmente em memória.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

try:  # suporta `python -m src.data.extract_openlibrary_editions` e execução direta
    from .cdd import normalize_cdd, truncate_cdd
except ImportError:  # pragma: no cover - caminho de compatibilidade do CLI legado
    from cdd import normalize_cdd, truncate_cdd

BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "data" / "ol_dump_editions_2026-08-31.txt"
DEFAULT_OUTPUT = BASE / "output" / "datasets" / "openlibrary_editions.csv"
FIELDS = (
    "titulo",
    "resumo",
    "cdd",
    "cdd_raw",
    "source",
    "edition_id",
    "work_id",
    "isbn",
    "idioma",
)


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _keys(values: object) -> list[str]:
    return [
        item["key"]
        for item in _as_list(values)
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    ]


def _codes(values: object, max_digits: int | None = None) -> tuple[list[str], list[str]]:
    raw_values = [str(value).strip() for value in _as_list(values) if str(value).strip()]
    normalizer = (lambda value: truncate_cdd(value, max_digits)) if max_digits else normalize_cdd
    canonical = list(dict.fromkeys(filter(None, (normalizer(value) for value in raw_values))))
    return raw_values, canonical


def _matches_prefix(code: str, prefixes: tuple[str, ...]) -> bool:
    return not prefixes or any(code.startswith(prefix) for prefix in prefixes)


def iter_editions(
    input_path: Path,
    prefixes: tuple[str, ...] = (),
    max_lines: int | None = None,
    max_digits: int | None = None,
) -> Iterator[dict]:
    """Produz registros Open Library válidos a partir do TSV, em streaming."""
    with input_path.open("r", encoding="utf-8", errors="replace") as dump:
        for line_number, line in enumerate(dump, start=1):
            if max_lines is not None and line_number > max_lines:
                return
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5 or parts[0] != "/type/edition":
                continue
            # Evita desserializar a grande maioria das edições sem CDD.
            if '"dewey_decimal_class"' not in parts[4]:
                continue
            try:
                record = json.loads(parts[4])
            except json.JSONDecodeError:
                continue
            raw_codes, codes = _codes(record.get("dewey_decimal_class"), max_digits)
            if not codes:
                continue
            title = str(record.get("title") or "").strip()
            if not title:
                continue
            # Uma edição pode listar várias CDDs. O experimento é de rótulo
            # único, então preservamos todas em `cdd_raw` e adotamos apenas a
            # primeira notação válida que pertence ao recorte solicitado.
            # Assim uma edição nunca aparece em treino e teste sob rótulos
            # distintos por causa da expansão da lista.
            matches = [code for code in codes if _matches_prefix(code, prefixes)]
            if not matches:
                continue
            code = matches[0]
            work_keys = _keys(record.get("works"))
            yield {
                "titulo": title,
                "resumo": "",
                "cdd": code,
                "cdd_raw": " | ".join(raw_codes),
                "source": "openlibrary_editions_2026-08-31",
                "edition_id": record.get("key") or parts[1],
                "work_id": work_keys[0] if work_keys else "",
                "isbn": " | ".join(str(v) for v in _as_list(record.get("isbn_13") or record.get("isbn_10"))),
                "idioma": " | ".join(_keys(record.get("languages"))),
            }


def extract(
    input_path: Path,
    output_path: Path,
    prefixes: tuple[str, ...] = (),
    max_per_cdd: int | None = None,
    max_records: int | None = None,
    max_lines: int | None = None,
    max_digits: int | None = None,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"dump não encontrado: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        for row in iter_editions(input_path, prefixes, max_lines, max_digits):
            if max_records is not None and written >= max_records:
                break
            if max_per_cdd is not None and counts[row["cdd"]] >= max_per_cdd:
                continue
            writer.writerow(row)
            counts[row["cdd"]] += 1
            written += 1
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": written,
        "classes": len(counts),
        "por_cdd_top": dict(counts.most_common(25)),
        "por_cdd_truncado": len(counts) > 25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai CDD do dump de edições Open Library em streaming")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cdd-prefix", nargs="*", default=[], help="prefixos CDD, ex.: 000 004 005")
    parser.add_argument("--max-per-cdd", type=int, default=None, help="limite por código completo")
    parser.add_argument("--max-records", type=int, default=None, help="limite global de registros gerados")
    parser.add_argument("--max-lines", type=int, default=None, help="limita linhas lidas; útil para teste")
    parser.add_argument("--max-digits", type=int, default=None, help="limita a CDD por algarismos; ex.: 303.4833 vira 303.48 com 5")
    args = parser.parse_args()
    prefixes = tuple(str(prefix).strip() for prefix in args.cdd_prefix if str(prefix).strip())
    report = extract(
        args.input,
        args.output,
        prefixes,
        args.max_per_cdd,
        args.max_records,
        args.max_lines,
        args.max_digits,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
