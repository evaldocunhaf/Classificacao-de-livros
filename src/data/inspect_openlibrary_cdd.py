"""Sondagem das CDDs disponíveis no dump de edições da Open Library.

Não cria um CSV: lê o TSV em streaming e conta as faixas solicitadas. Uma
edição pode corresponder a mais de uma CDD, mas nunca é repetida para a mesma
CDD truncada.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

try:
    from .cdd import truncate_cdd
except ImportError:  # pragma: no cover - compatibilidade com execução direta
    from cdd import truncate_cdd


BASE = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE / "data" / "ol_dump_editions_2026-08-31.txt"
DEFAULT_PREFIXES = ("001", "004", "005", "006", "025")


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def matching_cdds(record: dict[str, object], prefixes: set[str], max_digits: int) -> set[str]:
    """Obtém CDDs-alvo normalizadas e truncadas de uma edição."""
    matches = set()
    for raw in _as_list(record.get("dewey_decimal_class")):
        cdd = truncate_cdd(raw, max_digits)
        if cdd and cdd[:3] in prefixes:
            matches.add(cdd)
    return matches


def iter_dump_records(input_path: Path, max_lines: int | None = None) -> Iterator[dict[str, object]]:
    with input_path.open("r", encoding="utf-8", errors="replace") as dump:
        for line_number, line in enumerate(dump, start=1):
            if max_lines is not None and line_number > max_lines:
                return
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5 or parts[0] != "/type/edition":
                continue
            # A contagem só depende das edições que declaram alguma CDD.
            if '"dewey_decimal_class"' not in parts[4]:
                continue
            try:
                yield json.loads(parts[4])
            except json.JSONDecodeError:
                continue


def inspect(
    input_path: Path,
    prefixes: tuple[str, ...] | list[str] = DEFAULT_PREFIXES,
    max_digits: int = 5,
    max_lines: int | None = None,
) -> dict[str, object]:
    selected_prefixes = tuple(dict.fromkeys(prefixes))
    invalid = [prefix for prefix in selected_prefixes if len(prefix) != 3 or not prefix.isdigit()]
    if invalid:
        raise ValueError(f"CDD(s) de faixa inválida(s): {', '.join(invalid)}")

    selected_set = set(selected_prefixes)
    rows_per_prefix = Counter({prefix: 0 for prefix in selected_prefixes})
    cdd_counts: dict[str, Counter[str]] = defaultdict(Counter)
    editions_with_match = 0

    for record in iter_dump_records(input_path, max_lines):
        cdds = matching_cdds(record, selected_set, max_digits)
        if cdds:
            editions_with_match += 1
        for cdd in cdds:
            prefix = cdd[:3]
            rows_per_prefix[prefix] += 1
            cdd_counts[prefix][cdd] += 1

    return {
        "arquivo": str(input_path),
        "faixas_cdd": list(selected_prefixes),
        "max_digitos_cdd": max_digits,
        "linhas_lidas_limite": max_lines,
        "edicoes_com_ao_menos_uma_cdd_alvo": editions_with_match,
        "linhas_extraiveis_total": sum(rows_per_prefix.values()),
        "por_faixa": {
            prefix: {
                "linhas_extraiveis": rows_per_prefix[prefix],
                "classes_cdd_apos_truncamento": len(cdd_counts[prefix]),
                "top_cdds": cdd_counts[prefix].most_common(20),
            }
            for prefix in selected_prefixes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cdd-prefix", nargs="+", default=list(DEFAULT_PREFIXES))
    parser.add_argument("--max-digits", type=int, default=5)
    parser.add_argument("--max-lines", type=int, help="limita linhas do dump para uma sondagem curta")
    parser.add_argument("--output", type=Path, help="salva opcionalmente o relatório JSON")
    args = parser.parse_args()

    report = inspect(args.input, args.cdd_prefix, args.max_digits, args.max_lines)
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    print(formatted)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
