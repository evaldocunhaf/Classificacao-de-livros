"""Recalcula métricas de artefatos existentes sem sobrescrever a rodada original."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import save_eval


def recompute(input_csv: Path, output_dir: Path) -> dict:
    if not input_csv.is_file():
        raise FileNotFoundError(f"resultado não encontrado: {input_csv}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.read_csv(input_csv, dtype=str)
    out_base = output_dir / f"avaliacao_{input_csv.stem}_corrigida"
    report = save_eval(result, str(out_base))
    return {"input": str(input_csv), "output": str(out_base), "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="CSVs de predições existentes")
    parser.add_argument("--output-dir", type=Path, required=True, help="novo diretório para os artefatos corrigidos")
    args = parser.parse_args()
    reports = [recompute(path, args.output_dir) for path in args.input]
    (args.output_dir / "resumo_reavaliacao.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for item in reports:
        full = item["report"]["accuracy_por_nivel"]["full"]["accuracy"]
        print(f"{Path(item['input']).name}: full={full:.4f}")


if __name__ == "__main__":
    main()
