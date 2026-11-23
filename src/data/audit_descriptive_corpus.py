"""Mede se um corpus possui texto descritivo suficiente para testar keywords."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_descriptive_corpus(input_csv: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(input_csv, dtype=str)
    required = {"titulo", "cdd", "resumo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(sorted(missing))}")
    summary = df["resumo"].fillna("").astype(str).str.strip()
    descriptive = df[summary.ne("")].copy()
    counts = descriptive.groupby("cdd").size().rename("registros_com_resumo").reset_index()
    counts = counts.sort_values(["registros_com_resumo", "cdd"], ascending=[False, True]).reset_index(drop=True)
    thresholds = {
        str(threshold): {
            "classes": int((counts["registros_com_resumo"] >= threshold).sum()),
            "registros": int(counts.loc[counts["registros_com_resumo"] >= threshold, "registros_com_resumo"].sum()),
        }
        for threshold in (1, 2, 3, 5, 10)
    }
    report = {
        "input": str(input_csv),
        "registros_total": len(df),
        "registros_com_resumo": len(descriptive),
        "registros_sem_resumo": len(df) - len(descriptive),
        "classes_com_resumo": int(descriptive["cdd"].nunique()),
        "limiares_por_classe": thresholds,
    }
    return report, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, counts = audit_descriptive_corpus(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "descriptive_corpus_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts.to_csv(args.output_dir / "resumos_por_cdd.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
