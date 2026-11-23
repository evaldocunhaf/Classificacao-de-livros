"""Gera uma auditoria rastreável de um dataset CDD e de seus splits."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


def _summary(df: pd.DataFrame) -> dict:
    report: dict[str, object] = {
        "registros": len(df),
        "classes_cdd": int(df["cdd"].nunique()) if "cdd" in df else 0,
    }
    if "has_resumo" in df:
        present = df["has_resumo"].fillna("").astype(str).str.lower().eq("true")
    elif "resumo" in df:
        present = df["resumo"].fillna("").astype(str).str.strip().ne("")
    else:
        present = pd.Series(False, index=df.index)
    report["com_resumo"] = int(present.sum())
    report["sem_resumo"] = int((~present).sum())
    for column in ("source", "idioma"):
        if column in df:
            report[f"por_{column}"] = df[column].fillna("").value_counts().to_dict()
    return report


def _duplicate_report(df: pd.DataFrame) -> dict:
    if "edition_id" not in df:
        return {"edition_id_disponivel": False}
    edition = df["edition_id"].fillna("").astype(str)
    valid = edition.ne("")
    counts = edition[valid].value_counts()
    repeated_ids = counts[counts > 1].index
    repeated_rows = df[edition.isin(repeated_ids)]
    multi_cdd = 0
    if "cdd" in repeated_rows:
        multi_cdd = int((repeated_rows.groupby("edition_id")["cdd"].nunique() > 1).sum())
    return {
        "edition_id_disponivel": True,
        "edicoes_distintas": int(edition[valid].nunique()),
        "linhas_em_edicoes_repetidas": int(len(repeated_rows)),
        "edicoes_repetidas": int(len(repeated_ids)),
        "edicoes_com_mais_de_uma_cdd": multi_cdd,
    }


def audit(prepared_dir: Path) -> tuple[dict, pd.DataFrame]:
    names = ("train", "val", "test")
    splits: dict[str, pd.DataFrame] = {}
    for name in names:
        path = prepared_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"split não encontrado: {path}")
        splits[name] = pd.read_csv(path, dtype=str)

    combined = pd.concat(
        [frame.assign(split=name) for name, frame in splits.items()], ignore_index=True
    )
    group_column = "work_id" if "work_id" in combined else None
    overlaps = {}
    if group_column:
        group_sets = {
            name: set(frame[group_column].fillna("").astype(str)) - {""}
            for name, frame in splits.items()
        }
        overlaps = {
            f"{left}_{right}": len(group_sets[left] & group_sets[right])
            for left, right in combinations(names, 2)
        }

    distribution = (
        combined.groupby(["split", "cdd"], dropna=False).size().rename("n").reset_index()
        if "cdd" in combined
        else pd.DataFrame(columns=["split", "cdd", "n"])
    )
    report = {
        "prepared_dir": str(prepared_dir),
        "por_split": {name: _summary(frame) for name, frame in splits.items()},
        "total": _summary(combined),
        "duplicatas": _duplicate_report(combined),
        "intersecao_work_id": overlaps if group_column else "coluna work_id ausente",
        "classes_ausentes_por_split": {
            name: sorted(set(combined["cdd"]) - set(frame["cdd"]))
            for name, frame in splits.items()
        }
        if "cdd" in combined
        else {},
    }
    return report, distribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, distribution = audit(args.prepared_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    distribution.to_csv(args.output_dir / "distribuicao_por_split.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
