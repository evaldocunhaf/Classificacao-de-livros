import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:  # suporta `python -m src.data.prepare_dataset` e execução direta legada
    from .cdd import normalize_cdd
except ImportError:  # pragma: no cover - caminho de compatibilidade do CLI legado
    from cdd import normalize_cdd



def strip_author(title: str) -> str:
    return re.split(r"/", title, maxsplit=1)[0].strip()


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"titulo", "cdd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(sorted(missing))}")
    if "resumo" not in df:
        df["resumo"] = ""
    df["cdd"] = df["cdd"].map(normalize_cdd)
    df["titulo"] = df["titulo"].fillna("").astype(str).str.strip()
    df["titulo"] = df["titulo"].map(strip_author)
    df["resumo"] = df["resumo"].fillna("").astype(str).str.strip()
    df["texto"] = df["titulo"]
    mask = df["resumo"].str.len() > 0
    df.loc[mask, "texto"] = df.loc[mask, "titulo"] + "\n" + df.loc[mask, "resumo"]
    df["has_resumo"] = df["resumo"].str.len() > 0
    df = df[df["cdd"] != ""]
    df = df[~df["titulo"].eq("")]
    return df.reset_index(drop=True)


def _split_grouped(df: pd.DataFrame, group_column: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Faz split 60/20/20 estratificado sem separar edições da mesma obra."""
    groups = df[group_column].fillna("").astype(str)
    groups = groups.where(groups.str.len() > 0, df.index.astype(str))
    test_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    main_index, test_index = next(test_split.split(df, y=df["cdd"], groups=groups))
    main, test = df.iloc[main_index], df.iloc[test_index]
    main_groups = groups.iloc[main_index]
    val_split = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    train_index, val_index = next(val_split.split(main, y=main["cdd"], groups=main_groups))
    return main.iloc[train_index], main.iloc[val_index], test


def filter_and_cap_classes(
    df: pd.DataFrame,
    min_per_class: int | None,
    max_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Remove rótulos raros e limita os majoritários para uma rodada viável."""
    if min_per_class is not None:
        if min_per_class < 1:
            raise ValueError("min_per_class deve ser positivo")
        counts = df["cdd"].value_counts()
        df = df[df["cdd"].map(counts) >= min_per_class]

    if max_per_class is not None:
        if max_per_class < 1:
            raise ValueError("max_per_class deve ser positivo")
        pieces = [
            group.sample(n=min(len(group), max_per_class), random_state=random_state)
            for _, group in df.groupby("cdd", sort=True)
        ]
        df = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0]

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_dataset(
    input_csv: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    group_column: str | None = None,
    min_per_class: int | None = None,
    max_per_class: int | None = None,
    allowed_cdds_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv, dtype=str)
    df = load_and_clean(df)
    total_before_filter = len(df)
    classes_before_filter = df["cdd"].nunique()
    if allowed_cdds_path is not None:
        reference = pd.read_csv(allowed_cdds_path, dtype=str)
        if "cdd" not in reference:
            raise ValueError(f"arquivo de referência sem coluna cdd: {allowed_cdds_path}")
        allowed = set(reference["cdd"].map(normalize_cdd)) - {""}
        df = df[df["cdd"].isin(allowed)].copy()
    df = filter_and_cap_classes(df, min_per_class, max_per_class, random_state)
    if df.empty:
        raise ValueError("nenhum registro sobrou após os filtros de classe")

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "dataset_limpo.csv", index=False)

    active_group = group_column if group_column in df.columns else None
    if active_group:
        train, val, test = _split_grouped(df, active_group, random_state)
    else:
        counts = df["cdd"].value_counts()
        rare = set(counts[counts < 3].index)
        df["cdd_strat"] = df["cdd"].where(~df["cdd"].isin(rare), "outros")
        main, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["cdd_strat"]
        )
        train, val = train_test_split(
            main, test_size=0.25, random_state=random_state, stratify=main["cdd_strat"]
        )

    train_labels = set(train["cdd"])
    missing_from_train = sorted(set(df["cdd"]) - train_labels)
    if missing_from_train:
        raise ValueError(
            "há rótulos fora do treino após o split; aumente --min-per-class "
            f"ou ajuste o agrupamento: {', '.join(missing_from_train[:10])}"
        )
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    report = {
        "total_antes_filtro": total_before_filter,
        "classes_antes_filtro": classes_before_filter,
        "total": len(df),
        "classes": df["cdd"].nunique(),
        "min_per_class": min_per_class,
        "max_per_class": max_per_class,
        "allowed_cdds_path": str(allowed_cdds_path) if allowed_cdds_path else None,
        "com_resumo": int(df["has_resumo"].sum()),
        "sem_resumo": int((~df["has_resumo"]).sum()),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "group_column": active_group,
        "distribuicao_cdd": df["cdd"].value_counts().to_dict(),
    }
    if "source" in df.columns:
        report["fontes"] = df["source"].value_counts().to_dict()
    return df, train, val, test, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara um dataset CDD para uma rodada isolada")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV de entrada (padrão: output/livros_cdd_005.csv)",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=None,
        help="descarta CDDs com menos registros que este limiar",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="amostra no máximo esta quantidade por CDD para balancear a rodada",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="coluna que não pode cruzar splits (ex.: work_id); ausente = split estratificado legado",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Diretório para dataset_limpo.csv e os splits (padrão: output/prepared)",
    )
    parser.add_argument(
        "--allowed-cdds-from",
        type=Path,
        help="CSV de referência cuja coluna cdd define o espaço de rótulos permitido",
    )
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[2]
    raw = args.input or base / "output" / "livros_cdd_005.csv"
    out = args.output or base / "output" / "prepared"
    _, _, _, _, rep = prepare_dataset(
        raw,
        out,
        group_column=args.group_column,
        min_per_class=args.min_per_class,
        max_per_class=args.max_per_class,
        allowed_cdds_path=args.allowed_cdds_from,
    )
    print(json.dumps(rep, ensure_ascii=False, indent=2))
