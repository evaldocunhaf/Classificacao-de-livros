"""Experimento E: baseline kNN/TF-IDF para classificação CDD por título.

O módulo seleciona representação e número de vizinhos exclusivamente na
validação e aplica a configuração escolhida uma única vez no teste. Os splits
de origem não são modificados; os artefatos ficam em uma rodada própria.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import evaluate, json_safe, save_eval


@dataclass(frozen=True)
class KNNConfig:
    feature: str
    k: int


FEATURES = ("word", "char", "word_char")


def _text(df: pd.DataFrame) -> pd.Series:
    """Retorna somente títulos, mantendo este baseline estritamente title-only."""
    if "titulo" not in df:
        raise ValueError("split sem coluna obrigatória: titulo")
    return df["titulo"].fillna("").astype(str)


def _vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w[\w-]+\b",
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    return word, char


def _matrices(train_text: pd.Series, query_text: pd.Series, feature: str):
    if feature not in FEATURES:
        raise ValueError(f"feature inválida: {feature}. Use: {', '.join(FEATURES)}")
    word, char = _vectorizers()
    if feature == "word":
        return word.fit_transform(train_text), word.transform(query_text)
    if feature == "char":
        return char.fit_transform(train_text), char.transform(query_text)

    train_word = word.fit_transform(train_text)
    query_word = word.transform(query_text)
    train_char = char.fit_transform(train_text)
    query_char = char.transform(query_text)
    return hstack((train_word, train_char)), hstack((query_word, query_char))


def _choose_code(neighbors: pd.DataFrame) -> str:
    """Vota pela soma da similaridade; desempata por melhor vizinho e código."""
    scores = neighbors.groupby("cdd", sort=True)["similarity"].sum()
    best_sum = scores.max()
    candidates = scores.index[np.isclose(scores, best_sum)].tolist()
    if len(candidates) == 1:
        return str(candidates[0])
    best_neighbor = neighbors.groupby("cdd")["similarity"].max()
    return str(sorted(candidates, key=lambda code: (-best_neighbor[code], code))[0])


def predict_knn(train: pd.DataFrame, query: pd.DataFrame, config: KNNConfig) -> pd.DataFrame:
    """Prediz CDDs por kNN, sempre ajustado apenas no dataframe de treino."""
    if train.empty:
        raise ValueError("o treino está vazio")
    if "cdd" not in train or "cdd" not in query:
        raise ValueError("splits sem coluna obrigatória: cdd")
    if config.k < 1:
        raise ValueError("k deve ser positivo")

    train_matrix, query_matrix = _matrices(_text(train), _text(query), config.feature)
    similarities = cosine_similarity(query_matrix, train_matrix)
    labels = train["cdd"].fillna("").astype(str).reset_index(drop=True)
    rows: list[dict] = []

    for position, (_, row) in enumerate(query.iterrows()):
        sims = similarities[position]
        indices = np.argsort(-sims, kind="stable")[: min(config.k, len(train))]
        neighbors = pd.DataFrame({"cdd": labels.iloc[indices].to_numpy(), "similarity": sims[indices]})
        predicted = _choose_code(neighbors)
        rows.append(
            {
                "titulo": row.get("titulo", ""),
                "source": row.get("source", ""),
                "edition_id": row.get("edition_id", ""),
                "work_id": row.get("work_id", ""),
                "idioma": row.get("idioma", ""),
                "cdd_true": row["cdd"],
                "cdd_pred": predicted,
                "method": f"knn_tfidf_{config.feature}",
                "k": config.k,
                "neighbor_codes": "|".join(neighbors["cdd"].tolist()),
                "neighbor_scores": "|".join(f"{score:.6f}" for score in neighbors["similarity"]),
            }
        )
    return pd.DataFrame(rows)


def score_knn(
    train: pd.DataFrame,
    query: pd.DataFrame,
    config: KNNConfig,
    labels: list[str] | None = None,
) -> np.ndarray:
    """Retorna distribuições de similaridade por CDD para combinar com outro modelo.

    A massa de similaridade dos ``k`` vizinhos é somada por classe e normalizada
    por linha. Quando um título não compartilha nenhum termo com o treino, usa
    distribuição uniforme para não introduzir preferência arbitrária de classe.
    """
    if train.empty:
        raise ValueError("o treino está vazio")
    if config.k < 1:
        raise ValueError("k deve ser positivo")
    class_labels = labels or sorted(train["cdd"].dropna().astype(str).unique())
    if not class_labels:
        raise ValueError("nenhum rótulo no treino")
    class_index = {label: i for i, label in enumerate(class_labels)}
    train_matrix, query_matrix = _matrices(_text(train), _text(query), config.feature)
    similarities = cosine_similarity(query_matrix, train_matrix)
    train_labels = train["cdd"].fillna("").astype(str).to_numpy()
    scores = np.zeros((len(query), len(class_labels)), dtype=float)

    for position, sims in enumerate(similarities):
        indices = np.argsort(-sims, kind="stable")[: min(config.k, len(train))]
        for index in indices:
            label = train_labels[index]
            if label in class_index:
                scores[position, class_index[label]] += max(float(sims[index]), 0.0)
        total = scores[position].sum()
        if total > 0:
            scores[position] /= total
        else:
            scores[position].fill(1 / len(class_labels))
    return scores


def _report_row(config: KNNConfig, predictions: pd.DataFrame) -> dict:
    report = json_safe(evaluate(predictions))
    return {
        "feature": config.feature,
        "k": config.k,
        "full_accuracy": report["accuracy_por_nivel"]["full"]["accuracy"],
        "macro_f1": report["macro_f1"],
        "accuracy_1dec": report["accuracy_por_nivel"]["1dec"]["accuracy"],
        "accuracy_section": report["accuracy_por_nivel"]["section"]["accuracy"],
        "n": report["n"],
    }


def select_on_validation(train: pd.DataFrame, val: pd.DataFrame, configs: list[KNNConfig]) -> tuple[KNNConfig, pd.DataFrame]:
    """Seleciona por acurácia completa; F1 e níveis hierárquicos são desempates."""
    if not configs:
        raise ValueError("nenhuma configuração foi informada")
    rows = [_report_row(config, predict_knn(train, val, config)) for config in configs]
    table = pd.DataFrame(rows).sort_values(
        ["full_accuracy", "macro_f1", "accuracy_1dec", "accuracy_section", "k", "feature"],
        ascending=[False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = KNNConfig(feature=str(table.iloc[0]["feature"]), k=int(table.iloc[0]["k"]))
    return selected, table


def _read_split(prepared_dir: Path, name: str) -> pd.DataFrame:
    path = prepared_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"split não encontrado: {path}")
    return pd.read_csv(path, dtype=str)


def run_experiment(
    prepared_dir: Path,
    output_dir: Path,
    features: tuple[str, ...] = FEATURES,
    ks: tuple[int, ...] = (1, 3, 5, 7, 9, 11),
) -> dict:
    """Executa seleção em validação e teste final sem alterar a rodada fonte."""
    unknown = sorted(set(features) - set(FEATURES))
    if unknown:
        raise ValueError(f"features inválidas: {', '.join(unknown)}")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks precisa conter inteiros positivos")

    train = _read_split(prepared_dir, "train")
    val = _read_split(prepared_dir, "val")
    test = _read_split(prepared_dir, "test")
    configs = [KNNConfig(feature=feature, k=k) for feature in features for k in ks]
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, selection = select_on_validation(train, val, configs)
    selection.to_csv(output_dir / "selecao_E_val.csv", index=False)
    val_predictions = predict_knn(train, val, selected)
    val_predictions.to_csv(output_dir / "exp_E_val.csv", index=False)
    val_report = save_eval(val_predictions, str(output_dir / "avaliacao_E_val"))

    test_predictions = predict_knn(train, test, selected)
    test_predictions.to_csv(output_dir / "exp_E_test.csv", index=False)
    test_report = save_eval(test_predictions, str(output_dir / "avaliacao_E_test"))

    summary = {
        "experiment": "E",
        "description": "kNN/TF-IDF title-only; seleção na validação, teste congelado",
        "prepared_dir": str(prepared_dir),
        "selection_metric": "full_accuracy",
        "candidates": [asdict(config) for config in configs],
        "selected": asdict(selected),
        "validation": json_safe(val_report),
        "test": json_safe(test_report),
    }
    (output_dir / "resumo_E.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento E — baseline kNN/TF-IDF title-only")
    parser.add_argument("--prepared-dir", type=Path, required=True, help="diretório com train.csv, val.csv e test.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="diretório novo para os artefatos do experimento")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 7, 9, 11])
    args = parser.parse_args()
    summary = run_experiment(args.prepared_dir, args.output_dir, tuple(args.features), tuple(args.ks))
    selected = summary["selected"]
    test = summary["test"]
    print(
        f"E concluído | feature={selected['feature']} k={selected['k']} | "
        f"test full={test['accuracy_por_nivel']['full']['accuracy']:.4f} "
        f"macro-F1={test['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
