"""Baseline TF-IDF + regressão logística com seleção apenas na validação."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .metrics import evaluate, json_safe, save_eval


def _text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        raise ValueError(f"split sem coluna de texto: {column}")
    return df[column].fillna("").astype(str)


def fit_predict(
    train: pd.DataFrame,
    query: pd.DataFrame,
    c_value: float,
    text_column: str = "texto",
) -> pd.Series:
    """Ajusta exclusivamente no treino e prediz as linhas de consulta."""
    if "cdd" not in train:
        raise ValueError("split de treino sem coluna cdd")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(_text(train, text_column))
    x_query = vectorizer.transform(_text(query, text_column))
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
    )
    model.fit(x_train, train["cdd"].astype(str))
    return pd.Series(model.predict(x_query), index=query.index, dtype="string")


def prediction_frame(query: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    columns = [
        "titulo", "source", "edition_id", "work_id", "idioma", "cdd",
    ]
    available = [column for column in columns if column in query]
    result = query[available].copy()
    result = result.rename(columns={"cdd": "cdd_true"})
    result["cdd_pred"] = predictions.astype(str).to_numpy()
    result["method"] = "tfidf_logreg"
    return result.reset_index(drop=True)


def select_c(train: pd.DataFrame, val: pd.DataFrame, candidates: list[float], text_column: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for c_value in candidates:
        prediction = prediction_frame(val, fit_predict(train, val, c_value, text_column))
        report = json_safe(evaluate(prediction))
        rows.append(
            {
                "c": c_value,
                "full_accuracy": report["accuracy_por_nivel"]["full"]["accuracy"],
                "macro_f1": report["macro_f1"],
                "accuracy_section": report["accuracy_por_nivel"]["section"]["accuracy"],
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["full_accuracy", "macro_f1", "accuracy_section", "c"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return float(table.iloc[0]["c"]), table


def _read_split(prepared_dir: Path, name: str) -> pd.DataFrame:
    path = prepared_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split não encontrado: {path}")
    return pd.read_csv(path, dtype=str)


def run(prepared_dir: Path, output_dir: Path, candidates: list[float], text_column: str) -> dict:
    train = _read_split(prepared_dir, "train")
    val = _read_split(prepared_dir, "val")
    test = _read_split(prepared_dir, "test")
    selected_c, selection = select_c(train, val, candidates, text_column)

    val_predictions = prediction_frame(val, fit_predict(train, val, selected_c, text_column))
    test_predictions = prediction_frame(test, fit_predict(train, test, selected_c, text_column))
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "selecao_logreg_val.csv", index=False)
    val_predictions.to_csv(output_dir / "exp_logreg_val.csv", index=False)
    test_predictions.to_csv(output_dir / "exp_logreg_test.csv", index=False)
    val_report = save_eval(val_predictions, str(output_dir / "avaliacao_logreg_val"))
    test_report = save_eval(test_predictions, str(output_dir / "avaliacao_logreg_test"))
    summary = {
        "description": "baseline TF-IDF de palavra (1–2 gramas) + regressão logística",
        "text_column": text_column,
        "candidate_c": candidates,
        "selected_c": selected_c,
        "validation": val_report,
        "test": test_report,
    }
    (output_dir / "resumo_logreg.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + regressão logística")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--text-column", default="texto")
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.output_dir, args.c, args.text_column)
    test = summary["test"]
    print(
        f"logreg C={summary['selected_c']} | test full={test['accuracy_por_nivel']['full']['accuracy']:.4f} "
        f"macro-F1={test['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
