"""Experimento G: ensemble de probabilidades XLM-R e recuperação kNN/TF-IDF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .knn_baseline import KNNConfig, score_knn
from .metrics import evaluate, json_safe, save_eval


def blend_scores(xlmr_scores: np.ndarray, knn_scores: np.ndarray, alpha_xlmr: float) -> np.ndarray:
    """Combina duas distribuições por média ponderada."""
    if xlmr_scores.shape != knn_scores.shape:
        raise ValueError("escores XLM-R e kNN precisam ter a mesma forma")
    if not 0 <= alpha_xlmr <= 1:
        raise ValueError("alpha_xlmr deve estar entre 0 e 1")
    return alpha_xlmr * xlmr_scores + (1 - alpha_xlmr) * knn_scores


def predictions_from_scores(
    query: pd.DataFrame,
    labels: list[str],
    xlmr_scores: np.ndarray,
    knn_scores: np.ndarray,
    alpha_xlmr: float,
) -> pd.DataFrame:
    combined = blend_scores(xlmr_scores, knn_scores, alpha_xlmr)
    xlmr_idx = xlmr_scores.argmax(axis=1)
    knn_idx = knn_scores.argmax(axis=1)
    pred_idx = combined.argmax(axis=1)
    rows = []
    for position, (_, row) in enumerate(query.iterrows()):
        rows.append(
            {
                "titulo": row.get("titulo", ""),
                "source": row.get("source", ""),
                "edition_id": row.get("edition_id", ""),
                "work_id": row.get("work_id", ""),
                "idioma": row.get("idioma", ""),
                "cdd_true": row["cdd"],
                "cdd_pred": labels[pred_idx[position]],
                "method": "ensemble_xlmr_knn",
                "alpha_xlmr": alpha_xlmr,
                "cdd_xlmr": labels[xlmr_idx[position]],
                "score_xlmr": round(float(xlmr_scores[position, xlmr_idx[position]]), 6),
                "cdd_knn": labels[knn_idx[position]],
                "score_knn": round(float(knn_scores[position, knn_idx[position]]), 6),
                "score_ensemble": round(float(combined[position, pred_idx[position]]), 6),
            }
        )
    return pd.DataFrame(rows)


def select_alpha(
    val: pd.DataFrame,
    labels: list[str],
    xlmr_scores: np.ndarray,
    knn_scores: np.ndarray,
    alphas: list[float],
) -> tuple[float, pd.DataFrame]:
    """Escolhe peso por acurácia completa; macro-F1 e granularidade desempata."""
    if not alphas:
        raise ValueError("nenhum alpha informado")
    rows = []
    for alpha in alphas:
        prediction = predictions_from_scores(val, labels, xlmr_scores, knn_scores, alpha)
        report = json_safe(evaluate(prediction))
        rows.append(
            {
                "alpha_xlmr": alpha,
                "full_accuracy": report["accuracy_por_nivel"]["full"]["accuracy"],
                "macro_f1": report["macro_f1"],
                "accuracy_1dec": report["accuracy_por_nivel"]["1dec"]["accuracy"],
                "accuracy_section": report["accuracy_por_nivel"]["section"]["accuracy"],
                "n": report["n"],
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["full_accuracy", "macro_f1", "accuracy_1dec", "accuracy_section", "alpha_xlmr"],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return float(table.iloc[0]["alpha_xlmr"]), table


def _read_split(prepared_dir: Path, split: str) -> pd.DataFrame:
    path = prepared_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"split não encontrado: {path}")
    return pd.read_csv(path, dtype=str)


def run_experiment(
    prepared_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    knn_config: KNNConfig = KNNConfig(feature="word_char", k=11),
    alphas: list[float] | None = None,
    batch_size: int = 16,
) -> dict:
    """Seleciona o peso em validação e avalia no teste sem retreinar o XLM-R."""
    from src.bert_attention.train import predict_probabilities

    alphas = alphas if alphas is not None else [round(i / 20, 2) for i in range(0, 21)]
    if len(set(alphas)) != len(alphas):
        raise ValueError("alphas duplicados")
    train = _read_split(prepared_dir, "train")
    val = _read_split(prepared_dir, "val")
    test = _read_split(prepared_dir, "test")
    labels, xlmr_val = predict_probabilities(val, checkpoint_dir, batch_size)
    labels_test, xlmr_test = predict_probabilities(test, checkpoint_dir, batch_size)
    if labels != labels_test:
        raise ValueError("rótulos do checkpoint divergentes entre validação e teste")
    train_labels = sorted(train["cdd"].astype(str).unique())
    if labels != train_labels:
        raise ValueError("rótulos do checkpoint não correspondem ao treino do experimento")

    knn_val = score_knn(train, val, knn_config, labels)
    knn_test = score_knn(train, test, knn_config, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "escores_val.npz", labels=np.array(labels), xlmr=xlmr_val, knn=knn_val)
    np.savez_compressed(output_dir / "escores_test.npz", labels=np.array(labels), xlmr=xlmr_test, knn=knn_test)

    selected_alpha, selection = select_alpha(val, labels, xlmr_val, knn_val, alphas)
    selection.to_csv(output_dir / "selecao_G_val.csv", index=False)
    val_predictions = predictions_from_scores(val, labels, xlmr_val, knn_val, selected_alpha)
    val_predictions.to_csv(output_dir / "exp_G_val.csv", index=False)
    val_report = save_eval(val_predictions, str(output_dir / "avaliacao_G_val"))

    test_predictions = predictions_from_scores(test, labels, xlmr_test, knn_test, selected_alpha)
    test_predictions.to_csv(output_dir / "exp_G_test.csv", index=False)
    test_report = save_eval(test_predictions, str(output_dir / "avaliacao_G_test"))
    summary = {
        "experiment": "G",
        "description": "ensemble XLM-R + kNN/TF-IDF title-only; peso selecionado na validação",
        "prepared_dir": str(prepared_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "knn": asdict(knn_config),
        "selection_metric": "full_accuracy",
        "candidate_alphas_xlmr": alphas,
        "selected_alpha_xlmr": selected_alpha,
        "validation": json_safe(val_report),
        "test": json_safe(test_report),
    }
    (output_dir / "resumo_G.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento G — ensemble XLM-R + kNN")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-feature", choices=("word", "char", "word_char"), default="word_char")
    parser.add_argument("--knn-k", type=int, default=11)
    parser.add_argument("--alphas", nargs="+", type=float, default=[round(i / 20, 2) for i in range(0, 21)])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    summary = run_experiment(
        args.prepared_dir,
        args.checkpoint_dir,
        args.output_dir,
        KNNConfig(args.knn_feature, args.knn_k),
        args.alphas,
        args.batch_size,
    )
    test = summary["test"]
    print(
        f"G concluído | alpha_xlmr={summary['selected_alpha_xlmr']:.2f} | "
        f"test full={test['accuracy_por_nivel']['full']['accuracy']:.4f} "
        f"macro-F1={test['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
