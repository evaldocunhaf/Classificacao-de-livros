"""Seleção e avaliação final dos métodos no mesmo recorte descritivo.

O módulo separa explicitamente duas fases: ``select`` só lê treino/validação;
``test`` só é permitido depois de gravada a seleção. Assim, os 808 títulos de
teste com ``subjects`` permanecem fora de qualquer ajuste de hiperparâmetros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bert_attention.train import prediction_frame, predict_probabilities
from src.evaluate.ensemble_xlmr_knn import predictions_from_scores, select_alpha
from src.evaluate.knn_baseline import FEATURES, KNNConfig, predict_knn, score_knn, select_on_validation
from src.evaluate.logreg_baseline import fit_predict, prediction_frame as logreg_frame, select_c
from src.evaluate.metrics import json_safe, save_eval


SPLITS = ("train", "val", "test")
K_VALUES = (1, 3, 5, 7, 9, 11)
LOGREG_C_VALUES = (0.1, 1.0, 10.0)
ALPHAS = tuple(round(i / 20, 2) for i in range(21))


def _has_descriptive_text(df: pd.DataFrame) -> pd.Series:
    if "resumo" not in df:
        raise ValueError("split sem coluna resumo; não é possível formar o recorte descritivo")
    value = df["resumo"].fillna("").astype(str).str.strip()
    return value.ne("") & ~value.str.lower().isin({"nan", "none", "null"})


def prepare_descriptive_splits(source_prepared: Path, prepared_dir: Path) -> dict[str, int]:
    """Copia somente itens com subjects, preservando os identificadores originais."""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in SPLITS:
        source = source_prepared / f"{split}.csv"
        df = pd.read_csv(source, dtype=str)
        selected = df.loc[_has_descriptive_text(df)].copy()
        selected.to_csv(prepared_dir / f"{split}.csv", index=False)
        counts[split] = len(selected)
    (prepared_dir / "recorte_descritivo.json").write_text(
        json.dumps(
            {
                "source_prepared": str(source_prepared),
                "rule": "resumo/subjects não vazio",
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts


def _read(prepared_dir: Path, split: str) -> pd.DataFrame:
    path = prepared_dir / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split ausente: {path}")
    return pd.read_csv(path, dtype=str)


def _save(predictions: pd.DataFrame, results_dir: Path, name: str) -> dict:
    predictions.to_csv(results_dir / f"exp_{name}.csv", index=False)
    return json_safe(save_eval(predictions, str(results_dir / f"avaliacao_{name}")))


def _write_selection(
    results_dir: Path,
    knn: KNNConfig,
    logreg_c: float,
    alpha_xlmr: float,
    checkpoint_dir: Path,
    n: int,
) -> None:
    payload = {
        "selection_split": "val",
        "n_validation": n,
        "knn": {"feature": knn.feature, "k": knn.k},
        "logreg": {"c": logreg_c, "text_column": "texto"},
        "ensemble_g": {"alpha_xlmr": alpha_xlmr},
        "xlmr_checkpoint": str(checkpoint_dir),
    }
    (results_dir / "selecao_final.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def select(prepared_dir: Path, results_dir: Path, checkpoint_dir: Path, batch_size: int) -> dict:
    """Seleciona E, logística e G na validação; F usa o checkpoint selecionado por validação."""
    results_dir.mkdir(parents=True, exist_ok=True)
    train, val = _read(prepared_dir, "train"), _read(prepared_dir, "val")

    configs = [KNNConfig(feature, k) for feature in FEATURES for k in K_VALUES]
    knn_config, knn_table = select_on_validation(train, val, configs)
    knn_table.to_csv(results_dir / "selecao_E_val.csv", index=False)
    e_report = _save(predict_knn(train, val, knn_config), results_dir, "E_val")

    logreg_c, logreg_table = select_c(train, val, list(LOGREG_C_VALUES), "texto")
    logreg_table.to_csv(results_dir / "selecao_logreg_val.csv", index=False)
    logreg_report = _save(
        logreg_frame(val, fit_predict(train, val, logreg_c, "texto")), results_dir, "logreg_val"
    )

    labels, xlmr_scores = predict_probabilities(val, checkpoint_dir, batch_size)
    f_report = _save(prediction_frame(val, labels, xlmr_scores), results_dir, "F_val")

    knn_scores = score_knn(train, val, knn_config, labels)
    alpha, alpha_table = select_alpha(val, labels, xlmr_scores, knn_scores, list(ALPHAS))
    alpha_table.to_csv(results_dir / "selecao_G_val.csv", index=False)
    np.savez_compressed(results_dir / "escores_G_val.npz", labels=np.array(labels), xlmr=xlmr_scores, knn=knn_scores)
    g_report = _save(
        predictions_from_scores(val, labels, xlmr_scores, knn_scores, alpha), results_dir, "G_val"
    )

    _write_selection(results_dir, knn_config, logreg_c, alpha, checkpoint_dir, len(val))
    reports = {"E": e_report, "logreg": logreg_report, "F": f_report, "G": g_report}
    (results_dir / "resumo_validacao_supervisionados.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reports


def evaluate_test(prepared_dir: Path, results_dir: Path, batch_size: int) -> dict:
    """Aplica uma única vez no teste as escolhas gravadas por ``select``."""
    selection_path = results_dir / "selecao_final.json"
    if not selection_path.is_file():
        raise FileNotFoundError("selecao_final.json ausente; execute primeiro a fase select")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "val":
        raise ValueError("a seleção não foi registrada como proveniente da validação")
    train, test = _read(prepared_dir, "train"), _read(prepared_dir, "test")
    knn = KNNConfig(**selection["knn"])
    logreg_c = float(selection["logreg"]["c"])
    checkpoint = Path(selection["xlmr_checkpoint"])
    alpha = float(selection["ensemble_g"]["alpha_xlmr"])

    reports = {
        "E": _save(predict_knn(train, test, knn), results_dir, "E_test"),
        "logreg": _save(
            logreg_frame(test, fit_predict(train, test, logreg_c, "texto")), results_dir, "logreg_test"
        ),
    }
    labels, xlmr_scores = predict_probabilities(test, checkpoint, batch_size)
    reports["F"] = _save(prediction_frame(test, labels, xlmr_scores), results_dir, "F_test")
    knn_scores = score_knn(train, test, knn, labels)
    np.savez_compressed(results_dir / "escores_G_test.npz", labels=np.array(labels), xlmr=xlmr_scores, knn=knn_scores)
    reports["G"] = _save(
        predictions_from_scores(test, labels, xlmr_scores, knn_scores, alpha), results_dir, "G_test"
    )
    (results_dir / "resumo_teste_supervisionados.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Seleção final comparável no recorte Open Library com subjects")
    parser.add_argument("--source-prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, help="necessário na fase select")
    parser.add_argument("--phase", choices=("prepare", "select", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    prepared_dir, results_dir = args.run_dir / "prepared", args.run_dir / "results"
    if args.phase == "prepare":
        print(json.dumps(prepare_descriptive_splits(args.source_prepared, prepared_dir), ensure_ascii=False))
    elif args.phase == "select":
        if args.checkpoint_dir is None:
            parser.error("--checkpoint-dir é obrigatório na fase select")
        print(json.dumps(select(prepared_dir, results_dir, args.checkpoint_dir, args.batch_size), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(evaluate_test(prepared_dir, results_dir, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
