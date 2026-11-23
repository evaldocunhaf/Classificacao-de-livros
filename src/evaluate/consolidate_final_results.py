"""Consolida a rodada final, verifica pareamento e prepara a análise manual de erros."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


METHODS = (
    ("LLM L0", "exp_A_test_subjects.csv", "avaliacao_A_test_subjects.json"),
    ("kNN/TF-IDF", "exp_E_test.csv", "avaliacao_E_test.json"),
    ("XLM-R + atenção", "exp_F_test.csv", "avaliacao_F_test.json"),
    ("Ensemble G", "exp_G_test.csv", "avaliacao_G_test.json"),
    ("TF-IDF + regressão logística", "exp_logreg_test.csv", "avaliacao_logreg_test.json"),
)


def _section(code: str) -> str:
    return str(code).split(".", 1)[0][:3]


def _initial_category(row: pd.Series) -> str:
    if row["modelo"] == "LLM L0" and str(row.get("method", "")) == "knn":
        return "saída inválida/fallback"
    if _section(row["cdd_true"]) == _section(row["cdd_pred"]):
        return "granularidade excessiva/insuficiente"
    text = str(row.get("texto", "")).strip()
    if len(text.split()) < 5:
        return "texto insuficiente"
    return "aboutness / assunto principal divergente (a confirmar)"


def _stratified_errors(errors: pd.DataFrame, per_model: int = 10) -> pd.DataFrame:
    """Amostra determinística, alternando as cinco classes principais de CDD."""
    selected: list[pd.DataFrame] = []
    for model, group in errors.groupby("modelo", sort=False):
        group = group.sort_values(["classe_cdd", "edition_id"], kind="stable")
        pieces: list[pd.DataFrame] = []
        taken = 0
        positions = {key: 0 for key in group["classe_cdd"].unique()}
        buckets = {
            key: value.reset_index(drop=True)
            for key, value in group.groupby("classe_cdd", sort=True)
        }
        while taken < per_model:
            progressed = False
            for key, bucket in buckets.items():
                position = positions[key]
                if taken >= per_model:
                    break
                if position < len(bucket):
                    pieces.append(bucket.iloc[[position]])
                    positions[key] += 1
                    taken += 1
                    progressed = True
            if not progressed:
                break
        sample = pd.concat(pieces, ignore_index=True)
        if len(sample) != per_model:
            raise ValueError(f"{model}: apenas {len(sample)} erros disponíveis para a amostra")
        selected.append(sample)
    return pd.concat(selected, ignore_index=True)


def _markdown_sample(sample: pd.DataFrame) -> str:
    lines = [
        "# Caderno de análise manual — 50 erros da rodada final",
        "",
        "Amostra determinística de dez erros por método, distribuída entre as classes principais. "
        "`categoria_inicial` é uma triagem automática; `categoria_confirmada` deve ser preenchida após leitura humana, "
        "sem alterar treinamento nem resultados.",
        "",
    ]
    for index, row in sample.iterrows():
        text = " ".join(str(row["texto"]).split())[:500]
        lines.extend(
            [
                f"## {index + 1}. {row['modelo']} — {row['edition_id']}",
                f"- Título: {row['titulo']}",
                f"- CDD de referência: `{row['cdd_true']}`; previsto: `{row['cdd_pred']}`",
                f"- Categoria inicial: {row['categoria_inicial']}",
                "- Categoria confirmada: **pendente de revisão manual**",
                f"- Texto/pistas: {text}",
                "- Justificativa da revisão: ",
                "",
            ]
        )
    return "\n".join(lines)


def _svg_chart(table: pd.DataFrame) -> str:
    """Gera um gráfico vetorial simples, sem dependência de biblioteca gráfica."""
    width, height, left, top, row_height, scale = 1120, 360, 235, 52, 52, 330
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}.label{font-size:14px}.value{font-size:13px}.title{font-size:19px;font-weight:bold}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" class="title">Teste Open Library com subjects (n=808)</text>',
        '<text x="235" y="46" class="label">Acurácia CDD completo</text>',
        '<text x="650" y="46" class="label">Macro-F1</text>',
    ]
    for i, (_, row) in enumerate(table.iterrows()):
        y = top + i * row_height
        name = html.escape(str(row["método"]))
        full = float(row["acurácia_cdd_completo"])
        f1 = float(row["macro_f1"])
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" class="label">{name}</text>',
                f'<rect x="{left}" y="{y}" width="{scale * full:.1f}" height="24" fill="#2a6f97"/>',
                f'<text x="{left + scale + 8}" y="{y + 18}" class="value">{full:.3f}</text>',
                f'<rect x="650" y="{y}" width="{scale * f1:.1f}" height="24" fill="#e07a5f"/>',
                f'<text x="{650 + scale + 8}" y="{y + 18}" class="value">{f1:.3f}</text>',
            ]
        )
    return "".join(parts) + "</svg>"


def consolidate(run_dir: Path) -> dict:
    results = run_dir / "results"
    prepared = run_dir / "prepared"
    test = pd.read_csv(prepared / "test.csv", dtype=str)
    if test["edition_id"].duplicated().any():
        raise ValueError("split de teste possui edition_id duplicado")
    reference = test.set_index("edition_id", drop=False)
    expected_ids = set(reference.index)
    metric_rows, all_errors = [], []

    for name, prediction_file, metrics_file in METHODS:
        prediction = pd.read_csv(results / prediction_file, dtype=str)
        if prediction["edition_id"].duplicated().any():
            raise ValueError(f"{name}: edition_id duplicado")
        if set(prediction["edition_id"]) != expected_ids:
            raise ValueError(f"{name}: identificadores diferentes do teste")
        aligned = prediction.set_index("edition_id", drop=False).loc[reference.index]
        if not aligned["cdd_true"].astype(str).equals(reference["cdd"].astype(str)):
            raise ValueError(f"{name}: gabarito divergente")
        report = json.loads((results / metrics_file).read_text(encoding="utf-8"))
        levels = report["accuracy_por_nivel"]
        metric_rows.append(
            {
                "método": name,
                "n": report["n"],
                "acurácia_classe": levels["class"]["accuracy"],
                "acurácia_divisão": levels["division"]["accuracy"],
                "acurácia_seção": levels["section"]["accuracy"],
                "acurácia_1_decimal": levels["1dec"]["accuracy"],
                "acurácia_cdd_completo": levels["full"]["accuracy"],
                "macro_f1": report["macro_f1"],
                "predições_vazias": report["predicoes_vazias"],
                "predições_fora_gabarito": report["predicoes_fora_do_gabarito"],
            }
        )
        merged = aligned.join(reference[["texto"]], rsuffix="_dataset")
        errors = merged.loc[merged["cdd_true"].astype(str) != merged["cdd_pred"].astype(str)].copy()
        errors["modelo"] = name
        errors["classe_cdd"] = errors["cdd_true"].map(_section)
        all_errors.append(errors.reset_index(drop=True))

    table = pd.DataFrame(metric_rows)
    table.to_csv(results / "tabela_resultados_teste.csv", index=False)
    (results / "tabela_resultados_teste.md").write_text(table.to_markdown(index=False), encoding="utf-8")
    (results / "validacao_pareamento_teste.json").write_text(
        json.dumps(
            {"n": len(test), "edition_id_unicos": len(expected_ids), "todos_os_modelos_mesmos_ids": True},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (results / "grafico_metricas_teste.svg").write_text(_svg_chart(table), encoding="utf-8")

    sample = _stratified_errors(pd.concat(all_errors, ignore_index=True))
    sample["categoria_inicial"] = sample.apply(_initial_category, axis=1)
    sample["categoria_confirmada"] = ""
    sample["justificativa_revisao"] = ""
    columns = [
        "modelo", "edition_id", "titulo", "cdd_true", "cdd_pred", "classe_cdd", "method",
        "categoria_inicial", "categoria_confirmada", "texto", "justificativa_revisao",
    ]
    sample = sample[[column for column in columns if column in sample]]
    sample.to_csv(results / "amostra_50_erros.csv", index=False)
    (results / "caderno_50_erros.md").write_text(_markdown_sample(sample), encoding="utf-8")
    return {"table": table, "n_errors_sample": len(sample)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida o teste final do TCC")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = consolidate(args.run_dir)
    print(summary["table"].to_string(index=False))
    print(f"caderno de erros: {summary['n_errors_sample']} casos")


if __name__ == "__main__":
    main()
