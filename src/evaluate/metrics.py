"""Avaliação reproduzível de classificações CDD."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..data.cdd import truncate_cdd

LEVELS = ("class", "division", "section", "1dec", "full")


def canonical_cdd(value: object) -> str:
    """Normaliza uma CDD para no máximo cinco algarismos.

    Alguns CSVs perdem o zero inicial por inferência numérica (``25.46`` em
    vez de ``025.46``). Corrigimos esse caso sem alterar o artefato original.
    """
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    base, dot, decimal = text.partition(".")
    if dot and base.isdigit() and 1 <= len(base) < 3:
        text = f"{base.zfill(3)}.{decimal}"
    return truncate_cdd(text, max_digits=5)


def _levels(codigo: object) -> dict[str, str]:
    code = canonical_cdd(codigo)
    if not code:
        return {level: "" for level in LEVELS}
    base, _, decimal = code.partition(".")
    first_decimal = f"{base}.{decimal[0]}" if decimal else base
    return {
        "class": base[0],
        "division": base[:2],
        "section": base,
        "1dec": first_decimal,
        "full": code,
    }


def _per_level(df: pd.DataFrame) -> dict:
    report = {}
    for level in LEVELS:
        hits = int((df[f"true_{level}"] == df[f"pred_{level}"]).sum())
        report[level] = {"accuracy": round(hits / len(df), 4), "hits": hits, "total": len(df)}
    return report


def _evaluate_core(df: pd.DataFrame) -> dict:
    """Calcula métricas sem separar a origem da previsão."""
    df = df.copy().reset_index(drop=True)
    for level in LEVELS:
        df[f"true_{level}"] = df["cdd_true"].map(lambda c: _levels(c)[level])
        df[f"pred_{level}"] = df["cdd_pred"].map(lambda c: _levels(c)[level])

    # O macro-F1 representa só os rótulos requisitados pelo gabarito. Uma
    # classe inventada continua reduzindo o recall correto, mas não adiciona
    # uma classe extra de precisão zero que distorceria a média.
    labels = sorted(label for label in df["true_full"].unique() if label)
    p, r, f, _ = precision_recall_fscore_support(
        df["true_full"], df["pred_full"], labels=labels, average="macro", zero_division=0
    )
    per_class = {}
    for label in labels:
        n = int((df["true_full"] == label).sum())
        hits = int(((df["true_full"] == label) & (df["pred_full"] == label)).sum())
        per_class[label] = {"n": n, "hits": hits, "acc": round(hits / n, 3) if n else 0}

    # Não descarta silenciosamente as respostas fora do espaço do gabarito.
    outside = ~df["pred_full"].isin(labels)
    predicted_for_confusion = df["pred_full"].where(~outside, "<fora_do_gabarito>")
    confusion_labels = list(labels)
    if bool(outside.any()):
        confusion_labels.append("<fora_do_gabarito>")
    conf = pd.DataFrame(
        confusion_matrix(df["true_full"], predicted_for_confusion, labels=confusion_labels),
        index=confusion_labels,
        columns=confusion_labels,
    )

    return {
        "n": len(df),
        "accuracy_por_nivel": _per_level(df),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f), 4),
        "labels_gabarito": labels,
        "predicoes_vazias": int((df["pred_full"] == "").sum()),
        "predicoes_fora_do_gabarito": int(outside.sum()),
        "por_classe": per_class,
        "confusao": conf,
    }


def evaluate(df: pd.DataFrame) -> dict:
    """Avalia total e expõe LLM, fallback kNN e falhas separadamente."""
    missing = {"cdd_true", "cdd_pred"} - set(df.columns)
    if missing:
        raise ValueError(f"resultado sem colunas obrigatórias: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("não é possível avaliar um resultado vazio")

    report = _evaluate_core(df)
    if "method" in df:
        by_method = {}
        for method, subset in df.groupby(df["method"].fillna("none"), sort=True):
            by_method[str(method)] = json_safe(_evaluate_core(subset))
        report["por_origem_predicao"] = by_method
    return report


def json_safe(report: dict) -> dict:
    out = dict(report)
    out.pop("confusao", None)
    return out


def save_eval(exp_csv: pd.DataFrame, out_base: str) -> dict:
    report = evaluate(exp_csv)
    out = json_safe(report)
    Path(f"{out_base}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    report["confusao"].to_csv(f"{out_base}.confusion.csv")
    return out
