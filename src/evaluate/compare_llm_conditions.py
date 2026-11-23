"""Compara L0/L1 pareados e audita as palavras-chave enviadas ao LLM."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

import pandas as pd

try:
    from .metrics import canonical_cdd
except ImportError:  # pragma: no cover
    from metrics import canonical_cdd


def _mcnemar_exact(baseline_only: int, keywords_only: int) -> float:
    """p bilateral exato para os pares discordantes de McNemar."""
    discordant = baseline_only + keywords_only
    if not discordant:
        return 1.0
    lower_tail = sum(comb(discordant, k) for k in range(min(baseline_only, keywords_only) + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def compare(baseline: pd.DataFrame, keywords: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = {"edition_id", "cdd_true", "cdd_pred"}
    for name, frame in (("L0", baseline), ("L1", keywords)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} sem colunas: {', '.join(sorted(missing))}")
        if frame["edition_id"].duplicated().any():
            raise ValueError(f"{name} tem edition_id duplicado")
    merged = baseline.merge(
        keywords,
        on="edition_id",
        how="outer",
        suffixes=("_l0", "_l1"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("L0 e L1 não têm os mesmos edition_id")
    true = merged["cdd_true_l0"].map(canonical_cdd)
    if not true.equals(merged["cdd_true_l1"].map(canonical_cdd)):
        raise ValueError("gabaritos divergem entre L0 e L1")
    merged["acerto_l0"] = merged["cdd_pred_l0"].map(canonical_cdd).eq(true)
    merged["acerto_l1"] = merged["cdd_pred_l1"].map(canonical_cdd).eq(true)
    l0_only = int((merged["acerto_l0"] & ~merged["acerto_l1"]).sum())
    l1_only = int((~merged["acerto_l0"] & merged["acerto_l1"]).sum())

    terms: Counter[str] = Counter()
    empty = 0
    # L0 também registra a coluna (como lista vazia), então o merge a renomeia
    # para ``keywords_l1``. No formato legado, ela pode permanecer sem sufixo.
    keyword_column = "keywords_l1" if "keywords_l1" in merged else "keywords"
    for value in merged.get(keyword_column, pd.Series("[]", index=merged.index)).fillna("[]"):
        try:
            extracted = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            extracted = []
        if not extracted:
            empty += 1
        terms.update(str(term).strip() for term in extracted if str(term).strip())

    report = {
        "n": len(merged),
        "acertos_l0": int(merged["acerto_l0"].sum()),
        "acertos_l1": int(merged["acerto_l1"].sum()),
        "mudancas_predicao": int((merged["cdd_pred_l0"].map(canonical_cdd) != merged["cdd_pred_l1"].map(canonical_cdd)).sum()),
        "pares": {
            "ambos_acertam": int((merged["acerto_l0"] & merged["acerto_l1"]).sum()),
            "somente_l0_acerta": l0_only,
            "somente_l1_acerta": l1_only,
            "ambos_erram": int((~merged["acerto_l0"] & ~merged["acerto_l1"]).sum()),
        },
        "mcnemar_exato_bilateral_p": _mcnemar_exact(l0_only, l1_only),
        "keywords": {
            "linhas_sem_keywords": empty,
            "taxa_sem_keywords": round(empty / len(merged), 4) if len(merged) else 0.0,
            "termos_unicos": len(terms),
        },
    }
    top_terms = pd.DataFrame(terms.most_common(), columns=["keyword", "frequencia"])
    return report, merged, top_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l0", type=Path, required=True)
    parser.add_argument("--l1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, pairs, top_terms = compare(pd.read_csv(args.l0, dtype=str), pd.read_csv(args.l1, dtype=str))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparacao_l0_l1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pairs.to_csv(args.output_dir / "pares_l0_l1.csv", index=False)
    top_terms.to_csv(args.output_dir / "keywords_frequencia.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
