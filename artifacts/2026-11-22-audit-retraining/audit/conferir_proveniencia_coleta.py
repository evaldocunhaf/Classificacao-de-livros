#!/usr/bin/env python3
"""Confere proveniência usando CSVs locais, sem escrita, treino ou inferência.

Uso (raiz TCC do repositório tcc-files):
  python conferir_proveniencia_coleta.py --repo-root /caminho/para/TCC

Os dumps gigantes não são reprocessados. Os CSVs locais de origem, normalmente
ignorados no Git, são necessários para as asserções históricas e de semente.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def check(repo_root):
    pipeline = repo_root / "tcc-pipeline"
    sys.path.insert(0, str(pipeline))
    import pandas as pd
    import sklearn
    from src.data.cdd import truncate_cdd
    from src.data.prepare_dataset import (
        _split_grouped, filter_and_cap_classes, load_and_clean,
    )

    datasets = pipeline / "output/datasets"
    runs = pipeline / "output/runs"
    historical = rows(datasets / "openlibrary_cdd_001_004_005_006_025_5digits.csv")
    corrected = rows(datasets / "openlibrary_cdd_001_004_005_006_025_5digits_one_cdd.csv")
    original_prepared = runs / "2026-10-15-openlibrary-one-cdd/prepared"
    subject_prepared = runs / "2026-10-15-openlibrary-subjects-43cdd/prepared"
    final_prepared = runs / "2026-10-16-final-comparison-subjects/prepared"
    original = rows(original_prepared / "dataset_limpo.csv")

    historical_counts = Counter(row["cdd"] for row in historical)
    eligible = {code for code, count in historical_counts.items() if count >= 500}
    final_codes = {row["cdd"] for row in original}
    assert len(historical) == 102323
    assert len({row["edition_id"] for row in historical}) == 101787
    assert len(historical_counts) == 299
    assert len(eligible) == 43 and eligible == final_codes

    seen_editions, kept_counts, expected_pool = set(), Counter(), []
    prefixes = {"001", "004", "005", "006", "025"}
    for row in historical:
        edition_id = row["edition_id"]
        if edition_id in seen_editions:
            continue
        seen_editions.add(edition_id)
        codes = dict.fromkeys(
            truncate_cdd(value.strip(), 5)
            for value in row["cdd_raw"].split(" | ")
        )
        matches = [code for code in codes if code[:3] in prefixes]
        if not matches:
            continue
        code = matches[0]
        if kept_counts[code] >= 100:
            continue
        kept_counts[code] += 1
        expected_pool.append((edition_id, code))
    actual_pool = [(row["edition_id"], row["cdd"]) for row in corrected]
    assert expected_pool == actual_pool and len(actual_pool) == 16814
    selected_pool = {(row["edition_id"], row["cdd"]) for row in corrected
                     if row["cdd"] in eligible}
    assert selected_pool == {(row["edition_id"], row["cdd"]) for row in original}

    result = {
        "mode": "read_only",
        "pandas_version_used_for_check": pd.__version__,
        "sklearn_version_used_for_check": sklearn.__version__,
        "historical_43_eligible_codes_equal_final": True,
        "first_100_single_label_pool_entire_order_reproduced": True,
        "corrected_pool_rows": len(actual_pool),
        "selected_4300_equal_all_allowed_pool_rows": True,
        "dataset_order_asserts": {},
        "split_membership_asserts": {},
        "subject_filter_asserts": {},
    }
    cleaned_pool = load_and_clean(pd.DataFrame(corrected))
    cleaned_pool = cleaned_pool[cleaned_pool["cdd"].isin(eligible)].copy()
    reconstructed = filter_and_cap_classes(cleaned_pool, None, 100, 42)
    assert reconstructed["edition_id"].tolist() == [row["edition_id"] for row in original]
    result["dataset_order_asserts"]["one_cdd_max100_seed42"] = True

    enriched_csv = datasets / "openlibrary_cdd_001_004_005_006_025_5digits_one_cdd_subjects.csv"
    enriched = load_and_clean(pd.read_csv(enriched_csv, dtype=str))
    reconstructed_subjects = filter_and_cap_classes(enriched, None, 100, 42)
    saved_subjects = pd.read_csv(subject_prepared / "dataset_limpo.csv", dtype=str)
    assert reconstructed_subjects["edition_id"].tolist() == saved_subjects["edition_id"].tolist()
    result["dataset_order_asserts"]["subjects_max100_seed42"] = True

    for run in ("2026-10-15-openlibrary-one-cdd",
                "2026-10-15-openlibrary-subjects-43cdd"):
        prepared = runs / run / "prepared"
        dataset = pd.read_csv(prepared / "dataset_limpo.csv", dtype=str)
        reconstructed_splits = _split_grouped(dataset, "work_id", 42)
        checks = {}
        for split, frame in zip(("train", "val", "test"), reconstructed_splits):
            actual = pd.read_csv(prepared / (split + ".csv"), dtype=str)
            checks[split] = set(frame["edition_id"]) == set(actual["edition_id"])
            assert checks[split]
        result["split_membership_asserts"][run] = checks

    for split in ("train", "val", "test"):
        source = pd.read_csv(subject_prepared / (split + ".csv"), dtype=str)
        value = source["resumo"].fillna("").astype(str).str.strip()
        selected = source.loc[value.ne("") & ~value.str.lower().isin({"nan", "none", "null"})]
        actual = pd.read_csv(final_prepared / (split + ".csv"), dtype=str)
        assert selected["edition_id"].tolist() == actual["edition_id"].tolist()
        assert set(actual["cdd"]) == eligible
        result["subject_filter_asserts"][split] = {
            "before": len(source), "after": len(actual),
            "excluded": len(source) - len(actual),
            "all_43_codes_present": True,
            "entire_edition_order_reproduced": True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.repo_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
