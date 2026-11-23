#!/usr/bin/env python3
"""Auditoria pós-hoc dos splits e sensibilidade a coincidência de títulos.

Somente lê os dados originais; grava resultados novos em --output-dir.
Não treina classificadores, altera previsões ou decide duplicação bibliográfica.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SPECS = {
    "TF-IDF + regressão logística": "exp_logreg_test.csv",
    "XLM-R-AA": "exp_F_test.csv",
    "Ensemble XLM-R-AA + kNN": "exp_G_test.csv",
    "kNN/TF-IDF (complementar)": "exp_E_test.csv",
    "LLM L0 (complementar)": "exp_A_test_subjects.csv",
}
REPORTS = {
    "TF-IDF + regressão logística": "avaliacao_logreg_test.json",
    "XLM-R-AA": "avaliacao_F_test.json",
    "Ensemble XLM-R-AA + kNN": "avaliacao_G_test.json",
    "kNN/TF-IDF (complementar)": "avaliacao_E_test.json",
    "LLM L0 (complementar)": "avaliacao_A_test_subjects.json",
}
EXPECTED_HITS = {
    "TF-IDF + regressão logística": 529, "XLM-R-AA": 476,
    "Ensemble XLM-R-AA + kNN": 498, "kNN/TF-IDF (complementar)": 372,
    "LLM L0 (complementar)": 76,
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize_title(value):
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return " ".join(text.split())


def canonical_code(value):
    text = value.strip().replace("/", "")
    base, dot, decimal = text.partition(".")
    if base.isdigit() and len(base) < 3:
        text = base.zfill(3) + (dot + decimal if dot else "")
    match = re.search(r"(?<!\d)(\d{3})(?:\.(\d+))?", text)
    if not match:
        return ""
    base, decimal = match.groups()
    return base + ("." + decimal[:2] if decimal else "")


def metrics(rows, labels):
    gold = [canonical_code(row["cdd_true"]) for row in rows]
    predicted = [canonical_code(row["cdd_pred"]) for row in rows]
    gold_counts, pred_counts = Counter(gold), Counter(predicted)
    true_positive = Counter(t for t, p in zip(gold, predicted) if t == p)
    present = sorted(set(gold))

    def macro(chosen):
        if not chosen:
            return None
        values = []
        for label in chosen:
            tp = true_positive[label]
            denominator = gold_counts[label] + pred_counts[label]
            values.append(2 * tp / denominator if denominator else 0.0)
        return sum(values) / len(values)

    n = len(rows)
    hits = sum(t == p for t, p in zip(gold, predicted))
    return {
        "n": n,
        "hits": hits,
        "accuracy": hits / n if n else None,
        "macro_f1_labels_present": macro(present),
        "macro_f1_fixed_43": macro(labels),
        "labels_present_n": len(present),
        "labels_present": present,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-approximate", action="store_true")
    parser.add_argument("--historical-counts", action="store_true", help="Confere também as contagens específicas da rodada de outubro")
    args = parser.parse_args()
    run_dir, out_dir = args.run_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {split: run_dir / "prepared" / (split + ".csv") for split in ("train", "val", "test")}
    splits = {split: read_csv(path) for split, path in source_paths.items()}
    for split, rows in splits.items():
        assert {"edition_id", "work_id", "titulo", "cdd"}.issubset(rows[0]), f"schema inválido em {split}"
        ids = [r["edition_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"edition_id repetido em {split}"
    counts = {split: Counter(canonical_code(r["cdd"]) for r in rows) for split, rows in splits.items()}
    labels = sorted(set().union(*(set(c) for c in counts.values())))
    assert len(labels) == 43
    assert all(set(counts[s]) == set(labels) for s in splits), "rótulo ausente em algum split"
    distribution = [{"cdd": label, **{s: counts[s][label] for s in splits},
                     "total": sum(counts[s][label] for s in splits)} for label in labels]
    write_csv(out_dir / "distribuicao_43_cdd.csv", distribution, ["cdd", "train", "val", "test", "total"])

    totals = {s: len(rows) for s, rows in splits.items()}
    total = sum(totals.values())
    assert totals == {"train": 2423, "val": 803, "test": 808}
    assert total == 4034
    assert len(distribution) == 43
    assert sum(r["total"] for r in distribution) == total
    stats = {}
    for split in (*splits, "total"):
        values = [row[split] for row in distribution]
        stats[split] = {"n": sum(values), "labels": len(values), "min": min(values),
                        "median": statistics.median(values), "max": max(values)}

    title_maps = {}
    for split, rows in splits.items():
        mapping = defaultdict(list)
        for r in rows:
            mapping[normalize_title(r["titulo"])].append(r)
        assert "" not in mapping, f"título vazio após normalização em {split}"
        title_maps[split] = mapping

    overlaps, pairs = {}, []
    for a, b in combinations(splits, 2):
        common = set(title_maps[a]) & set(title_maps[b])
        overlap = {
            "normalized_title_groups": len(common),
            "records_" + a: sum(len(title_maps[a][t]) for t in common),
            "records_" + b: sum(len(title_maps[b][t]) for t in common),
            "cross_pairs": sum(len(title_maps[a][t]) * len(title_maps[b][t]) for t in common),
        }
        for field in ("edition_id", "work_id"):
            overlap[field] = len({r[field] for r in splits[a] if r[field]} & {r[field] for r in splits[b] if r[field]})
        overlaps[a + "/" + b] = overlap
        for t in sorted(common):
            for left in sorted(title_maps[a][t], key=lambda r: r["edition_id"]):
                for right in sorted(title_maps[b][t], key=lambda r: r["edition_id"]):
                    pairs.append({"split_a": a, "split_b": b, "normalized_title": t,
                                  "edition_id_a": left["edition_id"], "edition_id_b": right["edition_id"],
                                  "work_id_a": left["work_id"], "work_id_b": right["work_id"],
                                  "title_a": left["titulo"], "title_b": right["titulo"],
                                  "cdd_a": canonical_code(left["cdd"]), "cdd_b": canonical_code(right["cdd"]),
                                  "same_cdd": canonical_code(left["cdd"]) == canonical_code(right["cdd"])})
    write_csv(out_dir / "titulos_coincidentes_entre_splits.csv", pairs,
              ["split_a", "split_b", "normalized_title", "edition_id_a", "edition_id_b", "work_id_a", "work_id_b", "title_a", "title_b", "cdd_a", "cdd_b", "same_cdd"])

    exact_test_ids = {r["edition_id"] for r in splits["test"] if normalize_title(r["titulo"]) in title_maps["train"]}
    nearest, nearest_by_id = [], {}
    if not args.skip_approximate:
        train_titles = sorted(title_maps["train"])
        started = time.monotonic()
        for index, row in enumerate(splits["test"]):
            target = normalize_title(row["titulo"])
            if target in title_maps["train"]:
                best_ratio, best_title = 1.0, target
            else:
                # Examina primeiro comprimentos próximos para elevar cedo o limite.
                candidates = sorted(train_titles, key=lambda t: (abs(len(t) - len(target)), t))
                best_ratio, best_title = -1.0, ""
                matcher = difflib.SequenceMatcher(None, target, autojunk=False)
                for candidate in candidates:
                    upper = 2 * min(len(candidate), len(target)) / (len(candidate) + len(target))
                    if upper < best_ratio:
                        continue
                    matcher.set_seq2(candidate)
                    if matcher.quick_ratio() < best_ratio:
                        continue
                    score = matcher.ratio()
                    if score > best_ratio or (score == best_ratio and candidate < best_title):
                        best_ratio, best_title = score, candidate
            train_matches = sorted(title_maps["train"][best_title], key=lambda r: r["edition_id"])
            selected = train_matches[0]
            entry = {"edition_id_test": row["edition_id"], "work_id_test": row["work_id"], "title_test": row["titulo"],
                     "normalized_test": target, "nearest_normalized_train": best_title,
                     "similarity_ratio": best_ratio, "edition_id_train": selected["edition_id"],
                     "work_id_train": selected["work_id"], "title_train": selected["titulo"],
                     "cdd_test": canonical_code(row["cdd"]), "cdd_train": canonical_code(selected["cdd"]),
                     "train_records_with_nearest_title": len(train_matches),
                     "exact_normalized_match": row["edition_id"] in exact_test_ids}
            nearest.append(entry)
            nearest_by_id[row["edition_id"]] = best_ratio
            if (index + 1) % 100 == 0:
                print(f"títulos aproximados: {index + 1}/{len(splits['test'])}; {time.monotonic() - started:.1f}s", flush=True)
        write_csv(out_dir / "titulo_treino_mais_proximo_por_teste.csv", nearest, list(nearest[0]))

    groups = {
        "full_test": {r["edition_id"] for r in splits["test"]},
        "exact_title_overlap_train": exact_test_ids,
        "without_exact_title_overlap_train": {r["edition_id"] for r in splits["test"]} - exact_test_ids,
    }
    approximate_summary = {}
    for threshold in (0.90, 0.95):
        if not nearest_by_id:
            continue
        included = {identifier for identifier, score in nearest_by_id.items() if score >= threshold}
        tag = str(int(threshold * 100))
        approximate_summary[tag] = {"threshold": threshold, "test_records_at_or_above": len(included),
                                     "test_records_approximate_only": len(included - exact_test_ids)}
        groups["title_similarity_at_least_" + tag] = included
        groups["without_title_similarity_at_least_" + tag] = groups["full_test"] - included

    reference = {r["edition_id"]: canonical_code(r["cdd"]) for r in splits["test"]}
    model_reports, metric_rows = {}, []
    for model, filename in SPECS.items():
        path = run_dir / "results" / filename
        rows = read_csv(path)
        assert {"edition_id", "cdd_true", "cdd_pred"}.issubset(rows[0]), f"schema inválido em {filename}"
        identifiers = [r["edition_id"] for r in rows]
        assert len(identifiers) == len(set(identifiers)) == len(reference)
        assert set(identifiers) == set(reference)
        assert all(canonical_code(r["cdd_true"]) == reference[r["edition_id"]] for r in rows)
        source_paths[filename] = path
        report = {}
        for group, ids in groups.items():
            values = metrics([r for r in rows if r["edition_id"] in ids], labels)
            report[group] = values
            metric_rows.append({"model": model, "subset": group,
                                **{k: v for k, v in values.items() if k != "labels_present"}})
        model_reports[model] = report
        preserved_path = run_dir / "results" / REPORTS[model]
        preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
        if args.historical_counts:
            assert report["full_test"]["hits"] == EXPECTED_HITS[model]
        assert report["full_test"]["hits"] == preserved["accuracy_por_nivel"]["full"]["hits"]
        assert round(report["full_test"]["macro_f1_labels_present"], 4) == preserved["macro_f1"]
        source_paths[REPORTS[model]] = preserved_path
    write_csv(out_dir / "metricas_sensibilidade_titulos.csv", metric_rows, list(metric_rows[0]))

    summary = {
        "method": {"title_normalization": "Unicode NFKD, casefold, remoção de marcas combinantes, pontuação e símbolos para espaços, colapso de espaços",
                   "approximate_similarity": "difflib.SequenceMatcher(None, normalized_test, normalized_train, autojunk=False).ratio(); máximo exato sobre títulos normalizados distintos do treino",
                   "approximate_thresholds": [0.90, 0.95],
                   "code_normalization": "remoção de barras, correção do zero inicial, até cinco algarismos: três antes e dois após o ponto",
                   "macro_f1": "média de F1 por classe; denominador zero recebe zero; apresentar rótulos presentes no subconjunto e 43 rótulos fixos",
                   "evaluation": "sensibilidade pós-hoc das mesmas previsões, sem treinamento ou seleção de modelos; títulos coincidentes/próximos não provam identidade de obra"},
        "counts": totals, "total": total, "split_proportions": {s: n / total for s, n in totals.items()},
        "class_statistics": stats, "all_43_labels_present_in_each_split": True,
        "normalized_titles_unique_per_split": {s: len(m) for s, m in title_maps.items()},
        "overlaps": overlaps, "approximate_overlap": approximate_summary,
        "metrics": model_reports,
        "sources": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()},
    }
    (out_dir / "auditoria_titulos_distribuicao.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Auditoria de distribuição e sensibilidade a títulos", "", "Data de análise: 2026-11-22.", "",
             "Esta é uma análise pós-hoc das previsões preservadas. Não altera o treinamento, os splits ou os resultados originais. A coincidência ou proximidade de títulos é um indicador de possível dependência textual, não prova de duplicação bibliográfica.", "",
             "## Tamanho e distribuição", "", f"Total: {total}; treino {totals['train']}, validação {totals['val']}, teste {totals['test']}. Todos os 43 códigos estão nos três conjuntos.", "",
             "| Conjunto | N | Proporção | Mínimo/código | Mediana/código | Máximo/código |", "|---|---:|---:|---:|---:|---:|"]
    for s in splits:
        v = stats[s]
        lines.append(f"| {s} | {v['n']} | {100 * totals[s] / total:.4f}% | {v['min']} | {v['median']} | {v['max']} |")
    lines.extend(["", "## Sobreposição entre conjuntos", "", "| Pares | Títulos normalizados comuns | Registros no primeiro | Registros no segundo | Pares cruzados | Edition ID | Work ID |", "|---|---:|---:|---:|---:|---:|---:|"])
    for pair, values in overlaps.items():
        a, b = pair.split("/")
        lines.append(f"| {pair} | {values['normalized_title_groups']} | {values['records_' + a]} | {values['records_' + b]} | {values['cross_pairs']} | {values['edition_id']} | {values['work_id']} |")
    lines.extend(["", "## Sensibilidade no teste", "", "O macro-F1 abaixo considera somente os rótulos presentes em cada subconjunto. O CSV/JSON também informa o macro-F1 com os 43 rótulos fixos e a cobertura de classes, para evitar comparar médias com denominadores diferentes sem ressalva.", "",
                  "| Método | Subconjunto | N | Acertos | Acurácia | Macro-F1 (presentes) | Macro-F1 (43 fixos) | Classes presentes |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for model, reports in model_reports.items():
        for group, values in reports.items():
            accuracy, macro, fixed = values["accuracy"], values["macro_f1_labels_present"], values["macro_f1_fixed_43"]
            fmt = lambda x: "—" if x is None else f"{x:.6f}"
            lines.append(f"| {model} | {group} | {values['n']} | {values['hits']} | {fmt(accuracy)} | {fmt(macro)} | {fmt(fixed)} | {values['labels_present_n']} |")
    lines.extend(["", "## Títulos aproximados", "", summary["method"]["approximate_similarity"], "", "Os limiares 0,90 e 0,95 foram definidos para esta auditoria, após a rodada original. A retirada de registros altera a composição e dificuldade da amostra, e diferenças de desempenho não podem ser atribuídas causalmente a vazamento.", ""])
    for tag, values in approximate_summary.items():
        lines.append(f"- Similaridade ≥ {values['threshold']:.2f}: {values['test_records_at_or_above']} registros de teste; {values['test_records_approximate_only']} adicionais aos títulos normalizados idênticos.")
    lines.extend(["", "## Distribuição por código", "", "| CDD | Treino | Validação | Teste | Total |", "|---|---:|---:|---:|---:|"])
    for row in distribution:
        lines.append(f"| {row['cdd']} | {row['train']} | {row['val']} | {row['test']} | {row['total']} |")
    (out_dir / "auditoria_titulos_distribuicao.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"counts": totals, "class_statistics": stats, "overlaps": overlaps, "approximate_overlap": approximate_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
