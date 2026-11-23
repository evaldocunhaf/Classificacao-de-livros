#!/usr/bin/env python3
"""Build the allowlisted, auditable publication package without changing sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

RUN = "2026-10-16-final-comparison-subjects"
PREVIOUS_RUN = "2026-10-15-openlibrary-subjects-43cdd"
SOURCE_COMMIT = "564ef1aaaf9ec72f15797459a29f4d3b8d6c160b"
AUDIT_FILES = (
    "auditar_titulos_distribuicao.py", "auditoria_titulos_distribuicao.json",
    "auditoria_titulos_distribuicao.md", "distribuicao_43_cdd.csv",
    "titulos_coincidentes_entre_splits.csv", "titulo_treino_mais_proximo_por_teste.csv",
    "metricas_sensibilidade_titulos.csv",
)
OPTIONAL_PROVENANCE = ("proveniencia_coleta.md", "proveniencia_coleta.json", "conferir_proveniencia_coleta.py", "conferir_checkpoints.py", "checkpoints_e_splits.json")
PRIVATE_LLM_COLUMNS = {"razao", "resposta_bruta", "keywords", "tempo_s"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_json(value, replacements):
    if isinstance(value, dict):
        return {key: "models/bert_attention" if key == "xlmr_checkpoint"
                else sanitize_json(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    return value


def public_llm_csv(data: bytes) -> bytes:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    allowed = {"titulo", "source", "edition_id", "work_id", "idioma", "cdd_true", "cdd_pred", "method"}
    fields = [name for name in reader.fieldnames or [] if name in allowed]
    if not {"edition_id", "cdd_true", "cdd_pred"}.issubset(fields):
        raise ValueError("LLM predictions lack required audit columns")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in reader)
    return output.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pipeline", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--publication-date", default="2026-11-22")
    args = parser.parse_args()
    source = args.source_pipeline.resolve()
    audit = args.audit_dir.resolve()
    pipeline = Path(__file__).resolve().parents[1]
    destination = (args.destination or pipeline / "artifacts" / RUN).resolve()
    run = source / "output" / "runs" / RUN
    previous = source / "output" / "runs" / PREVIOUS_RUN
    replacements = [(str(source), "tcc-pipeline"), (str(audit), "audit-workspace"),
                    (str(source.parent), "repository")]
    entries = []

    def add(source_path, package_path, mode="copy", origin=None):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        entries.append((source_path, package_path, mode,
                        origin or "tcc-pipeline/" + str(source_path.relative_to(source))))

    for split in ("train", "val", "test"):
        add(run / "prepared" / (split + ".csv"), "prepared/" + split + ".csv")
    add(previous / "prepared" / "dataset_limpo.csv", "prepared/cohort_before_subject_filter.csv")
    add(run / "prepared" / "recorte_descritivo.json", "prepared/recorte_descritivo.json", "json")
    for model in ("E", "F", "G", "logreg"):
        for split in ("val", "test"):
            stem = model + "_" + split
            add(run / "results" / ("exp_" + stem + ".csv"), "results/exp_" + stem + ".csv")
            add(run / "results" / ("avaliacao_" + stem + ".json"),
                "results/avaliacao_" + stem + ".json", "json")
    add(run / "results" / "exp_A_test_subjects.csv", "results/exp_A_test_subjects.csv", "llm_csv")
    add(run / "results" / "avaliacao_A_test_subjects.json", "results/avaliacao_A_test_subjects.json", "json")
    for name in ("selecao_E_val.csv", "selecao_logreg_val.csv", "selecao_G_val.csv",
                 "escores_G_val.npz", "escores_G_test.npz", "tabela_resultados_teste.csv",
                 "validacao_pareamento_teste.json", "selecao_final.json"):
        add(run / "results" / name, "results/" + name, "json" if name.endswith(".json") else "copy")
    for condition in ("A", "B"):
        for prefix, suffix, mode in (("exp_", ".csv", "llm_csv"), ("avaliacao_", ".json", "json")):
            name = prefix + condition + "_val_subjects" + suffix
            add(previous / "results" / name, "results/complementary_llm/" + name, mode)
    add(previous / "results" / "comparacao_l0_l1" / "comparacao_l0_l1.json",
        "results/complementary_llm/comparacao_l0_l1.json", "json")
    for name in AUDIT_FILES + tuple(name for name in OPTIONAL_PROVENANCE if (audit / name).is_file()):
        mode = "json" if name.endswith(".json") else "text" if name.endswith(".md") else "copy"
        add(audit / name, "audit/" + name, mode, "audit-workspace/" + name)
    for name in ("gerar_fluxo_metodologia.py", "fluxo-metodologia.svg", "fluxo-metodologia.png", "auditar_resultados.py"):
        if (audit / name).is_file():
            add(audit / name, "audit/" + name, "copy", "audit-workspace/" + name)
    historical_audit = audit / "resultados-historicos"
    for name in ("auditoria_resultados.json", "metricas_validacao_teste_bootstrap.csv", "mcnemar_10_pares.csv", "stuart_maxwell_10_pares.csv", "profundidade_e_niveis.csv", "confusao_secoes_RL.csv", "confusao_secoes_F.csv", "confusao_secoes_G.csv", "amostra_90_erros.csv", "textos_identicos_referencias_distintas.csv"):
        if (historical_audit / name).is_file():
            add(historical_audit / name, "audit/historical_results/" + name, "json" if name.endswith(".json") else "copy", "audit-workspace/resultados-historicos/" + name)
    readme_source = pipeline / "tools" / "final_artifacts_README.md"
    add(readme_source, "README.md", "copy", "tcc-pipeline/tools/final_artifacts_README.md")

    expected = {package_path for _, package_path, _, _ in entries} | {"manifest.json"}
    if destination.exists():
        unexpected = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()} - expected
        if unexpected:
            raise ValueError("Destination contains files outside the allowlist: " + ", ".join(sorted(unexpected)))
    records = []
    for source_path, package_path, mode, origin in entries:
        original = source_path.read_bytes()
        published = original
        transformation = "unchanged copy"
        if mode == "json":
            published = (json.dumps(sanitize_json(json.loads(original), replacements),
                                    ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            transformation = "JSON reserialized; absolute local paths replaced by portable paths; xlmr_checkpoint made relative"
        elif mode == "llm_csv":
            published = public_llm_csv(original)
            transformation = "CSV reserialized; excluded columns: " + ", ".join(sorted(PRIVATE_LLM_COLUMNS))
        elif mode == "text":
            text = original.decode("utf-8")
            for old, new in replacements:
                text = text.replace(old, new)
            published = text.encode("utf-8")
            transformation = "local source paths replaced by portable paths when present"
        target = destination / package_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(published)
        records.append({"path": package_path, "size_bytes": len(published), "sha256": sha256(published),
                        "source_path": origin, "source_sha256": sha256(original), "transformation": transformation})
    manifest = {
        "schema_version": 1, "run": RUN, "publication_date": args.publication_date,
        "source_repository": "https://github.com/evaldocunhaf/tcc-files",
        "source_code_commit": args.source_commit,
        "source_commit_scope": "historical code consulted; manuscript working-tree edits are not included",
        "limitations": ["No neural checkpoint, source dumps, caches or raw LLM replies are distributed.",
                        "Training arguments and historical dependency versions were not preserved.",
                        "Archived neural validation scores used by the ensemble differ from exp_F_val.csv.",
                        "Title-overlap sensitivity is post-hoc and does not establish bibliographic duplication."],
        "file_count": len(records), "total_bytes": sum(row["size_bytes"] for row in records),
        "files": sorted(records, key=lambda row: row["path"]),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(destination), "files": len(records),
                      "bytes_excluding_manifest": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
