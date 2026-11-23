import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.llm.classifier import CDDClassifier
from src.llm.ollama_client import OllamaClient
from src.llm.prompts.cdd_prompt import build_classification_prompt

BASE = Path(__file__).resolve().parent
EXPERIMENTS = {
    "A": {"use_keywords": False, "label": "baseline (texto bruto)"},
    "B": {"use_keywords": True, "label": "keywords (Módulo 1 + votes)"},
}


def _yake_language(row: pd.Series) -> str:
    """Converte o identificador de idioma Open Library para YAKE quando disponível."""
    language = str(row.get("idioma", "")).lower()
    if "eng" in language:
        return "en"
    if "por" in language:
        return "pt"
    return "pt"


def load_config() -> dict:
    with open(BASE / "config.yaml") as f:
        return yaml.safe_load(f)


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))


def sample_per_class(df: pd.DataFrame, max_per_class: int, random_state: int) -> pd.DataFrame:
    """Amostra o mesmo máximo por CDD para tornar avaliações LLM viáveis."""
    if max_per_class < 1:
        raise ValueError("max_per_class deve ser positivo")
    pieces = [
        group.sample(n=min(len(group), max_per_class), random_state=random_state)
        for _, group in df.groupby("cdd", sort=True)
    ]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=random_state)


def only_descriptive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém somente registros com texto adicional para a ablação L0/L1."""
    if "resumo" not in df:
        raise ValueError("split sem coluna resumo; não é possível selecionar texto descritivo")
    summary = df["resumo"].fillna("").astype(str).str.strip()
    return df.loc[summary.ne("")].copy()


def classify_batch(
    df: pd.DataFrame,
    classifier: CDDClassifier,
    model: str,
    use_keywords: bool,
    cache: dict,
    cache_path: Path,
) -> pd.DataFrame:
    model_key = f"{model}:{classifier.llm_cfg}"
    rows = []
    for i, row in df.iterrows():
        text = row["texto"]
        keywords = classifier._extract_keywords(text, _yake_language(row)) if use_keywords else None
        prompt = build_classification_prompt(text, keywords, classifier.candidate_codes)
        key = hashlib.sha256(f"{model_key}:::{prompt}".encode()).hexdigest()

        t0 = time.time()
        if key in cache:
            result = cache[key]
        else:
            result = (
                classifier._classify(text, keywords)
                if use_keywords
                else classifier._classify(text, None)
            )
            cache[key] = result
            save_cache(cache, cache_path)
        elapsed = time.time() - t0

        rows.append(
            {
                "titulo": row["titulo"],
                "source": row.get("source", ""),
                "edition_id": row.get("edition_id", ""),
                "work_id": row.get("work_id", ""),
                "idioma": row.get("idioma", ""),
                "cdd_true": row["cdd"],
                "cdd_pred": result["codigo"],
                "method": result["method"],
                "attempts": result["attempts"],
                "razao": result["razao"],
                "keywords": json.dumps(keywords or [], ensure_ascii=False),
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                "resposta_bruta": result["raw"],
                "tempo_s": round(elapsed, 2),
            }
        )
        if (i + 1) % 10 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] ok")
    return pd.DataFrame(rows)


def run_experiment(
    exp: str,
    split: str,
    df: pd.DataFrame,
    cfg: dict,
    cache: dict,
    cache_path: Path,
    prepared_dir: Path,
    results_dir: Path,
) -> None:
    use_keywords = EXPERIMENTS[exp]["use_keywords"]
    label = EXPERIMENTS[exp]["label"]

    client = OllamaClient(host=cfg["host"], model=cfg["model"])
    classifier = CDDClassifier(
            client,
            temperature=cfg.get("temperature", 0.0),
            extractor_cfg=cfg.get("extractors", {}),
            consolidation_cfg=cfg.get("consolidation", {}),
            llm_cfg=cfg.get("llm", {}),
            train_path=prepared_dir / "train.csv",
            candidate_codes=pd.read_csv(prepared_dir / "train.csv", dtype=str)["cdd"].dropna().tolist(),
        )

    print(f"[{exp}] {label} | modelo={cfg['model']} | n={len(df)}")
    if not client.is_available():
        raise SystemExit("Ollama indisponível. Inicie com: ollama serve")
    results = classify_batch(df, classifier, cfg["model"], use_keywords, cache, cache_path)
    out = results_dir / f"exp_{exp}_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"-> salvo {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TCC pipeline — Módulo 2 (LLM CARP)")
    parser.add_argument("--exp", choices=list(EXPERIMENTS) + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dry-run", action="store_true", help="só testa 2 registros")
    parser.add_argument(
        "--descriptive-only",
        action="store_true",
        help="usa somente registros com texto descritivo (necessário para L0/L1 pareado)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="avalia no máximo esta quantidade de exemplos por CDD e gera arquivos *_capN",
    )
    parser.add_argument("--sample-seed", type=int, default=42, help="seed da amostra por CDD")
    parser.add_argument("--eval", action="store_true", help="só avalia os resultados salvos")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="diretório isolado com prepared/ e results/; sem ele preserva o layout legado",
    )
    args = parser.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        prepared_dir = BASE / "output" / "prepared"
        results_dir = BASE / "output" / "results"
    else:
        if not run_dir.is_absolute():
            run_dir = BASE / run_dir
        prepared_dir = run_dir / "prepared"
        results_dir = run_dir / "results"
    cache_path = results_dir / "cache.json"
    result_split = args.split
    if args.descriptive_only:
        result_split = f"{result_split}_subjects"
    if args.max_per_class is not None:
        result_split = f"{result_split}_cap{args.max_per_class}"

    if args.eval:
        from src.evaluate.metrics import save_eval

        exps = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
        for exp in exps:
            csv_path = results_dir / f"exp_{exp}_{result_split}.csv"
            if not csv_path.exists():
                print(f"sem resultados p/ {exp}/{result_split}: {csv_path.name}")
                continue
            df = pd.read_csv(csv_path, dtype=str)
            out = results_dir / f"avaliacao_{exp}_{result_split}"
            print(f"[eval] {exp} sobre {result_split}")
            from json import dumps as _jd

            print(_jd(save_eval(df, str(out)), ensure_ascii=False, indent=2))
        return

    split = args.split
    cfg = load_config()
    cache = load_cache(cache_path)

    df = pd.read_csv(prepared_dir / f"{split}.csv", dtype=str)
    if args.descriptive_only:
        before = len(df)
        df = only_descriptive_rows(df)
        print(f"recorte descritivo: n={len(df)}/{before}")
    if args.max_per_class is not None:
        df = sample_per_class(df, args.max_per_class, args.sample_seed)
        split = result_split
        print(f"amostra balanceada: n={len(df)}; máximo {args.max_per_class} por CDD")
    if args.dry_run:
        result_split = "_dryrun"
        df = df.head(2)

    exps = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    for exp in exps:
        run_experiment(
            exp, result_split, df, cfg, cache, cache_path, prepared_dir, results_dir
        )


if __name__ == "__main__":
    main()
