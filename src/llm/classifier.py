import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..data.cdd import truncate_cdd
from ..extractors.consolidation import extract_all, consolidate
from .ollama_client import OllamaClient
from .prompts.cdd_prompt import build_classification_prompt

CDD_FULL_RE = re.compile(r"^\d{3}(?:\.\d+)?$")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
RETRY_HINT = (
    "\nATENÇÃO: sua resposta anterior não foi um código CDD válido. "
    "Responda apenas o JSON com o campo \"codigo\" usando uma notação CDD válida."
)


def parse_codigo(raw: str, candidate_codes: set[str] | None = None) -> str | None:
    """Lê apenas o campo JSON pedido pelo prompt; não caça números soltos."""
    m = JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    for key in ("codigo", "code", "codigo_cdd"):
        value = str(data.get(key, "")).strip()
        if not value or not CDD_FULL_RE.fullmatch(value):
            continue
        codigo = truncate_cdd(value, max_digits=5)
        if codigo and (candidate_codes is None or codigo in candidate_codes):
            return codigo
    return None


@lru_cache(maxsize=8)
def _tfidf_model(train_path: str, max_features: int = 5000):
    df = pd.read_csv(train_path, dtype=str)
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"\b[a-zà-ú][a-zà-ú\-]{2,}\b",
        max_features=max_features,
    )
    dtm = tfidf.fit_transform(df["texto"].fillna(""))
    return tfidf, dtm, df


def knn_predict(query_text: str, train_path: str, k: int = 5) -> tuple[str, str]:
    tfidf, dtm, df = _tfidf_model(str(train_path))
    q = tfidf.transform([query_text])
    sim = cosine_similarity(q, dtm).ravel()
    idx = sim.argsort()[::-1][:k]
    neighbors = df.iloc[idx].assign(sim=sim[idx])
    best = (
        neighbors.groupby("cdd")["sim"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    reason = f"kNN fallback (k={k}, vizinhos mais próximos do corpus treino)"
    return best, reason


class CDDClassifier:
    def __init__(
        self,
        client: OllamaClient,
        temperature: float = 0.0,
        extractor_cfg: dict | None = None,
        consolidation_cfg: dict | None = None,
        llm_cfg: dict | None = None,
        train_path: Path | None = None,
        candidate_codes: list[str] | tuple[str, ...] | None = None,
    ):
        self.client = client
        self.temperature = temperature
        self.extractor_cfg = extractor_cfg or {}
        self.consolidation_cfg = consolidation_cfg or {"mode": "votes", "min_votes": 2}
        self.llm_cfg = llm_cfg or {}
        self.train_path = train_path
        self.candidate_codes = tuple(sorted(set(candidate_codes or ())))
        self._candidate_set = set(self.candidate_codes)

    def _extract_keywords(self, text: str, language: str | None = None) -> list[str]:
        """Extrai keywords respeitando a configuração por extrator do YAML."""
        tfidf_cfg = dict(self.extractor_cfg.get("tfidf", {}))
        keybert_cfg = dict(self.extractor_cfg.get("keybert", {}))
        yake_cfg = dict(self.extractor_cfg.get("yake", {}))
        results = {}
        if tfidf_cfg.get("enabled", True):
            ngram = tuple(tfidf_cfg.get("ngram_range", (1, 2)))
            results["tfidf"] = extract_all(
                text, top_n=tfidf_cfg.get("top_n", 10), use_keybert=False, use_yake=False,
                ngram_range=ngram,
            )["tfidf"]
        if keybert_cfg.get("enabled", True):
            ngram = tuple(keybert_cfg.get("ngram_range", (1, 2)))
            results["keybert"] = extract_all(
                text, top_n=keybert_cfg.get("top_n", 10), use_tfidf=False, use_yake=False,
                ngram_range=ngram,
                model_name=keybert_cfg.get("model", "paraphrase-multilingual-MiniLM-L12-v2"),
                use_mmr=keybert_cfg.get("use_mmr", True),
            )["keybert"]
        if yake_cfg.get("enabled", True):
            results["yake"] = extract_all(
                text, top_n=yake_cfg.get("top_n", 10), use_tfidf=False, use_keybert=False,
                language=language or yake_cfg.get("language", "pt"),
            )["yake"]
        mode = self.consolidation_cfg.get("mode", "votes")
        min_votes = self.consolidation_cfg.get("min_votes", 2)
        return consolidate(results, mode=mode, min_votes=min_votes)

    def _call_llm(self, prompt: str) -> str:
        return self.client.generate(
            prompt,
            temperature=self.temperature,
            think=self.llm_cfg.get("think", False),
            num_predict=self.llm_cfg.get("max_output_tokens"),
        )

    def _valid_codigo(self, codigo: str | None) -> str | None:
        if codigo and CDD_FULL_RE.fullmatch(codigo):
            canonical = truncate_cdd(codigo, max_digits=5)
            if canonical and (not self._candidate_set or canonical in self._candidate_set):
                return canonical
        return None

    def classify(self, text: str, use_keywords: bool = True) -> dict:
        keywords = self._extract_keywords(text) if use_keywords else None
        return self._classify(text, keywords)

    def classify_raw(self, text: str) -> dict:
        return self._classify(text, None)

    def classify_with_keywords(self, text: str, keywords: list[str]) -> dict:
        return self._classify(text, keywords)

    def prompt_for(self, text: str, use_keywords: bool = True) -> str:
        keywords = self._extract_keywords(text) if use_keywords else None
        return build_classification_prompt(text, keywords, self.candidate_codes)

    def _classify(self, text: str, keywords: list[str] | None) -> dict:
        prompt = build_classification_prompt(text, keywords, self.candidate_codes)

        raw = self._call_llm(prompt)
        codigo = self._valid_codigo(parse_codigo(raw, self._candidate_set or None))
        attempts = 1
        razao = _extract_reason(raw)

        if codigo is None:
            raw = self._call_llm(prompt + RETRY_HINT)
            attempts = 2
            codigo = self._valid_codigo(parse_codigo(raw, self._candidate_set or None))
            razao = _extract_reason(raw) or razao

        if codigo is None and self.train_path is not None:
            codigo, reason = knn_predict(text, self.train_path)
            razao = f"{reason} | razão LLM: {razao or 'não informada'}"
            return {
                "codigo": codigo,
                "razao": razao,
                "method": "knn",
                "attempts": attempts,
                "raw": raw,
            }

        return {
            "codigo": codigo,
            "razao": razao,
            "method": "llm" if codigo else "none",
            "attempts": attempts,
            "raw": raw,
        }


def _extract_reason(raw: str) -> str:
    m = JSON_BLOCK_RE.search(raw)
    if m:
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return ""
        if isinstance(data, dict):
            return data.get("razao", "") or ""
    return ""
