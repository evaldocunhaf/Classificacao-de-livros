from collections import Counter

from .text_utils import clean_keyword
from .tfidf import extract_keywords_tfidf
from .yake import extract_keywords_yake
from .keybert import extract_keywords_keybert


def extract_all(
    text: str,
    top_n: int = 10,
    use_tfidf: bool = True,
    use_keybert: bool = True,
    use_yake: bool = True,
    **kwargs,
) -> dict[str, list[str]]:
    results = {}
    if use_tfidf:
        results["tfidf"] = extract_keywords_tfidf(text, top_n=top_n, **kwargs)
    if use_keybert:
        results["keybert"] = extract_keywords_keybert(text, top_n=top_n, **kwargs)
    if use_yake:
        results["yake"] = extract_keywords_yake(text, top_n=top_n, **kwargs)
    return results


def consolidate(
    method_results: dict[str, list[str]],
    mode: str = "union",
    min_votes: int = 2,
) -> list[str]:
    if mode == "union":
        return _union(method_results)
    if mode == "intersection":
        return list(_intersection(method_results))
    if mode == "votes":
        return _by_votes(method_results, min_votes)
    raise ValueError(f"modo inválido: {mode}")


def _clean_sets(method_results: dict[str, list[str]]) -> list[list[str]]:
    return [
        [k for k in map(clean_keyword, kws) if k]
        for kws in method_results.values()
    ]


def _union(method_results: dict[str, list[str]]) -> list[str]:
    seen = set()
    out = []
    for kws in _clean_sets(method_results):
        for k in kws:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _intersection(method_results: dict[str, list[str]]) -> set[str]:
    sets = _clean_sets(method_results)
    if not sets:
        return set()
    return set(sets[0]).intersection(*sets[1:])


def _by_votes(method_results: dict[str, list[str]], min_votes: int) -> list[str]:
    counter = Counter()
    for kws in _clean_sets(method_results):
        counter.update(set(kws))
    return [k for k, v in counter.most_common() if v >= min_votes]