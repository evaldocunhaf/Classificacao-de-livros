from yake import KeywordExtractor


def extract_keywords_yake(
    text: str,
    top_n: int = 10,
    language: str = "pt",
    ngram_size: int = 3,
    dedup_lim: float = 0.9,
    dedup_func: str = "seqm",
    window_size: int = 1,
) -> list[str]:
    extractor = KeywordExtractor(
        lan=language,
        n=ngram_size,
        dedupLim=dedup_lim,
        dedupFunc=dedup_func,
        windowsSize=window_size,
        top=top_n,
    )
    keywords = extractor.extract_keywords(text)
    return [kw for kw, _ in keywords][:top_n]