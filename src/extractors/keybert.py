from keybert import KeyBERT

from .stopwords import STOPWORDS

_MODEL_CACHE: dict[str, KeyBERT] = {}


def _get_model(model_name: str) -> KeyBERT:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = KeyBERT(model=model_name)
    return _MODEL_CACHE[model_name]


def extract_keywords_keybert(
    text: str,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    use_mmr: bool = True,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    diversity: float = 0.7,
) -> list[str]:
    model = _get_model(model_name)
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        use_mmr=use_mmr,
        diversity=diversity,
        stop_words=list(STOPWORDS),
        top_n=top_n,
    )
    return [kw for kw, _ in keywords]