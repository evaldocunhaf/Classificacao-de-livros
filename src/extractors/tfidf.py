import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .stopwords import STOPWORDS


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n|;", text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def extract_keywords_tfidf(
    text: str,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        token_pattern=r"\b[a-zà-ú][a-zà-ú\-]{2,}\b",
        stop_words=list(STOPWORDS),
    )
    try:
        tfidf = vectorizer.fit_transform(sentences)
    except ValueError as error:
        # Títulos muito curtos ou formados apenas por stopwords são válidos no
        # dataset Open Library; nesse caso o extrator simplesmente não contribui.
        if "empty vocabulary" in str(error):
            return []
        raise

    scores = tfidf.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()
    ranked = [name for _, name in sorted(zip(scores, feature_names), reverse=True) if _ > 0]

    return ranked[:top_n]
