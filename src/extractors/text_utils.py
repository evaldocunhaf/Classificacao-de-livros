import re

import unicodedata

from .stopwords import STOPWORDS

PUNCT_RE = re.compile(r"[^\w\s\-.]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def clean_keyword(kw: str) -> str:
    kw = normalize(kw).strip().strip(".").strip()
    return kw if kw else None