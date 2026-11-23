"""Normalização conservadora de notações da Classificação Decimal de Dewey."""

import re

CDD_RE = re.compile(r"(?<!\d)(\d{3})(?:\.(\d+))?")


def normalize_cdd(raw: object) -> str:
    """Extrai uma notação CDD canônica sem descartar a precisão decimal.

    Alguns catálogos representam o ponto decimal como ``/.`` (por exemplo,
    ``028/.9``). O valor original deve ser preservado em ``cdd_raw`` para
    auditoria; esta função só produz a coluna operacional ``cdd``.
    """
    if raw is None:
        return ""
    # Open Library preserva a barra tipográfica usada por alguns registros
    # MARC: `028/.9` significa `028.9` e `303.48/33` significa `303.4833`.
    value = str(raw).strip().replace("/", "")
    match = CDD_RE.search(value)
    if not match:
        return ""
    base, decimal = match.groups()
    return f"{base}.{decimal}" if decimal else base


def truncate_cdd(raw: object, max_digits: int = 5) -> str:
    """Normaliza uma CDD e limita sua notação ao total de algarismos indicado.

    Preserva os três algarismos da classe principal. Com o padrão de cinco,
    por exemplo, ``303.4833`` passa a ser ``303.48``.
    """
    if max_digits < 3:
        raise ValueError("max_digits deve ser ao menos 3 para preservar a classe CDD")

    normalized = normalize_cdd(raw)
    if not normalized:
        return ""

    base, separator, decimal = normalized.partition(".")
    remaining_digits = max_digits - len(base)
    if not separator or remaining_digits <= 0:
        return base[:max_digits]

    truncated_decimal = decimal[:remaining_digits]
    return f"{base}.{truncated_decimal}" if truncated_decimal else base
