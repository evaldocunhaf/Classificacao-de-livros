SYSTEM_PROMPT = """Você é um bibliotecário especializado em Classificação Decimal de Dewey (CDD).
Classifique a obra no código CDD que melhor representa seu assunto.
Responda APENAS com um JSON válido, sem markdown, no formato:
{"codigo": "DDD.DD", "razao": "explicação breve"}

Regras:
- O campo "codigo" deve conter somente um código da lista de candidatos fornecida.
- Escolha a notação mais específica que tenha evidência explícita no texto.
- Se o texto for vago, prefira a classe mais geral em vez de inventar detalhes."""


def _format_candidates(candidate_codes: list[str] | tuple[str, ...] | None) -> str:
    if not candidate_codes:
        return "(nenhum candidato fornecido)"
    return "\n".join(f"- {code}" for code in candidate_codes)


def build_classification_prompt(
    text: str,
    keywords: list[str] | None = None,
    candidate_codes: list[str] | tuple[str, ...] | None = None,
) -> str:
    parts = [SYSTEM_PROMPT, "", "## Códigos CDD candidatos", _format_candidates(candidate_codes)]

    parts.append("")
    parts.append("## Texto da obra")
    text = text.strip()
    parts.append(text if text else "(sem texto)")

    if keywords:
        parts.append("")
        parts.append("## Pistas (palavras-chave extraídas)")
        parts.append("\n".join(f"- {k}" for k in keywords))
        clue_source = "texto e nas palavras-chave (pistas)"
    else:
        clue_source = "texto"
    parts.append("")

    parts.append(
        "## Tarefa (raciocínio CARP)\n"
        "1. CLUE — identifique os termos-indício relevantes para classificação no " + clue_source + ".\n"
        "2. REASON — relacione cada pista à classe/subclasse CDD correspondente e justifique.\n"
        "3. PREDICT — escolha o código CDD mais específico e responda no JSON."
    )

    return "\n".join(parts)


def classification_prompt_for_text(text: str, candidate_codes: list[str] | tuple[str, ...] | None = None) -> str:
    return build_classification_prompt(text, keywords=None, candidate_codes=candidate_codes)


def classification_prompt_for_keywords(
    text: str,
    keywords: list[str],
    candidate_codes: list[str] | tuple[str, ...] | None = None,
) -> str:
    return build_classification_prompt(text, keywords=keywords, candidate_codes=candidate_codes)
