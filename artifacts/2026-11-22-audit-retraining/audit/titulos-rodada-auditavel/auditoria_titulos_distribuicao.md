# Auditoria de distribuição e sensibilidade a títulos

Data de análise: 2026-11-22.

Esta é uma análise pós-hoc das previsões preservadas. Não altera o treinamento, os splits ou os resultados originais. A coincidência ou proximidade de títulos é um indicador de possível dependência textual, não prova de duplicação bibliográfica.

## Tamanho e distribuição

Total: 4034; treino 2423, validação 803, teste 808. Todos os 43 códigos estão nos três conjuntos.

| Conjunto | N | Proporção | Mínimo/código | Mediana/código | Máximo/código |
|---|---:|---:|---:|---:|---:|
| train | 2423 | 60.0645% | 47 | 57 | 60 |
| val | 803 | 19.9058% | 15 | 19 | 20 |
| test | 808 | 20.0297% | 16 | 19 | 20 |

## Sobreposição entre conjuntos

| Pares | Títulos normalizados comuns | Registros no primeiro | Registros no segundo | Pares cruzados | Edition ID | Work ID |
|---|---:|---:|---:|---:|---:|---:|
| train/val | 25 | 34 | 30 | 39 | 0 | 0 |
| train/test | 25 | 39 | 35 | 68 | 0 | 0 |
| val/test | 11 | 12 | 17 | 18 | 0 | 0 |

## Sensibilidade no teste

O macro-F1 abaixo considera somente os rótulos presentes em cada subconjunto. O CSV/JSON também informa o macro-F1 com os 43 rótulos fixos e a cobertura de classes, para evitar comparar médias com denominadores diferentes sem ressalva.

| Método | Subconjunto | N | Acertos | Acurácia | Macro-F1 (presentes) | Macro-F1 (43 fixos) | Classes presentes |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + regressão logística | full_test | 808 | 529 | 0.654703 | 0.649260 | 0.649260 | 43 |
| TF-IDF + regressão logística | exact_title_overlap_train | 35 | 24 | 0.685714 | 0.622727 | 0.318605 | 22 |
| TF-IDF + regressão logística | without_exact_title_overlap_train | 773 | 505 | 0.653299 | 0.645716 | 0.645716 | 43 |
| TF-IDF + regressão logística | title_similarity_at_least_90 | 58 | 43 | 0.741379 | 0.678571 | 0.473422 | 30 |
| TF-IDF + regressão logística | without_title_similarity_at_least_90 | 750 | 486 | 0.648000 | 0.639619 | 0.639619 | 43 |
| TF-IDF + regressão logística | title_similarity_at_least_95 | 43 | 31 | 0.720930 | 0.660494 | 0.414729 | 27 |
| TF-IDF + regressão logística | without_title_similarity_at_least_95 | 765 | 498 | 0.650980 | 0.643170 | 0.643170 | 43 |
| XLM-R-AA | full_test | 808 | 470 | 0.581683 | 0.577291 | 0.577291 | 43 |
| XLM-R-AA | exact_title_overlap_train | 35 | 22 | 0.628571 | 0.512121 | 0.262016 | 22 |
| XLM-R-AA | without_exact_title_overlap_train | 773 | 448 | 0.579560 | 0.572815 | 0.572815 | 43 |
| XLM-R-AA | title_similarity_at_least_90 | 58 | 35 | 0.603448 | 0.555238 | 0.387375 | 30 |
| XLM-R-AA | without_title_similarity_at_least_90 | 750 | 435 | 0.580000 | 0.571955 | 0.571955 | 43 |
| XLM-R-AA | title_similarity_at_least_95 | 43 | 27 | 0.627907 | 0.528395 | 0.331783 | 27 |
| XLM-R-AA | without_title_similarity_at_least_95 | 765 | 443 | 0.579085 | 0.571409 | 0.571409 | 43 |
| Ensemble XLM-R-AA + kNN | full_test | 808 | 484 | 0.599010 | 0.593055 | 0.593055 | 43 |
| Ensemble XLM-R-AA + kNN | exact_title_overlap_train | 35 | 21 | 0.600000 | 0.509091 | 0.260465 | 22 |
| Ensemble XLM-R-AA + kNN | without_exact_title_overlap_train | 773 | 463 | 0.598965 | 0.591069 | 0.591069 | 43 |
| Ensemble XLM-R-AA + kNN | title_similarity_at_least_90 | 58 | 37 | 0.637931 | 0.598889 | 0.417829 | 30 |
| Ensemble XLM-R-AA + kNN | without_title_similarity_at_least_90 | 750 | 447 | 0.596000 | 0.586378 | 0.586378 | 43 |
| Ensemble XLM-R-AA + kNN | title_similarity_at_least_95 | 43 | 26 | 0.604651 | 0.525926 | 0.330233 | 27 |
| Ensemble XLM-R-AA + kNN | without_title_similarity_at_least_95 | 765 | 458 | 0.598693 | 0.589692 | 0.589692 | 43 |
| kNN/TF-IDF (complementar) | full_test | 808 | 372 | 0.460396 | 0.451437 | 0.451437 | 43 |
| kNN/TF-IDF (complementar) | exact_title_overlap_train | 35 | 19 | 0.542857 | 0.407576 | 0.208527 | 22 |
| kNN/TF-IDF (complementar) | without_exact_title_overlap_train | 773 | 353 | 0.456662 | 0.449025 | 0.449025 | 43 |
| kNN/TF-IDF (complementar) | title_similarity_at_least_90 | 58 | 38 | 0.655172 | 0.564762 | 0.394020 | 30 |
| kNN/TF-IDF (complementar) | without_title_similarity_at_least_90 | 750 | 334 | 0.445333 | 0.437476 | 0.437476 | 43 |
| kNN/TF-IDF (complementar) | title_similarity_at_least_95 | 43 | 26 | 0.604651 | 0.489065 | 0.307087 | 27 |
| kNN/TF-IDF (complementar) | without_title_similarity_at_least_95 | 765 | 346 | 0.452288 | 0.444534 | 0.444534 | 43 |
| LLM L0 (complementar) | full_test | 808 | 76 | 0.094059 | 0.061716 | 0.061716 | 43 |
| LLM L0 (complementar) | exact_title_overlap_train | 35 | 6 | 0.171429 | 0.122294 | 0.062569 | 22 |
| LLM L0 (complementar) | without_exact_title_overlap_train | 773 | 70 | 0.090556 | 0.058721 | 0.058721 | 43 |
| LLM L0 (complementar) | title_similarity_at_least_90 | 58 | 7 | 0.120690 | 0.078954 | 0.055084 | 30 |
| LLM L0 (complementar) | without_title_similarity_at_least_90 | 750 | 69 | 0.092000 | 0.059927 | 0.059927 | 43 |
| LLM L0 (complementar) | title_similarity_at_least_95 | 43 | 6 | 0.139535 | 0.093827 | 0.058915 | 27 |
| LLM L0 (complementar) | without_title_similarity_at_least_95 | 765 | 70 | 0.091503 | 0.059005 | 0.059005 | 43 |

## Títulos aproximados

difflib.SequenceMatcher(None, normalized_test, normalized_train, autojunk=False).ratio(); máximo exato sobre títulos normalizados distintos do treino

Os limiares 0,90 e 0,95 foram definidos para esta auditoria, após a rodada original. A retirada de registros altera a composição e dificuldade da amostra, e diferenças de desempenho não podem ser atribuídas causalmente a vazamento.

- Similaridade ≥ 0.90: 58 registros de teste; 23 adicionais aos títulos normalizados idênticos.
- Similaridade ≥ 0.95: 43 registros de teste; 8 adicionais aos títulos normalizados idênticos.

## Distribuição por código

| CDD | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| 001.42 | 58 | 20 | 18 | 96 |
| 001.43 | 59 | 18 | 19 | 96 |
| 001.51 | 60 | 20 | 20 | 100 |
| 001.53 | 60 | 17 | 20 | 97 |
| 001.64 | 58 | 19 | 19 | 96 |
| 001.9 | 55 | 19 | 18 | 92 |
| 001.94 | 55 | 18 | 19 | 92 |
| 004 | 52 | 18 | 19 | 89 |
| 004.01 | 57 | 19 | 19 | 95 |
| 004.16 | 56 | 20 | 18 | 94 |
| 004.35 | 59 | 20 | 19 | 98 |
| 004.6 | 53 | 18 | 20 | 91 |
| 004.67 | 53 | 18 | 17 | 88 |
| 004.68 | 59 | 18 | 19 | 96 |
| 005.1 | 59 | 19 | 17 | 95 |
| 005.11 | 59 | 18 | 20 | 97 |
| 005.13 | 57 | 20 | 18 | 95 |
| 005.26 | 57 | 20 | 18 | 95 |
| 005.27 | 57 | 18 | 19 | 94 |
| 005.36 | 54 | 19 | 20 | 93 |
| 005.43 | 58 | 20 | 20 | 98 |
| 005.44 | 60 | 18 | 20 | 98 |
| 005.71 | 58 | 20 | 20 | 98 |
| 005.72 | 59 | 20 | 19 | 98 |
| 005.74 | 58 | 20 | 18 | 96 |
| 005.75 | 58 | 19 | 18 | 95 |
| 005.8 | 54 | 18 | 16 | 88 |
| 006.3 | 50 | 17 | 18 | 85 |
| 006.31 | 47 | 15 | 16 | 78 |
| 006.33 | 57 | 18 | 20 | 95 |
| 006.6 | 58 | 19 | 20 | 97 |
| 006.68 | 55 | 18 | 19 | 92 |
| 006.69 | 54 | 16 | 20 | 90 |
| 006.7 | 56 | 19 | 19 | 94 |
| 006.78 | 57 | 19 | 19 | 95 |
| 025.04 | 58 | 20 | 16 | 94 |
| 025.06 | 54 | 19 | 20 | 93 |
| 025.17 | 55 | 18 | 19 | 92 |
| 025.3 | 57 | 19 | 18 | 94 |
| 025.34 | 58 | 20 | 19 | 97 |
| 025.46 | 56 | 20 | 20 | 96 |
| 025.49 | 57 | 19 | 19 | 95 |
| 025.52 | 52 | 16 | 19 | 87 |
