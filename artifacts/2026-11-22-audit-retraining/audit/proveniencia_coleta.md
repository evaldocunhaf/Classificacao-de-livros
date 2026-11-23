---
title: Proveniência da coleta Open Library e conferência do recorte
date: 2026-11-22
tags: [tcc, metodologia, proveniencia, reproducibilidade]
created_by: codex
links:
  - "[[TCC/TCC-Paper/TCC-Evaldo/Metodologia]]"
  - "[[TCC/Planejamento/Plano até 31 de Outubro de 2026]]"
---

# Proveniência da coleta e conferência do recorte

Conferência de 22/11/2026, somente leitura, usando código e arquivos preservados. Os caminhos abaixo são relativos à raiz do repositório `tcc-files`, correspondente à pasta `TCC`. Não houve novo treinamento, inferência ou download.

## Fonte e datas

A Open Library disponibiliza [dumps mensais](https://openlibrary.org/developers/dumps) em TSV, com campos de tipo, identificador, revisão, última modificação e JSON.

| Arquivo local | Data nominal do snapshot | Tamanho local |
|---|---|---:|
| `tcc-pipeline/data/ol_dump_editions_2026-08-31.txt` | 31/08/2026 | 61.943.618.646 bytes |
| `tcc-pipeline/data/ol_dump_works_2026-09-30.txt.gz` | 30/09/2026 | 4.073.320.823 bytes |

As datas dos nomes identificam versões nominais; não demonstram o dia do download. Não foram localizados manifesto da obtenção, URL exata do arquivo baixado ou checksum original. Não se calcularam hashes dos dumps gigantes.

A documentação em `Wiki/log.md` registra a sondagem/extração inicial em 21/09/2026. O processamento do dump de obras em 15/10/2026 está registrado em `Planejamento/Plano até 31 de Outubro de 2026.md`, seção “Registro da etapa 2”, e em `Wiki/log.md`, entrada “Corpus descritivo por subjects do dump Open Library”. A rodada final é identificada por `2026-10-16-final-comparison-subjects`.

## Escolha das cinco seções e dos 43 códigos

`Planejamento/Brainstorm-Dados.md`, seção “Sondagem de cinco faixas CDD (2026-09-21)”, documenta que 001, 004, 005, 006 e 025 já estavam presentes no dataset de referência. A seleção não representa toda a CDD.

O CSV histórico `tcc-pipeline/output/datasets/openlibrary_cdd_001_004_005_006_025_5digits.csv` contém 102.323 relações edição–CDD, 101.787 edições distintas e 299 códigos. Exatamente 43 códigos possuíam pelo menos 500 registros; o menor deles tinha 522. Conferiu-se que essa lista coincide integralmente com os 43 códigos finais. Há 7 códigos em 001, 7 em 004, 13 em 005, 8 em 006 e 8 em 025.

## Correção de outubro e limite de 100 por código

O CSV corrigido `openlibrary_cdd_001_004_005_006_025_5digits_one_cdd.csv` contém 16.814 edições distintas, 298 códigos e no máximo 100 registros por código. A regra “primeira CDD válida nas faixas solicitadas, até os primeiros 100 por código em ordem de leitura” foi reconstruída a partir do CSV histórico: reproduziu todas as 16.814 linhas, inclusive a ordem.

Os 4.300 registros de `2026-10-15-openlibrary-one-cdd/prepared/dataset_limpo.csv` são exatamente todas as linhas desse pool que pertencem à lista histórica de 43 códigos, com 100 por código. Não se deve aplicar o limiar 500 novamente ao CSV corrigido, que já foi limitado a 100.

Consequentemente, os exemplos de outubro não constituem amostra aleatória da população completa do dump. O embaralhamento posterior não altera o fato de que a extração retém os primeiros registros elegíveis. Essa dependência da ordem de leitura é uma limitação do recorte.

## Normalização e preparação

O extrator lê o dump em streaming e retém registros de edição com JSON válido, título não vazio e CDD válida pertencente às faixas. A edição recebe a primeira notação válida que corresponde ao recorte, enquanto `cdd_raw` preserva todas as notações de origem. Quando há várias obras associadas, o campo operacional usa o primeiro `work_id`.

O extrator consultado não impõe filtro de idioma, ano de publicação ou ISBN obrigatório. Esses campos não devem ser apresentados como critérios de inclusão desta coleta.

A normalização remove barras de segmentação MARC e limita a notação a cinco algarismos, sem preencher decimais: `303.48/33 → 303.4833 → 303.48`. A preparação remove espaços externos e corta o título na primeira barra `/`, conservando a parte anterior; isso é a regra implementada, não uma garantia de que toda barra separa autoria.

O código permite selecionar a lista de códigos por CSV de referência. Na conferência, `load_and_clean` seguido de `filter_and_cap_classes(..., max_per_class=100, random_state=42)` reproduziu a ordem inteira dos dois `dataset_limpo.csv` de outubro, tanto antes quanto depois do enriquecimento.

## Particionamento e semente

Foram reproduzidos em memória os identificadores de todas as partições com `_split_grouped(dataset_limpo, "work_id", 42)`, obtendo igualdade exata em treino, validação e teste para as rodadas `2026-10-15-openlibrary-one-cdd` e `2026-10-15-openlibrary-subjects-43cdd`.

A implementação usa `StratifiedGroupKFold` em dois passos: cinco partições para retirar o teste de 20%; quatro sobre os 80% restantes para retirar a validação, resultando em 60/20/20. Aplica estratificação por CDD, agrupamento por `work_id`, embaralhamento e semente 42. Identificadores de obra vazios recebem grupos individuais. Antes do filtro descritivo, todos os 43 códigos têm exatamente 60 registros de treino, 20 de validação e 20 de teste.

O repreparo após o enriquecimento alterou os membros dos splits: a mesma contagem não significa os mesmos identificadores. As interseções entre os splits correspondentes de `one-cdd` e `subjects` são 1.628 no treino, 179 na validação e 299 no teste. O filtro final preserva os splits da rodada `subjects`, e não os splits anteriores title-only.

## Enriquecimento e 4.300 → 4.034

O dump de obras é percorrido em streaming para recuperar os `subjects` dos `work_id` necessários. Os termos são limpos nas extremidades e deduplicados por `casefold`, preservando a ordem; permanecem em JSON na coluna `subjects` e são concatenados por quebra de linha em `resumo`, com origem `openlibrary_work_subjects`.

Na rodada final, `resumo` significa assuntos, não um resumo narrativo. O piloto anterior da API com descrições/excertos é uma etapa separada e não constitui a fonte do texto do corpus final.

| Conjunto | Antes do filtro | Com assuntos | Excluídos |
|---|---:|---:|---:|
| Treino | 2.580 | 2.423 | 157 |
| Validação | 860 | 803 | 57 |
| Teste | 860 | 808 | 52 |
| Total | 4.300 | 4.034 | 266 |

A função `_has_descriptive_text` remove conteúdo vazio ou equivalente a `nan`, `none` ou `null`, após limpeza. Todas as 43 classes permanecem nos três conjuntos. A cobertura descritiva é de 93,81%.

## Rastreabilidade desta conferência

- `tcc-pipeline/src/data/extract_openlibrary_editions.py`: primeira CDD válida no recorte e limite por código na ordem de leitura.
- `tcc-pipeline/src/data/cdd.py`: normalização e truncamento.
- `tcc-pipeline/src/data/prepare_dataset.py`: limpeza, seleção e split estratificado/agrupado.
- `tcc-pipeline/src/data/enrich_openlibrary_works_dump.py`: enriquecimento por assuntos.
- `tcc-pipeline/src/evaluate/select_final_models.py`: filtro descritivo final.
- `tcc-pipeline/output/inspections/openlibrary_cdd_001_004_005_006_025.json`: sondagem histórica.
- Auditorias em `output/runs/2026-10-15-openlibrary-one-cdd/audit/` e `output/runs/2026-10-15-openlibrary-subjects-43cdd/audit/`.
- `proveniencia_coleta.json`: resultados estruturados da conferência.
- `conferir_proveniencia_coleta.py`: verificação somente leitura; requer os CSVs locais e pandas/scikit-learn. Não depende de reprocessar os dumps de dezenas de GB.

O script foi executado com Python 3.14, pandas 3.0.6 e scikit-learn 1.9.1; todas as asserções passaram. Essas versões descrevem o ambiente da conferência de novembro, não comprovam o ambiente da execução histórica de outubro.
