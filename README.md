# Pipeline de classificação CDD

## Rodada auditável e resultados utilizados no artigo

A [rodada auditável de retreinamento](artifacts/2026-11-22-audit-retraining/README.md)
usa as partições atuais e três sementes (42, 43 e 44), com argumentos, versões,
logs, predições, escores e manifesto SHA-256. A execução detalhada de semente 42
obteve 65,47% de acurácia na regressão logística, 58,17% no XLM-R-AA e 59,90%
no ensemble. O teste já havia sido examinado; essa rodada não é um teste externo
independente. O repositório permanece privado.

O pacote permite recalcular métricas, intervalos, testes pareados e análises de
títulos sem baixar modelos ou dumps. O README do pacote contém os comandos de
conferência e retreinamento. Os pesos do encoder XLM-R ficaram congelados; apenas
as camadas adicionais foram treinadas. Os novos resultados substituem os valores
neurais históricos na redação atual do artigo.

## Preservação da rodada histórica

Os [artefatos da rodada histórica de outubro](artifacts/2026-10-16-final-comparison-subjects/README.md)
incluem os conjuntos reduzidos, predições de validação e teste, métricas, seleções,
escores do ensemble, distribuição por CDD, auditoria de títulos e manifesto SHA-256.
O pacote pode ser conferido sem treinar modelos ou baixar dumps. Seu preparo não
altera a visibilidade do repositório, cujo acesso era restrito na conferência.

Há uma limitação preservada: os escores neurais de validação usados pelo ensemble
produzem 315/803 acertos, enquanto o CSV neural isolado registra 480/803. O teste
é aritmeticamente consistente com os escores publicados, mas não se comprova a
seleção do peso nas condições do mesmo checkpoint final. Veja a descrição no pacote.

O `requirements.txt` declara as dependências do código disponível, incluindo
PyTorch e Transformers. Não representa o ambiente histórico exato: as versões
e os argumentos efetivos do treino de outubro não foram preservados. Os comandos
abaixo documentam os fluxos implementados e exemplos de uso; não constituem um
registro integral dos comandos efetivamente executados na rodada final.

## Exemplos históricos de uso do pipeline

Os comandos abaixo ilustram os módulos e os caminhos usados no desenvolvimento.
Para reproduzir a rodada utilizada no artigo, siga o pacote de retreinamento acima.

### Dataset Open Library

O dump bruto fica em `data/` e é ignorado pelo Git. O extrator lê o TSV linha a
linha, sem carregar o arquivo inteiro na memória, e gera um CSV rastreável.

Antes de gerar dados, faça uma sondagem das cinco faixas iniciais. Ela não cria
CSV e limita a CDD a cinco algarismos (`303.4833` vira `303.48`):

```bash
python -m src.data.inspect_openlibrary_cdd \
  --input data/ol_dump_editions_2026-08-31.txt \
  --cdd-prefix 001 004 005 006 025 \
  --max-digits 5 \
  --output output/inspections/openlibrary_cdd_probe.json
```

```bash
python -m src.data.extract_openlibrary_editions \
  --input data/ol_dump_editions_2026-08-31.txt \
  --output output/datasets/openlibrary_editions.csv \
  --cdd-prefix 000 100 200 300 400 500 600 700 800 900 \
  --max-digits 5 \
  --max-records 20000
```

Cada linha produzida tem `titulo`, `resumo` (vazio nesta primeira etapa), `cdd`,
`cdd_raw`, `source`, `edition_id`, `work_id`, ISBN e idioma. Ao baixar o dump de
obras, use `work_id` para o enriquecimento. No corpus final, `resumo` contém
assuntos (`subjects`) do dump de obras, e não descrições narrativas; o piloto
de descrições via API é uma etapa distinta.

Prepare uma rodada isolada sem misturar edições da mesma obra entre os splits:

```bash
python -m src.data.prepare_dataset \
  --input output/datasets/openlibrary_editions.csv \
  --output output/runs/openlibrary-v1/prepared \
  --group-column work_id
```

Execute os experimentos LLM e avalie-os:

```bash
python main.py --exp all --split test --run-dir output/runs/openlibrary-v1
python main.py --exp all --split test --eval --run-dir output/runs/openlibrary-v1
python -m src.bert_attention.train --run-dir output/runs/openlibrary-v1
```

Os resultados, cache e checkpoint ficam em `output/runs/openlibrary-v1/`; os
resultados anteriores permanecem intactos.

### Experimento E — baseline kNN/TF-IDF (*title-only*)

O baseline seleciona a melhor combinação de representação (palavra, caractere
ou ambas) e `k` **somente na validação**. Depois gera a avaliação do teste uma
única vez, em uma pasta diferente da rodada que forneceu os splits:

```bash
python -m src.evaluate.knn_baseline \
  --prepared-dir output/runs/2026-09-21-openlibrary-five-cdd/prepared \
  --output-dir output/runs/2026-10-05-knn-title-only
```

Artefatos: `selecao_E_val.csv` (todas as configurações), `exp_E_val.csv`,
`exp_E_test.csv`, as avaliações JSON/matrizes de confusão e `resumo_E.json`.
O experimento não usa `resumo`, palavras-chave nem LLM; portanto mede apenas o
sinal recuperável dos títulos.

### Experimento G — ensemble XLM-R + kNN

O ensemble reaproveita o checkpoint do Experimento F e o kNN escolhido no E.
Ele testa pesos de 0 a 1 na **validação**, congela o melhor e só então avalia o
teste. Nenhum modelo é retreinado e a rodada original é preservada:

```bash
python -m src.evaluate.ensemble_xlmr_knn \
  --prepared-dir output/runs/2026-09-21-openlibrary-five-cdd/prepared \
  --checkpoint-dir output/runs/2026-09-21-openlibrary-five-cdd/models/bert_attention \
  --output-dir output/runs/2026-10-06-ensemble-xlm-knn
```

O diretório de saída contém as probabilidades comprimidas dos dois modelos,
a tabela `selecao_G_val.csv`, predições/avaliações de validação e teste e o
`resumo_G.json` com o peso escolhido.

### Piloto de resumos por obra

Quando o dump de obras não estiver disponível, use a API apenas para um piloto
pequeno. O comando preserva um cache JSONL e preenche `resumo` primeiro com
`description`, depois com `excerpts`:

```bash
python -m src.data.enrich_openlibrary_works \
  --input output/runs/2026-09-21-openlibrary-five-cdd/prepared/dataset_limpo.csv \
  --output output/datasets/openlibrary_five_cdd_enriched_pilot.csv \
  --cache output/cache/openlibrary_work_enrichment.jsonl \
  --max-works 300
```

Para milhares de obras, baixar o dump de obras é mais adequado do que fazer
milhares de requisições à API pública.
