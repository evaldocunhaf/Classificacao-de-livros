# Rodada auditável: três sementes no recorte atual

Este pacote corresponde ao retreinamento documentado iniciado na revisão de 22/11/2026. Os manifestos contêm os horários UTC efetivos. O código-base dos classificadores é o commit `564ef1aaaf9ec72f15797459a29f4d3b8d6c160b`; as funções de classificação não foram alteradas para essa rodada.

Este pacote está disponível no repositório público Classificacao-de-livros. Para citar uma versão fixa, use o endereço do pacote com o hash do commit.

## Desenho e interpretação

- Partições fixas: 2.423 exemplos de treino, 803 de validação e 808 de teste; 43 códigos; sem interseção de `edition_id` ou `work_id` entre conjuntos.
- Encoder `xlm-roberta-base` congelado, arquitetura adicional preservada; treino AdamW, lote 16, taxa 0,0005, weight decay 0,01, até 60 épocas, paciência 8 e pesos de classe balanceados.
- Sementes de treino: 42, 43 e 44. A execução 42 é detalhada; as demais são replicações. Não foi escolhida a melhor semente pelo teste.
- Cada checkpoint foi selecionado na validação. Depois foram selecionados C, kNN e peso do ensemble na validação correspondente, antes da avaliação de teste.
- Pesos neurais do ensemble: 0,50 / 0,70 / 0,55. Os escores de validação e teste conferem integralmente com os CSVs de cada execução.
- O teste já havia sido inspecionado em rodadas anteriores. Este retreinamento resolve a origem documentada do novo checkpoint e a inconsistência da seleção, mas não constitui novo teste externo cego.
- LLM L0 é uma referência histórica que inclui fallback. Seus resultados não foram produzidos novamente nesta rodada.

O pacote histórico de outubro permanece em `../2026-10-16-final-comparison-subjects/`. Não misturar tabelas, previsões ou amostras das duas versões.

## Conteúdo e integridade

Cada `seed_*/` contém entradas, predições, métricas, escores, seleção, rótulos, manifesto e logs. Os checkpoints e dumps foram excluídos. `audit/` inclui proveniência, distribuição dos 43 códigos, coincidência de títulos, bootstrap, McNemar, Stuart–Maxwell, matriz por seção e amostra de 90 erros.

JSONs e logs receberam substituição de caminhos absolutos por caminhos portáveis; os CSVs do LLM conservam apenas colunas bibliográficas e de avaliação. Os hashes antes e depois dessas transformações estão no manifesto.

Na raiz `tcc-pipeline/`:

```bash
python tools/verify_artifact_manifest.py --artifact-dir artifacts/2026-11-22-audit-retraining
python artifacts/2026-11-22-audit-retraining/audit/conferir_nova_rodada.py \
  --root artifacts/2026-11-22-audit-retraining --output /tmp/conferencia-sementes.json
```

A conferência de escores usa NumPy. A auditoria de métricas/bootstrap usa Python padrão e, para Stuart–Maxwell, NumPy e SciPy. Os scripts de proveniência histórica exigem os CSVs intermediários locais que não estão integralmente neste pacote; essa verificação é distinta de recalcular métricas publicadas.

```bash
python artifacts/2026-11-22-audit-retraining/audit/auditar_resultados.py \
  --run-dir artifacts/2026-11-22-audit-retraining/seed_42 \
  --llm-run artifacts/2026-10-16-final-comparison-subjects \
  --llm-validation-file artifacts/2026-10-16-final-comparison-subjects/results/complementary_llm/exp_A_val_subjects.csv \
  --output-dir /tmp/resultados-recalculados
```

## Resultados centrais

Na semente 42, RL: 529/808 (65,47%; macro-F1 0,6493); F: 470/808 (58,17%; 0,5773); G: 484/808 (59,90%; 0,5931). Nas três sementes, F: 58,79% ± 0,62 ponto percentual; G: 60,60% ± 0,64. Os relatórios preservam os valores não arredondados.

O bootstrap tem 5.000 reamostragens por obra (791 grupos), semente 20261122 e intervalos percentis de 95%. O macro-F1 considera as classes presentes no gabarito; os subconjuntos da sensibilidade também informam 43 classes fixas. Os testes de McNemar e Stuart–Maxwell usam famílias separadas de dez pares, com correção de Holm.

A amostra atual contém 30 erros de RL, F e G (90 pares, 85 edições). As relações formais são automáticas e as interpretações temáticas permanecem sem confirmação humana. Não é amostra proporcional de todos os erros nem validação por bibliotecários.

## Reexecutar o treinamento

O script preservado exige macOS com MPS disponível, ambiente Python compatível e
`xlm-roberta-base` com seu tokenizer já no cache do Hugging Face. Ele opera em modo
offline e recusa sobrescrever uma pasta de semente existente. As versões efetivas
da rodada estão nos manifestos; o `requirements.txt` declara dependências sem
fixar um ambiente histórico completo. A preparação dos dumps não é necessária
para reusar as partições distribuídas.

Na raiz `tcc-pipeline/`, com o ambiente ativado:

```bash
python artifacts/2026-11-22-audit-retraining/audit/treinar_rodada_auditavel.py \
  --pipeline . \
  --site-packages "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
  --source-run artifacts/2026-11-22-audit-retraining/seed_42 \
  --output-root output/runs/nova-replicacao-auditavel \
  --seeds 42 43 44
```

Esse comando treina, seleciona na validação e avalia o teste para RL, kNN,
XLM-R-AA e ensemble. Não executa o LLM histórico. Uma reexecução em outro ambiente
pode produzir diferenças numéricas; as predições distribuídas são a referência
para conferir as tabelas do artigo.
