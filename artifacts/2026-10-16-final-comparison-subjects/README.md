# Artefatos auditáveis da comparação final

Este pacote preserva dados bibliográficos, predições, métricas e tabelas de seleção da rodada `2026-10-16-final-comparison-subjects`. Permite conferir os resultados sem instalar PyTorch, baixar os dumps da Open Library ou treinar modelos. O código consultado está no repositório `https://github.com/evaldocunhaf/tcc-files`, commit-base `564ef1aaaf9ec72f15797459a29f4d3b8d6c160b`.

O repositório permanece privado por decisão do proprietário. A banca precisa receber autorização de acesso; a publicação desta branch não altera a visibilidade nem concede convites automaticamente. O permalink do commit deve ser informado no artigo.

Este é o pacote histórico de outubro, preservado para rastreabilidade. A [nova rodada auditável com três sementes](../2026-11-22-audit-retraining/README.md) está concluída e é distribuída separadamente. Não misturar suas previsões nem suas amostras de erro com as deste pacote.

## Dados e arquivos

- `prepared/train.csv`, `val.csv` e `test.csv`: os 4.034 registros com assuntos, separados em 2.423/803/808 registros. Todos os 43 códigos aparecem nos três conjuntos.
- `prepared/cohort_before_subject_filter.csv`: a coorte anterior, com 4.300 edições, antes de retirar os 266 registros sem assuntos. Ela não é um dump da Open Library nem todo o universo de coleta.
- `results/exp_{E,F,G,logreg}_{val,test}.csv` e JSONs `avaliacao_...`: kNN isolado, XLM-R com agregação por autoatenção, ensemble e regressão logística.
- `results/exp_A_test_subjects.csv`: resultado complementar LLM L0, que inclui o fallback previsto pelo sistema; não corresponde a um LLM puro em todos os registros.
- `results/complementary_llm/`: condições L0/L1 de validação preservadas da rodada anterior, `2026-10-15-openlibrary-subjects-43cdd`, e sua comparação pareada. Não devem ser confundidas com uma nova inferência desta publicação.
- `results/selecao_*.csv` e `selecao_final.json`: escolhas arquivadas de representação, número de vizinhos, C e peso do ensemble. Os caminhos locais do JSON foram convertidos em referências portáveis.
- `results/escores_G_{val,test}.npz`: rótulos e escores comprimidos dos dois componentes; não são checkpoints de modelos.
- `audit/`: script e resultados da distribuição, sobreposição de títulos e análise de sensibilidade. A documentação da proveniência da coleta também é incluída quando disponível na preparação.
- `manifest.json`: lista exata dos arquivos, tamanhos, SHA-256 publicados, SHA-256 das fontes e transformações aplicadas.

Os registros contêm metadados bibliográficos, não textos integrais das obras. A lista de publicação exclui dumps, checkpoints, cache de prompts e respostas brutas. Nos CSVs LLM são mantidas somente as colunas bibliográficas selecionadas, identificação, referência, previsão e método. Os JSONs foram reserializados e caminhos absolutos locais foram substituídos por caminhos portáveis. Essas transformações são registradas individualmente no manifesto.

## Verificação do pacote

Execute a partir de `tcc-pipeline/`, usando apenas a biblioteca padrão do Python 3:

```bash
python tools/verify_artifact_manifest.py \
  --artifact-dir artifacts/2026-10-16-final-comparison-subjects
```

O verificador confere todos os arquivos e rejeita arquivos adicionais fora do manifesto. O próprio `manifest.json` não contém hash de si mesmo; a integridade desse arquivo é fornecida pelo commit Git que distribui o pacote.

Para recalcular a auditoria exata de títulos e as métricas de sensibilidade sem modificar os dados ou resultados publicados:

```bash
python artifacts/2026-10-16-final-comparison-subjects/audit/auditar_titulos_distribuicao.py \
  --run-dir artifacts/2026-10-16-final-comparison-subjects \
  --output-dir /tmp/tcc-auditoria-recalculada \
  --skip-approximate
```

Retire `--skip-approximate` para também recalcular a busca exaustiva por títulos próximos, que demora mais. O diretório de saída deve ser novo ou destinado a essa conferência. A normalização, os limiares e as convenções de macro-F1 estão documentados no script e no relatório. Os hashes e caminhos do relatório recalculado podem diferir dos registros de origem por causa da exportação LLM e dos novos caminhos; as predições, referências e métricas não foram alteradas por essa redução de colunas.

## Convenções e resultados preservados

“Código completo” significa igualdade após normalização para no máximo cinco algarismos: três antes do ponto e até dois depois dele. Não significa igualdade da notação original com comprimento arbitrário. O macro-F1 é a média dos F1 por classe. A análise de sensibilidade apresenta tanto as classes presentes no subconjunto quanto o conjunto fixo de 43 classes.

No teste de 808 registros, as contagens preservadas são: regressão logística 529 acertos; ensemble 498; XLM-R 476; kNN isolado 372; LLM L0 76. As métricas da coorte integral são verificadas pelo script de auditoria contra os JSONs preservados. A retirada de títulos coincidentes é uma análise pós-hoc das mesmas predições. Ela altera a composição do teste e não prova identidade bibliográfica ou efeito causal de vazamento.

## Limitação conhecida da seleção do ensemble

O peso aplicado no teste é 0,45 para o componente neural e 0,55 para o kNN. Os parâmetros registrados do componente lexical são `char_wb`, n-gramas de 3 a 5 caracteres, k=9 e similaridade cosseno. A regressão logística usa TF-IDF de palavras (unigramas/bigramas) e C=10.

Os escores neurais de validação arquivados para o ensemble produzem 315/803 acertos (39,23%), enquanto `exp_F_val.csv`, preservado para o método neural isolado, registra 480/803 (59,78%). Identificadores, referências e rótulos estão alinhados; 324/803 predições coincidem. A combinação arquivada de validação é coerente com seus próprios escores, mas não comprova seleção do peso com as saídas do mesmo checkpoint posteriormente usado no teste. A origem dessa divergência não foi estabelecida.

No teste, os arrays reproduzem todas as 808 previsões neurais e todas as 808 do ensemble. Assim, a mistura e os resultados de teste podem ser conferidos, mas essa conferência não elimina a limitação na seleção do peso. Este pacote não refaz inferência, treinamento ou escolha de parâmetros e não corrige números por suposição.

## Reexecução dos modelos e ambiente

O pacote não contém checkpoint neural e não permite reconstruir exatamente o treinamento de outubro. O log estava vazio, os argumentos efetivos do treinamento não foram recuperados e as versões históricas das dependências não foram fixadas. A lista atual em `requirements.txt` declara dependências do código disponível, incluindo PyTorch e Transformers; não é um arquivo de ambiente histórico.

## Reconstrução da exportação

Com acesso aos arquivos locais de origem, execute o construtor de lista permitida:

```bash
python tools/package_final_artifacts.py \
  --source-pipeline /caminho/para/tcc-pipeline \
  --audit-dir /caminho/para/auditoria-2026-11-22 \
  --destination artifacts/2026-10-16-final-comparison-subjects
```

O construtor não altera as fontes, não acessa a rede e não inclui arquivos fora da lista explícita. O campo `source_code_commit` identifica o código-base consultado; o permalink do commit que contém a publicação deve ser usado para referenciar esta exportação e seus scripts.
