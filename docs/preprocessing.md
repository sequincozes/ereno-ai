# Preprocessador de features (E6)

## Objetivo

O épico E6 do plano de 60 dias pede um "preprocessador modular" com critério
de pronto "fit/transform sem leakage; manifest reproduzível". Antes do E6, o
preparo de features (colunas descartadas, imputação, encoding categórico)
vivia embutido e privado em `core/ids_evaluator.py::IdsEvaluator`, com um
leakage real: `train_baseline` ajustava essas decisões sobre o dataset
**inteiro** antes de fazer o `train_test_split` — a detecção de coluna
constante e o vocabulário categórico viam as linhas de teste.

O E6 extrai esse preparo para `core/preprocessor.py::FeaturePreprocessor`, um
componente independente com contrato `fit`/`transform`, reaproveitável por
qualquer detector sob o mesmo protocolo (pré-requisito do E8 — RF/DT/SVM). O
`IdsEvaluator` agora faz o `train_test_split` sobre o X **cru** e só então
chama `fit` — exclusivamente na partição de treino.

Undersampling e seleção de features por mutual information (o resto da janela
D25-35 do plano) são o **épico E7**, que consome `FeatureManifest.feature_columns`
como ponto de partida — não fazem parte deste componente.

## Contrato de saída

`domain/feature_manifest.py::FeatureManifest` (`extra="forbid"`,
`frozen=True`) é o retrato do que um `fit` produziu — reconstruível sozinho
via `FeaturePreprocessor.from_manifest`, sem pickle:

- `label_column`, `feature_columns` (ordem final ajustada);
- `dropped_columns`: cada coluna descartada e o motivo (`always_drop`,
  `constant` ou `cb_status`);
- `numeric_features` / `categorical_features`, partição exaustiva de
  `feature_columns`;
- `numeric_imputation` (`zero` ou `median`) + `numeric_fill_values`, um valor
  por feature numérica;
- `categorical_codes`, um dicionário `{valor: código}` por feature
  categórica;
- `scaler` (`none` ou `standard`) + `scaler_stats` (média/desvio por
  numérica, exigido para todas quando `scaler="standard"`);
- `fitted_rows`: quantas linhas de **treino** alimentaram o ajuste — nunca o
  total do dataset;
- `fitted_content_hash`: sha256 do conteúdo exato que foi ajustado (mesma
  convenção de `DatasetBundle.content_hash`).

Validadores (`@model_validator(mode="after")`) impõem: `numeric_features` e
`categorical_features` particionam `feature_columns` sem sobra nem
sobreposição; `dropped_columns` não repete nome e não colide com
`feature_columns`; `numeric_fill_values` e `categorical_codes` cobrem
**exatamente** as features do seu tipo (não um subconjunto); `scaler_stats`
cobre todas as `numeric_features` quando `scaler="standard"`; nenhum valor
ajustado (`numeric_fill_values`, `scaler_stats`) pode ser NaN/Infinity.

## Anti-leakage

`FeaturePreprocessor.fit` só enxerga o dataframe que recebe — o chamador é
responsável por já ter separado treino e teste antes de chamar `fit`.
`transform` nunca recalcula nada a partir dos dados que recebe; só aplica o
que `fit` já decidiu (categoria nunca vista → `unseen_category_code` (-1),
coluna ausente no dataframe de entrada → `missing_column_fill` (0.0), depois
imputada/escalada como qualquer outro valor). `FeatureManifest.fitted_rows` é
a prova, no próprio artefato, de que o ajuste viu só a partição de treino —
`tests/test_intent_loop_orchestrator.py` verifica esse número diretamente
(14 de 20 linhas do seed cacheado, `test_size=0.3` padrão).

Uma coluna pode se qualificar para mais de um motivo de descarte ao mesmo
tempo (no dataset real do ERENO, 21 das 33 colunas descartadas são
simultaneamente `always_drop` e `constant`) — o motivo registrado segue
sempre `always_drop` > `cb_status` > `constant`, nunca uma combinação.

Limitação conhecida, não é leakage mas é honesto documentar: o tipo
numérico-vs-categórico de cada coluna vem do dtype que `pandas.read_csv`
infere sobre o arquivo inteiro, antes de qualquer split — uma coluna não
pode trocar de tipo entre `fit` e `transform`, mas essa decisão específica
não é exclusiva da partição de treino.

## `PREPROCESSOR_MODE` — comparação antes/depois

`config.settings.PREPROCESSOR_MODE` (env var, default `"modular"`) e o
parâmetro homônimo de `IdsEvaluator.__init__` selecionam **quando** o `fit`
acontece em relação ao split — não uma lógica de preprocessamento diferente,
as duas variantes chamam o mesmo `FeaturePreprocessor`:

- `"modular"` (default): split sobre o X cru, `fit` só depois, na partição
  de treino. Sem leakage.
- `"legacy"`: `fit` sobre o dataset inteiro antes do split — reproduz de
  propósito o comportamento anterior ao E6, só para comparar métricas.

Nos dados reais do ERENO (`data/baseline_dataset.csv`) as duas variantes
produzem exatamente as mesmas métricas: todas as 11 colunas categóricas do
protocolo já são descartadas por `always_drop`, então não sobra vocabulário
categórico para vazar, e nenhuma das 20 features numéricas sobreviventes é
constante só no dataset inteiro mas não no treino — `git diff` de
`data/iteration_history.json` após regenerar o golden confirma isso: zero
diferença fora do caminho absoluto do dataset. A correção importa para
datasets onde essa coincidência não se sustenta.

## Falha do estágio

`FeaturePreprocessor.fit` levanta `PreprocessorError` (mensagem acionável)
quando o dataframe recebido está vazio, quando nenhuma feature sobrevive ao
descarte, ou quando uma categórica excede o teto de cardinalidade (1000
valores distintos — defensivo contra uma coluna de alta cardinalidade, como
um ID ou timestamp em texto, virar um manifest de megabytes). `transform`
levanta antes de qualquer `fit`. No pipeline intent-driven, qualquer uma
dessas falhas marca o `LoopStage` `detector` como `failed` com a causa —
mesmo mecanismo genérico de `IntentLoopOrchestrator._stage` que os demais
estágios usam.

`feature_manifest.json` é persistido **antes** do gate de matriz binária do
`DetectionReport` (`core/detection_reporter.py::build_detection_report`), não
depois — se a avaliação falhar ali, o manifest do que foi ajustado no
baseline já está em disco para diagnóstico.

## Fora de escopo e trabalho futuro

- **Undersampling e seleção de features (E7)**: consomem
  `FeatureManifest.feature_columns` como ponto de partida; não implementados
  aqui — ver `core/feature_selector.py`, `core/undersampler.py` e
  `docs/feature_selection.md`.
- **Detector interface RF/DT/SVM (E8)**: depende deste componente para que os
  três detectores comparem sob o mesmo preparo de dados; SVM é sensível a
  escala, por isso `scaler="standard"` já existe (desligado por default).
- **Mapeamento feature → técnica defensiva**: mencionado como extensão
  natural em `docs/feedback_policy.md`; não é este épico.

## Testes

- `tests/test_preprocessor.py`: fit/transform básico (imputação, encoding,
  ordem de colunas), anti-leakage (coluna constante só no treino, categoria
  não vista, `fitted_rows`, escala com estatísticas só do treino, salvaguarda
  de desvio zero), erros (`fit` vazio, sem features sobreviventes, teto de
  cardinalidade, `transform`/`manifest` antes de `fit`), determinismo do
  manifest e round-trip via `from_manifest`/JSON.
- `tests/test_new_contracts.py`: `FeatureManifest` versionado/congelado, e
  cada validador de consistência interna.
- `tests/test_intent_loop_orchestrator.py`: `feature_manifest.json`
  persistido no estágio DETECTOR, validado como `FeatureManifest`, com
  `fitted_rows` conferido contra o tamanho real da partição de treino.
- `tests/test_detection_reporter.py`, `tests/test_orchestrator_workflow.py`,
  `tests/test_feature_importances.py`: `IdsEvaluator` treinado de verdade
  (este último sobre o dataset real de ~80k linhas) continuam passando sem
  alteração de valores esperados.
