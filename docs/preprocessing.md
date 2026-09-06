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
qualquer detector sob o mesmo protocolo (consumido pelo E8 — RF/DT/SVM, ver
`docs/detectors.md`). O
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

## `PROTOCOL_FEATURES_MODE` — identidade não é o mesmo que semântica

A lista de sempre-descartar veio inteira do commit inicial do framework e
tratava dois grupos muito diferentes sob a mesma justificativa
("temporais/sequência, derivados fortes"). Medindo cada coluna descartada
contra a classe em `data/baseline_dataset.csv` — 21 das 33 são constantes e a
medição só faz sentido para as outras 12, com teto **H(classe) = 0.6611 nats**
— a separação é inequívoca:

| coluna | MI | % do teto | grupo |
|---|---|---|---|
| `sqDiff` | 0.5367 | 81% | delta |
| `tDiff` | 0.5335 | 81% | delta |
| `SqNum` | 0.5192 | 79% | delta |
| `timeFromLastChange` | 0.4343 | 66% | delta |
| `timestampDiff` | 0.2251 | 34% | delta |
| `stDiff` | 0.1958 | 30% | delta |
| `Time` | 0.0000 | 0% | identidade |
| `StNum` | 0.0002 | 0,03% | identidade |
| `t`, `GooseTimestamp`, `receivedTimestamp`, `delay` | ~0.0001 | ~0% | identidade |

Identificadores e valores absolutos não carregam sinal — e o que eles
carregariam num gerador sintético seria o bloco de geração, não o ataque.
Descartá-los está certo. Os deltas são o oposto: são a semântica de sequência
e temporização do GOOSE, e replay, flooding e grayhole **são** anomalias de
`sqDiff`/`tDiff`. Sem eles, o detector não tem como enxergar nada além da
falta forjada.

`config.settings.PROTOCOL_FEATURES_MODE` (env var) e o parâmetro homônimo de
`IdsEvaluator`/`FeaturePreprocessor` escolhem entre os dois regimes:

- `"drop"` (**default**): descarta os dois grupos, 33 colunas. Reproduz
  exatamente o comportamento herdado — `data/iteration_history.json` continua
  byte-estável. Sobram 20 features, 18 delas grandezas elétricas.
- `"deltas"`: descarta só as 24 de identidade. Sobram 29; as 9 a mais são
  `stDiff`, `sqDiff`, `SqNum`, `tDiff`, `timestampDiff`, `timeFromLastChange`,
  `gooseLengthDiff`, `apduSizeDiff`, `frameLengthDiff`.

`SqNum` bruto entra no grupo de deltas, e não no de identidade, porque em
GOOSE ele zera quando o `stNum` incrementa — carrega estado, não índice. É o
caso menos claro do grupo, e a assimetria com `StNum` (0,03% de MI) é
deliberada.

### Ablação

Sobre 8 000 linhas balanceadas do dataset de referência, Random Forest:

| modo | features | F1 | top-4 por importância |
|---|---|---|---|
| `drop` | 20 | 1.0000 | `vsbBTrapAreaSum` .168, `vsbCTrapAreaSum` .162, `vsbATrapAreaSum` .160, `cbStatus` .130 |
| `deltas` | 26 | 1.0000 | `SqNum` .253, `timeFromLastChange` .117, `sqDiff` .114, `tDiff` .102 |

A métrica não se move: `masquerade_fault` já era perfeitamente separável só
pela assinatura analógica, e é o único ataque fisicamente mensurável em
`--generator-mode cached`. **O que muda é em que o detector se apoia** — e
isso corta nos dois sentidos. `SqNum` virar a feature dominante para detectar
uma forma de onda forjada é plausível (o masquerade injeta quadros e perturba
a cadência), mas é também exatamente com o que um artefato do gerador se
pareceria. É por isso que o modo é medível por ablação em vez de estar ligado:
o ganho real está nos ataques que precisam do jar, e a decisão é do
experimento.

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
- **Detector interface RF/DT/SVM (E8)**: ~~fora de escopo~~ — **entregue**.
  `scaler="standard"` deixou de ser só uma preparação para o futuro: os dois
  SVMs registrados em `core/detectors.py` o pedem, e `IdsEvaluator` resolve a
  escala a partir do detector escolhido. Ver `docs/detectors.md`.
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
