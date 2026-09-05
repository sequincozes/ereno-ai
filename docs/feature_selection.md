# Undersampling + seleção de features (E7)

## Objetivo

O épico E7 do plano de 60 dias é o resto da janela D25-35 que o E6 deixou
explicitamente de fora: "undersampling e seleção de features por mutual
information", com critério de pronto "estratégias configuráveis com
ablação". Os dois componentes consomem `FeatureManifest.feature_columns`
(E6) como ponto de partida — não fazem parte do `FeaturePreprocessor`, que já
documentava essa fronteira desde o E6.

- `core/feature_selector.py::FeatureSelector` — seleção de features por
  mutual information (`sklearn.feature_selection.mutual_info_classif`).
- `core/undersampler.py::RandomUndersampler` — balanceamento de classes da
  partição de treino por reamostragem aleatória sem reposição.
- `domain/selection_manifest.py::SelectionManifest` — contrato congelado que
  registra o que as duas estratégias decidiram, persistido como
  `selection_manifest.json` no pipeline intent-driven (mesmo estágio
  DETECTOR que já persiste `feature_manifest.json`).

Ambas as estratégias são `"none"` por default
(`config.settings.FEATURE_SELECTION_MODE`/`UNDERSAMPLING_MODE`) —
comportamento idêntico a antes do E7 existir, até alguém optar
explicitamente por uma delas.

## Ordem do pipeline

`IdsEvaluator.train_baseline` (`core/ids_evaluator.py`), depois que o
`FeaturePreprocessor` (E6) já fez `fit_transform`/`transform` no treino/teste:

1. **Seleção de features primeiro**, ajustada sobre **todo** o treino
   (`FeatureSelector.fit_transform(X_train, y_train)`, aplicado também em
   `X_test` via `transform`) — mutual information é mais estável com mais
   linhas, e a classe minoritária de ataque não deve perder peso na
   pontuação de relevância antes mesmo de decidir quais features sobrevivem.
2. **Undersampling depois**, só no espaço de features já reduzido
   (`RandomUndersampler.fit_resample(X_train, y_train)`) — nunca em
   `X_test`/`y_test`, nem em nenhum dataset passado a `evaluate_variant`:
   reamostrar o conjunto de avaliação distorceria a própria métrica que se
   quer medir (a distribuição real de ataque vs. normal é o que o detector
   precisa enfrentar em produção).

O modelo só é treinado depois dos dois passos; `evaluate_variant` e
`get_shap_importances` aplicam `preprocessor.transform` seguido de
`feature_selector.transform` (nunca undersampling) — o mesmo espaço de
colunas que o modelo aprendeu.

## `FeatureSelector` — mutual information

- `strategy="none"` (default): mantém todas as features candidatas, na
  ordem que o preprocessador produziu. `scores` fica `{}` — a estratégia
  "none" não calcula pontuação nenhuma.
- `strategy="mutual_info"`: exige `top_k` e/ou `min_score` (sem critério de
  corte não há como decidir quantas features manter — levanta
  `FeatureSelectorError` na construção). Ajusta
  `mutual_info_classif(X_train, y_train, random_state=...)`, ordena por
  pontuação decrescente com desempate determinístico por nome de coluna, e
  aplica `top_k` e/ou `min_score` (interseção, quando os dois são
  informados). `selected_features` preserva a ordem original das colunas
  candidatas, não a ordem de ranking — mantém o layout estável para o
  modelo treinado. Levanta `FeatureSelectorError` se nada sobreviver ao
  corte.

## `RandomUndersampler` — reamostragem aleatória

- `strategy="none"` (default): devolve `X`/`y` inalterados.
- `strategy="random"`: para cada classe acima do tamanho da classe
  minoritária, sub-amostra sem reposição até igualar a minoritária,
  determinístico por `random_state` (mesmo `random_state=42` do
  `IdsEvaluator`, reaproveitado — não é um parâmetro novo e independente).
  Não usa `imbalanced-learn`: a estratégia é simples o bastante para
  `numpy`/`pandas` puros, evitando puxar mais uma dependência para um MVP
  que já tem 11 ataques + scikit-learn.

## Contrato de saída — `SelectionManifest`

`domain/selection_manifest.py::SelectionManifest` (`extra="forbid"`,
`frozen=True`), irmão de `FeatureManifest` e fora dele:

- `feature_selection`/`feature_selection_top_k`/`feature_selection_min_score`
  — a estratégia e o(s) critério(s) de corte usados;
- `candidate_features`/`selected_features` — antes/depois da seleção;
- `feature_scores` — só preenchido quando `feature_selection="mutual_info"`,
  cobre exatamente `candidate_features`;
- `undersampling`/`undersampling_random_state` — a estratégia e a seed;
- `class_counts_before`/`class_counts_after` — contagem por rótulo
  **original** (não o código do `LabelEncoder`), antes/depois do
  undersampling;
- `fitted_rows_before_undersampling`/`fitted_rows_after_undersampling` —
  linhas de **treino**, nunca teste nem total do dataset (mesma prova de
  anti-leakage de `FeatureManifest.fitted_rows`).

Validadores impõem: `selected_features` é subconjunto de
`candidate_features`; `feature_scores` vazio quando `feature_selection="none"`
e completo quando `"mutual_info"`; `mutual_info` sem `top_k` nem `min_score`
é rejeitado (o manifest não pode fingir que houve corte sem dizer qual);
linhas depois do undersampling nunca aumentam, e `undersampling="none"` não
pode mudar a contagem; a soma de `class_counts_before`/`class_counts_after`
bate com `fitted_rows_before_undersampling`/`fitted_rows_after_undersampling`
quando as contagens são informadas.

## Persistência no pipeline intent-driven

`agents/orchestrator/intent_loop.py::IntentLoopOrchestrator._run_detector`
persiste `selection_manifest.json` no estágio DETECTOR, logo depois de
`feature_manifest.json` — mesmo motivo de ordem: se a avaliação falhar no
gate de matriz binária do `DetectionReport` logo em seguida, o que a
seleção/undersampling decidiu sobre o treino já está em disco para
diagnóstico. Não é um `LoopStage` próprio, é artefato do DETECTOR, igual ao
manifest do E6.

## Ablação e reprodutibilidade

Critério de pronto do E7: "estratégias configuráveis com ablação".
`tests/test_ids_evaluator_selection.py` cobre isso na integração real com o
`IdsEvaluator`:

- por default (`"none"`/`"none"`), `SelectionManifest.selected_features ==
  candidate_features` e `fitted_rows_before_undersampling ==
  fitted_rows_after_undersampling` — comportamento idêntico a antes do E7;
- `feature_selection="mutual_info"` com `top_k=1` derruba as features de
  ruído do modelo (`evaluator.feature_columns` encolhe) e reproduz a mesma
  seleção com a mesma seed;
- `undersampling="random"` balanceia as classes de treino
  (`class_counts_after` com a mesma contagem nas duas classes) sem nunca
  mudar `dataset_rows` da avaliação (a prova de que o teste/variante não foi
  tocado) e reproduz a mesma contagem final com a mesma seed;
- as duas estratégias compõem: seleção reduz colunas, undersampling reduz
  linhas do que sobrou, na ordem descrita acima.

`tests/test_feature_selector.py` e `tests/test_undersampler.py` cobrem os
dois componentes isolados (contrato fit/transform, erros, determinismo,
preservação de ordem de colunas); `tests/test_new_contracts.py` cobre cada
validador de consistência interna do `SelectionManifest`;
`tests/test_intent_loop_orchestrator.py` confere que
`selection_manifest.json` é persistido e válido no loop real.

## Fora de escopo e trabalho futuro

- **Detector interface RF/DT/SVM (E8)**: depende deste componente e do E6
  para que os três detectores comparem sob o mesmo preparo/seleção de
  dados.
- **Estratégias de undersampling mais ricas** (ex.: NearMiss, Tomek links) e
  **outros critérios de seleção** (ex.: RFE, importância do próprio modelo)
  não são deste épico — `strategy` nos dois componentes já é um `Literal`
  extensível para isso, sem quebrar o contrato existente.
