# Detector plugável RF/DT/SVM (E8)

## Objetivo

O épico E8 é a janela D36-46 do plano de 60 dias: "mesmo split/protocolo para
todos os detectores", com critério de pronto "RF/DT/SVM comparáveis". É o
épico que o E6 e o E7 vinham anunciando como consumidor — os dois já tinham
tornado o preparo de dados detector-agnóstico; o que faltava era o detector
deixar de ser um `RandomForestClassifier` embutido em `core/ids_evaluator.py`.

- `core/detectors.py::Detector` — adaptador fit/predict sobre um estimador do
  scikit-learn registrado, resolvido por chave.
- `domain/detector_manifest.py::DetectorManifest` — contrato congelado que
  registra qual detector treinou e sob que preparo de dados, persistido como
  `detector_manifest.json` no pipeline intent-driven (mesmo estágio DETECTOR
  que já persiste `feature_manifest.json` e `selection_manifest.json`).
- `config.settings.DETECTOR_MODE` / `DETECTOR_SCALER` — os dois knobs de
  ambiente. O default é `random_forest` sem escala: **comportamento idêntico
  ao de antes do E8**, e o histórico golden segue byte-estável.

O que o E8 **não** muda: nenhum split, nenhuma métrica, nenhum preparo de
feature. Toda a comparabilidade entre detectores vem de eles serem alimentados
pelo mesmo pipeline a montante (E6 → E7), não de nada feito dentro do
detector.

## Detectores registrados

| chave | estimador | escala pedida | importância | SHAP | custo no volume do ERENO |
|---|---|---|---|---|---|
| `random_forest` (default) | `RandomForestClassifier` | `none` | Gini | sim | baixo (`n_jobs=-1`) |
| `decision_tree` | `DecisionTreeClassifier` | `none` | Gini | sim | muito baixo |
| `svm_linear` | `LinearSVC` (liblinear) | `standard` | `\|coef_\|` | não | médio, escala bem |
| `svm_rbf` | `SVC(kernel="rbf")` | `standard` | **nenhuma** | não | **O(n²)–O(n³) em linhas de treino** |

### Por que quatro chaves e não uma chave `svm` com parâmetro de kernel

O kernel muda tanto o custo quanto a capacidade de explicação do detector, e
esses dois fatos precisam estar visíveis no nome do que se está rodando. Uma
chave `svm` que silenciosamente fosse linear (rápida, explicável) ou RBF
(lenta, opaca) faria dois experimentos incomparáveis parecerem o mesmo.

### Desvios dos defaults do scikit-learn

Os hiperparâmetros do registro são os defaults do próprio sklearn, com duas
exceções deliberadas (ambas comentadas em `core/detectors.py`):

- `svm_linear`: `max_iter=5000` em vez de 1000 — com dezenas de milhares de
  linhas o liblinear frequentemente não converge no default, e um modelo
  mal-convergido produz uma métrica que mede o otimizador, não o detector.
- `svm_rbf`: `cache_size=500` em vez de 200 MB — o gargalo do libsvm nesse
  volume é o cache do kernel. Não muda o modelo, só o tempo.

O `random_forest` reproduz exatamente a construção embutida anterior ao E8
(`n_estimators=100, random_state=42, n_jobs=-1`); os demais defaults do
registro são valores default do sklearn, então passá-los explicitamente não
altera o modelo resultante.

## Escala (E6 × E8)

`scaler="standard"` existe em `FeaturePreprocessor` desde o E6, desligado por
default, precisamente para este épico — árvores são invariantes a
transformação monotônica de feature, SVMs não são. O E8 fecha o circuito:

- `IdsEvaluator(scaler=None)` (default) **delega ao detector**: `standard`
  para os dois SVMs, `none` para as árvores.
- `IdsEvaluator(scaler="none"|"standard")` **força** os dois lados a usarem a
  mesma escala — que é o que uma ablação controlada entre detectores precisa.

Rodar um SVM sem escala é legítimo (é uma ablação válida) e por isso não é
erro. Mas fica registrado: `DetectorManifest` carrega `requires_scaling` (a
recomendação) e `resolved_scaler` (o que foi aplicado) lado a lado, então um
SVM que rodou sem escala explica sozinho uma métrica ruim, sem precisar
reexecutar nada.

## Limite da garantia de "mesmo protocolo"

A comparabilidade do E8 é literal enquanto `FEATURE_SELECTION_MODE="none"`
(o default): os quatro detectores recebem o mesmo split, as mesmas colunas
candidatas e as mesmas linhas de treino, e a única diferença é a escala — que
árvores ignoram por construção.

Com `FEATURE_SELECTION_MODE=mutual_info` a garantia fica **condicional**. A
seleção do E7 é ajustada sobre o `X` já preprocessado, ou seja **já escalado**,
e `mutual_info_classif` não é estritamente invariante a escala (o ruído que ele
injeta é proporcional à amplitude de cada coluna, e o estimador é baseado em
distâncias de k-vizinhos). Em princípio, portanto, uma árvore (`scaler="none"`)
e um SVM (`scaler="standard"`) podem receber conjuntos de features diferentes
enquanto os dois manifests dizem ter rodado "sob o mesmo protocolo".

Na prática isso não se manifestou no baseline do ERENO — medi RF e `svm_linear`
com `top_k=5` e a seleção saiu idêntica. Mas não é uma garantia, e sim uma
coincidência dos dados. **Para uma comparação controlada com seleção ligada,
force `scaler=` no mesmo valor dos dois lados**:

```bash
export DETECTOR_SCALER=standard FEATURE_SELECTION_MODE=mutual_info FEATURE_SELECTION_TOP_K=15
uv run adversarial-ids --engine live --detector svm_linear
uv run adversarial-ids --engine live --detector random_forest
```

`tests/test_ids_evaluator_detectors.py::test_forcing_the_same_scaler_pins_the_selected_features_across_detectors`
fixa esse remédio como garantia.

## Importâncias: mesmo formato, significados diferentes

`DetectionReport.top_features` mantém o formato congelado da issue #15
(`{"feature": str, "importance": float}`) para os quatro detectores, mas o
número **não é comparável entre eles**:

- **Gini** (RF/DT): soma 1 sobre as features.
- **`|coef_|`** (SVM linear): está na unidade do espaço já padronizado — só é
  interpretável *porque* o detector pediu `scaler="standard"`. Sem escala, a
  feature de maior amplitude receberia o menor coeficiente e o ranking diria
  o contrário do que a fronteira de decisão realmente usa.
- **Nenhuma** (SVM RBF): `top_features` vem vazio por construção.

`DetectorManifest.importance_kind` é o campo que diz qual dos três está em
jogo — comparar ranking de features entre detectores só é honesto depois de
olhar esse campo.

Consequência para o Blue Team (E5): com `svm_rbf`, o `DefensePlan` continua
podendo citar todas as métricas escalares e a matriz de confusão do relatório
(ver `agents/defender/tools.py::select_evidence_candidates`), mas perde as
evidências por feature. É menos evidência, não uma falha — o guardrail
continua valendo, e nenhuma evidência inventada passa.

SHAP segue a mesma disciplina: `shap.TreeExplainer` só sabe ler modelo baseado
em árvore, então `get_shap_importances` devolve `[]` para os dois SVMs pelo
mesmo caminho best-effort de quando o pacote `shap` está ausente — nunca um
explicador errado aplicado a um modelo que ele não sabe ler.

## Os três manifests do estágio DETECTOR

O E8 fecha a cadeia que o E6 e o E7 abriram. Cada execução do pipeline
intent-driven grava os três lado a lado em `outputs/intent_loop/<run_id>/`:

| artefato | épico | responde |
|---|---|---|
| `feature_manifest.json` | E6 | como as features foram preparadas |
| `selection_manifest.json` | E7 | quais sobreviveram, sobre quantas linhas |
| `detector_manifest.json` | E8 | qual detector aprendeu sobre esse espaço |

É essa tripla que torna a exigência do D36-46 auditável a partir dos artefatos
em disco: dois relatórios com `model_name` diferente só provam que rodaram sob
o mesmo protocolo se os manifests de preparo baterem. Nenhum dos três é um
`LoopStage` próprio — são artefatos do DETECTOR, persistidos **antes** do gate
de matriz binária do `DetectionReport`, para que uma avaliação que falhe ali
ainda deixe em disco o que foi ajustado e treinado.

`DetectorManifest.trained_rows`/`trained_features` seguem a mesma disciplina
anti-leakage de `FeatureManifest.fitted_rows`: contam o que o `fit` viu — a
partição de treino **depois** da seleção de features e do undersampling —
nunca o dataset inteiro nem a partição de teste.

## Como escolher o detector

```bash
# CLI — vale para --engine live e --engine intent
uv run adversarial-ids --engine intent --prompt "..." --detector decision_tree
uv run adversarial-ids --engine live --attack random_replay --detector svm_linear

# Ambiente (afeta todo caminho que herda o default de settings)
DETECTOR_MODE=svm_rbf uv run adversarial-ids --engine live ...

# Ablação de escala controlada: força os dois lados na mesma escala
DETECTOR_MODE=svm_linear DETECTOR_SCALER=none uv run ...
```

```python
# Programaticamente
from adversarial_ids.core.ids_evaluator import IdsEvaluator

evaluator = IdsEvaluator(detector="svm_linear")            # escala resolvida sozinha
evaluator = IdsEvaluator(detector="svm_rbf", scaler="none")  # ablação explícita
evaluator = IdsEvaluator(
    detector="decision_tree", detector_hyperparameters={"max_depth": 5}
)
```

`--engine demo` **rejeita** `--detector`: o histórico golden é um replay já
gravado, nenhum detector é treinado ali, e aceitar a escolha em silêncio seria
mentir sobre o que a execução faz.

`n_estimators` continua na assinatura do `IdsEvaluator` (é anterior ao E8),
mas só se aplica ao `random_forest` — é um hiperparâmetro dele. Para os
demais, use `detector_hyperparameters`; um hiperparâmetro que o detector não
suporta é rejeitado na construção com `DetectorError`, mesma disciplina do
`extra="forbid"` dos contratos do `domain/`.

## Reprodutibilidade e a fixture golden

`DETECTOR_MODE` e `DETECTOR_SCALER` são env vars que mudam o modelo treinado —
o mesmo risco latente que `PREPROCESSOR_MODE`/`FEATURE_SELECTION_MODE` já
tinham. Por isso `scripts/generate_golden_history.py` **fixa** o detector e a
escala explicitamente (`detector="random_forest", scaler="none"`) em vez de
herdar o default: sem isso, um `DETECTOR_MODE` no shell de quem regenera
reescreveria `data/iteration_history.json` com outro modelo.

Todos os quatro detectores recebem `random_state` (42 por default) e produzem
métricas idênticas entre execuções com a mesma seed — coberto em
`tests/test_detectors.py` e `tests/test_ids_evaluator_detectors.py`.

Uma ressalva sobre o próprio `detector_manifest.json`: ele carrega
`train_duration_seconds`, que é relógio de parede e portanto **difere entre
execuções idênticas**. Isso é intencional (é um dado de custo, e custo é
justamente o que distingue `svm_rbf` dos demais) e inofensivo porque o arquivo
vive em `outputs/`, nunca em `data/` — nenhuma fixture versionada nem
comparação golden lê esse arquivo. Se algum dia um teste precisar comparar dois
run dirs byte a byte, este é o campo a excluir.

## Fora de escopo e trabalho futuro

- **Seleção automática de detector**: nada aqui escolhe o melhor detector
  sozinho, nem roda os quatro numa varredura. Cada execução treina um. Uma
  campanha comparativa é hoje N execuções com `--detector` diferente, cujos
  manifests provam que rodaram sob o mesmo protocolo.
- **Balanceamento por `class_weight`**: exposto como hiperparâmetro
  sobrescrevível nos quatro registros, mas `None` por default — mexer no
  default mudaria o caminho `random_forest`, que precisa continuar idêntico
  ao de antes do E8. A alavanca de desbalanço deste repositório é o
  undersampling do E7 (ver `docs/feature_selection.md`).
- **Detectores fora do scikit-learn** (XGBoost, uma rede rasa): `DetectorSpec`
  já é um registro extensível e `DetectorLike` é o protocolo que o
  `IdsEvaluator` de fato consome, mas cada nova chave exige uma entrada no
  `Literal` de `domain/detector_manifest.py::DetectorKey` — de propósito: o
  vocabulário de detectores é contrato, não configuração livre.
- **Explicação para modelos não-árvore**: `KernelExplainer`/`LinearExplainer`
  do SHAP cobririam os SVMs, mas o primeiro custa ordens de grandeza mais que
  o `TreeExplainer` já usado. Fora deste épico; hoje o SVM RBF simplesmente
  não oferece ranking de feature.

## Testes

- `tests/test_detectors.py` — registro (chaves, defaults, hiperparâmetro não
  suportado), fit/predict dos quatro pela mesma interface, erros
  (`predict`/`manifest` antes de `fit`, entrada vazia, tamanhos diferentes),
  importâncias por tipo (Gini/coef/nenhuma, truncagem por `top_n`, espaço de
  features desalinhado), determinismo por seed e round-trip JSON do manifest.
- `tests/test_ids_evaluator_detectors.py` — integração com o pipeline E6/E7:
  os quatro treinam e avaliam variante; todos compartilham o mesmo split e o
  mesmo espaço de features; a escala recomendada chega ao
  `FeaturePreprocessor` e um `scaler` explícito a sobrescreve; o manifest bate
  com `SelectionManifest.fitted_rows_after_undersampling`; SHAP não é tentado
  em não-árvore.
- `tests/test_new_contracts.py` — `DetectorManifest` versionado/congelado e
  cada validador de consistência interna (`importance_kind` × detector, SHAP
  só em árvore, chave não registrada, espaço de features vazio).
- `tests/test_intent_loop_orchestrator.py` — `detector_manifest.json`
  persistido e válido no loop real; a escolha do detector chega ao
  `DetectionReport.model_name`, ao manifest e ao Defensor; um SVM resolve
  `scaler="standard"` até o `FeatureManifest`; chave inválida estoura na
  construção do orquestrador, não como falha de estágio.
- `tests/test_live_wiring.py` / `tests/test_cli_intent_engine.py` — o
  `--detector` chega ao `run_live_workflow` e ao `run_intent_loop`, e o engine
  `demo` rejeita a escolha.
