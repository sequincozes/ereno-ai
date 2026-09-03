# Catálogo de capacidades intent-driven

Referência do que cada ataque expõe ao pipeline intent-driven
(`--engine intent`) — a allowlist entre uma `IntentSpec` e os campos que
`core/intent_compiler.py` pode alterar. Fonte de verdade:
`config/attack_capabilities.py` + `config/capabilities/*.py`. Este documento
é gerado a partir do catálogo real (ver comando no fim) e não deve divergir
dele.

## Convenções

- **`capability_id`**: `"<attack_key>.v1"` — uma por ataque registrado em
  `config/attacks_registry.py`, inclusive as 4 variantes `delayed_replay*`
  (mesmo `delayed_replay_double_drop` sendo campo-a-campo idêntico ao
  `delayed_replay` base: compartilhar um id faria `AttackCapability.attack_key`
  mentir para três das quatro specs).
- **`polarity`**: `"direct"` (padrão) — valor maior = ataque mais intenso.
  `"inverse"` — valor maior = ataque **menos** intenso (gaps, intervalos entre
  rajadas). O compilador inverte a direção efetiva desses campos, para que
  "evadir detecção" nunca signifique apertar um gap.
- **Efeitos**: `lower_f1`, `lower_recall`, `mimic_normal_traffic` (as três
  "de evasão" — sempre com o mesmo conjunto de campos numa capacidade),
  `increase_attack_activity`. `increase_resource_pressure` existe no contrato
  mas **nenhum catálogo o anuncia** — ver seção própria abaixo.
- **`minimum`/`maximum`**: o piso/teto que o compilador respeita. É a única
  defesa contra compilar o ataque para fora de existência — o pipeline
  intent-driven não roda `shared/validator.py` (isso é só do loop legado).

## Ataques e campos

### `masquerade_fault` (`masquerade_fault.v1`) — 12 campos, cobertura total

| campo | tipo | pol. | bounds/choices | efeitos |
|---|---|---|---|---|
| `fault.prob` | number | direct | [0.0, 1.0] | evasão + atividade |
| `fault.durationMs.min` | integer | direct | ≥0 | evasão |
| `fault.durationMs.max` | integer | direct | ≥0 | evasão |
| `cbStatus` | integer | direct | {0, 1} | evasão |
| `incrementStNumOnFault` | boolean | direct | {False, True} | evasão |
| `sqnumMode` | string | direct | {fast, normal} | evasão |
| `ttlMsValues` | integer_list | direct | — | evasão |
| `analog.deltaAbs.min` | number | direct | ≥0.0 | evasão |
| `analog.deltaAbs.max` | number | direct | ≥0.0 | evasão |
| `trapArea.multiplier.min` | number | direct | ≥0.0 | evasão |
| `trapArea.multiplier.max` | number | direct | ≥0.0 | evasão |
| `trapArea.spikeProb` | number | direct | [0.0, 1.0] | evasão + atividade |

### `random_replay` (`random_replay.v1`) — 15 campos, cobertura total

| campo | tipo | pol. | bounds | efeitos |
|---|---|---|---|---|
| `count.lambda` | integer | direct | [50, 5000] | evasão + atividade |
| `windowS.{min,max}` | number | direct | [0.1, 30.0] | evasão |
| `delayMs.{min,max}` | number | direct | [1.0, 2000.0] | evasão |
| `burst.prob` | number | direct | [0.0, 1.0] | evasão + atividade |
| `burst.{min,max}` | integer | direct | [1, 50] | evasão + atividade |
| `burst.gapMs.{min,max}` | number | **inverse** | [0.05, 1000.0] | evasão + atividade |
| `reorderProb` | number | direct | [0.0, 1.0] | evasão |
| `ttlOverride.valuesMs` | integer_list | direct | — | evasão |
| `ttlOverride.prob` | number | direct | [0.0, 1.0] | evasão + atividade |
| `ethSpoof.srcProb` | number | direct | [0.0, 1.0] | evasão |
| `ethSpoof.dstProb` | number | direct | [0.0, 1.0] | evasão |

> **Incerteza registrada**: `delayMs.{min,max}` está tagueado `direct` (menos
> atraso ≈ temporização mais próxima do normal), mas um replay quase sem
> atraso também é uma duplicata mais óbvia — plausivelmente `inverse`.
> Resolver com um experimento A/B (`--generator-mode jar`, comparar recall)
> em vez de garantir.

### `inverse_replay` (`inverse_replay.v1`) — 12 campos, cobertura total

Mesma família estrutural do `random_replay`. `blockLen` substitui `windowS`;
`delayMs` é **inteiro** aqui (vs. float no uc01); sem `ethSpoof`/`reorderProb`.
`burst.gapMs.{min,max}` também é `inverse`, pelo mesmo motivo.

### `injection` (`injection.v1`) — 2 de 12 campos

| campo | tipo | pol. | bounds/choices | efeitos |
|---|---|---|---|---|
| `numInjectedMessages` | integer | direct | [5, 5000] | evasão + atividade |
| `injectionPattern` | string | — | {random, uniform} | evasão |

O piso 5 de `numInjectedMessages` existe porque o gate E4
(`DatasetBundle`) exige `min_attack_rows=5` por padrão — um valor menor
produziria um candidato que o próprio pipeline rejeitaria no PREPROCESS.

### `high_stnum` (`high_stnum.v1`) — 9 de 11 campos

| campo | tipo | pol. | bounds/choices | efeitos |
|---|---|---|---|---|
| `count.lambda` | integer | direct | [20, 4000] | evasão + atividade |
| `jump.{min,max}` | integer | direct | [1, 1000] | evasão |
| `sqnumResetProb` | number | direct | [0.0, 1.0] | evasão |
| `sqNumDelta.max` | integer | direct | [0, 100] | evasão |
| `randomizeTimestamp` | boolean | direct | {False, True} | evasão |
| `ttlOverride.valuesMs` | integer_list | direct | — | evasão |
| `ttlOverride.prob` | number | direct | [0.0, 1.0] | evasão + atividade |
| `padBytes.max` | integer | direct | [0, 256] | evasão |

### `flooding` (`flooding.v1`) — 8 campos, cobertura total

| campo | tipo | pol. | bounds/choices | efeitos |
|---|---|---|---|---|
| `burst.{min,max}` | integer | direct | [1, 2000] | evasão + atividade |
| `gapMs.{min,max}` | number | **inverse** | [0.01, 10.0] | evasão + atividade |
| `stnumEveryPacket` | boolean | direct | {False, True} | evasão |
| `sqnumStrideValues` | integer_list | direct | — | **só atividade** |
| `ttlMs` | integer | direct | [1, 1000] | evasão |
| `ethSrcSpoofProb` | number | direct | [0.0, 1.0] | evasão |

`gapMs` é a alavanca principal do ataque — gap **menor** entre pacotes é o
que caracteriza flooding, daí `inverse`. `sqnumStrideValues` só carrega
`increase_attack_activity`: escalar a lista elementwise por um fator &lt;1
(evasão) produz um multiset degenerado (`[1,2,4]` × 0.5 → `[1,1,2]`, duplicando
o stride 1), então a direção de evasão não é segura para esse campo.

### `grayhole` (`grayhole.v1`) — 11 de 12 campos

| campo | tipo | pol. | bounds | efeitos |
|---|---|---|---|---|
| `dropRate.{min,max}` | number | direct | [0.05, 1.0] | evasão + atividade |
| `burstDropProb` | number | direct | [0.0, 1.0] | evasão + atividade |
| `burstDropLen.{min,max}` | integer | direct | [1, 100] | evasão + atividade |
| `extraDelayMs.{min,max}` | number | direct | [1.0, 2000.0] | evasão |
| `delayBurstProb` | number | direct | [0.0, 1.0] | evasão |
| `delayBurstLen.{min,max}` | integer | direct | [1, 100] | evasão |
| `statusChangeDropProb` | number | direct | [0.0, 1.0] | evasão + atividade |

`dropRate.{min,max}` é semanticamente uma probabilidade cujas folhas se
chamam `min`/`max` — o clamp genérico legado (que casa por substring `"prob"`
na chave) não a alcança; aqui os limites são explícitos, com piso `0.05` (não
`0.0`) para não deixar o compilador reduzir a taxa de descarte a um ponto
onde o ataque deixa de existir fisicamente.

### `delayed_replay` / `_backoff` / `_batch_dump` / `_double_drop` (uc10)

Campos comuns às 4 variantes:

| campo | tipo | pol. | bounds | efeitos |
|---|---|---|---|---|
| `burstInterval.{min,max}` | integer | **inverse** | [10, 5000] | evasão + atividade |
| `burstMax` | integer | direct | [1, 2000] | evasão + atividade |
| `selectionProb.value` | number | direct | [0.05, 1.0] | evasão + atividade |
| `networkDelayMs.{min,max}` | number | direct | [1.0, 5000.0] | evasão |
| `shiftSendTimestamp` | boolean | direct | {False, True} | evasão |
| `replaceWithFake` | boolean | direct | {False, True} | evasão |

Extras por variante: `_backoff` adiciona `rateMultiplier` (direct,
`[1.0, 10.0]` — `1.0` é "backoff desligado", o piso semântico correto, não
`0.0`); `_batch_dump` adiciona `microGapMs` (**inverse**, `[0.1, 100.0]`).
`_double_drop` não adiciona nada — a variante é dispatch-only, diferenciada
só pelo `attackType`.

`selectionProb.value` é uma probabilidade cuja chave-folha é `value` — o
clamp genérico legado não a alcança.

## Campos excluídos e por quê

| campo | ataque(s) | motivo |
|---|---|---|
| `orderBy` | `delayed_replay*` (4) | string sem enum documentado em nenhum lugar do repo. `FieldCapability._string_fields_need_choices` rejeitaria a declaração sem `choices` na hora do import. Recuperável via arqueologia no bytecode do JAR (`generator_runtime/ereno-generator.jar`), como foi feito para `sqnumMode` — decompilar a classe criadora do uc10 e achar o `switch`/`equalsIgnoreCase` sobre a string. |
| `randomSeed` | `injection` | controle de reprodutibilidade, não uma alavanca de intensidade. |
| `stNum`, `sqNum`, `cbStatus.values`, `ttlMs`, `confRev` (9 caminhos) | `injection` | só têm efeito físico quando `injectionPattern == "synthetic"`; o baseline usa `"random"` (clona mensagens legítimas, ignora essas faixas). Compilar uma mudança nelas produziria um diff sem efeito físico — corromperia a política E10, que mediria "sem melhora" e escalaria uma alavanca morta. Habilitá-los exige mudar o baseline (`inputs/attacks/uc05_injection.json`) para `"synthetic"`, o que também mudaria o que `--engine live --attack injection` (loop legado) otimiza — decisão de dados fora do escopo deste catálogo. |
| `sqNumDelta.min`, `padBytes.min` | `high_stnum` | baseline `0`, morto nas duas direções: `decrease` não desce de um piso `0`, e `increase` sem `maximum` é multiplicativo (`0 × fator = 0`). Só os `.max` de cada par entram. |
| `protectStatusChanges` | `grayhole` | gate booleano de `statusChangeDropProb` (com o gate `true`, a probabilidade fica inerte). Catalogar os dois permitiria ao compilador ligar o gate e mudar a probabilidade que ele desativa no mesmo round. Mantém-se só `statusChangeDropProb`. |

## `increase_resource_pressure` — por que nenhum catálogo o anuncia

É o efeito fisicamente mais natural para `flooding`/`grayhole`, mas
`DetectionReport` não tem uma métrica que meça pressão de recurso — as únicas
candidatas seriam `latency_ms` (mede nosso próprio processo Python, não o
barramento GOOSE) ou algo derivado de `DatasetBundle.class_counts`/linhas por
segundo (exigiria uma segunda fonte no `decide_feedback`, mudança de contrato
fora do escopo desta entrega). Anunciar o efeito sem uma métrica real faria
`core/feedback_policy.py` "escolher uma métrica arbitrária e fingir medição" —
exatamente o que o módulo documenta que não faz. Uma campanha que peça esse
efeito é rejeitada no portão de entrada (`resolve_candidate_paths`) com uma
mensagem acionável, em vez de morrer tarde na rodada 1 com
`NO_OBJECTIVE_METRIC`. `increase_attack_activity` cobre a metade fisicamente
representável do pedido nos mesmos campos (`burst.*`, `gapMs.*`,
`dropRate.*`, `count.lambda`) — a perda é lexical, não física.

## Limitação de modo cached

Em `--generator-mode cached` o dataset servido é sempre
`data/baseline_dataset.csv`, cuja classe de ataque é `masquerade_fake_fault`.
O portão de integração E4 (`build_dataset_bundle`,
`expected_attack_label=spec.label`) exige que esse rótulo esteja presente no
trace — então **os 10 ataques que não são `masquerade_fault` só são
fisicamente mensuráveis em `--generator-mode jar`**; em modo cacheado o
pipeline para no estágio PREPROCESS com um erro do gate, não com um crash.
Os testes (`tests/test_intent_loop_orchestrator.py`) sintetizam um CSV
mínimo por rótulo para exercitar os sete estágios sem Java.

As 4 variantes `delayed_replay*` compartilham `label="delayed_replay"` no
registry — o gate E4 não as distingue por classe entre si.

## Como este documento foi gerado

```bash
uv run python -c "
from adversarial_ids.config.attack_capabilities import ATTACK_CAPABILITY_CATALOG
from adversarial_ids.config.attacks_registry import list_attack_keys, get_attack_spec
for k in list_attack_keys():
    cap = ATTACK_CAPABILITY_CATALOG[get_attack_spec(k).intent_capability_id]
    print(cap.capability_id, len(cap.fields))
"
```

`tests/test_attack_capabilities_catalog.py` mantém este catálogo honesto —
qualquer alavanca morta, caminho não documentado ou efeito sem campo que o
produza falha ali antes de chegar aqui.
