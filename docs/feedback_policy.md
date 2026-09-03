# Política de feedback e campanhas multi-rodada (E10)

## Objetivo

O agente Defensor atua como o estágio DEFENDER do pipeline intent-driven; o
estágio FEEDBACK vem logo depois e fecha o loop Red×Blue. Recebe o
`DetectionReport` do estágio DETECTOR e o `DefensePlan` do estágio DEFENDER
desta rodada e decide, deterministicamente: se a campanha continua e, se
sim, qual é a próxima `IntentSpec`. Nenhuma LLM participa deste estágio —
mesmo guardrail central do pipeline: o modelo só produz a intenção ou o
plano de defesa, tudo a jusante é Python determinístico
(`core/feedback_policy.py::decide_feedback`).

`IntentLoopOrchestrator.run(prompt)` continua resolvendo **uma** rodada.
`IntentLoopOrchestrator.run_campaign(prompt, rounds=N)` encadeia até `N`
rodadas: a partir da segunda, a `IntentSpec` não vem mais de uma nova
interpretação do prompt — vem de `FeedbackDecision.next_intent` da rodada
anterior.

## Contrato de saída

`domain/feedback_decision.py::FeedbackDecision` (`extra="forbid"`,
`frozen=True`):

- `should_continue` / `stop_reason` — sempre concordam entre si (validador).
- `objective_metric` (`"f1"` ou `"recall"`, ou `None`) e `objective_value` —
  o que foi medido nesta rodada.
- `best_value_so_far` / `best_round` — o melhor valor da campanha até aqui.
- `improvement` — `melhor_valor_anterior − valor_desta_rodada` (positivo =
  melhorou).
- `round` / `run_id` / `parent_run_id` — a linhagem desta decisão.
- `defense_priority` — a prioridade do `DefensePlan` da rodada (vem do
  plano, não recalculada: o portão do E5 já garante que ela concorda com
  `priority_from_report`).
- `next_intent` — presente se e somente se `should_continue`.
- `rationale` — frase determinística com os números que justificam a
  decisão.

Persistido como `feedback.json` no diretório da rodada, pelo mesmo
`_stage(...)` que persiste os outros seis estágios — sem I/O especial.

## Métrica-objetivo por efeito

| `desired_effect` | métrica minimizada |
|---|---|
| `lower_f1` | `f1` |
| `lower_recall` | `recall` |
| `mimic_normal_traffic` | `recall` |
| `increase_attack_activity` | nenhuma |
| `increase_resource_pressure` | nenhuma |

Os dois últimos efeitos não têm uma métrica de evasão no `DetectionReport`
para minimizar — inventar uma seria fingir medição. A política para na
primeira rodada com `no_objective_metric`.

## Escada de escalonamento

Só duas alavancas mudam de fato a variante compilada por
`core/intent_compiler.py` (ver o módulo `feedback_policy.py` para a análise
completa), então a escada usa exatamente elas, "aprofundar antes de
alargar":

1. `intensity`: `low` → `medium` → `high`.
2. `restrictions.max_fields_changed`: `+1` por rodada, até o número de
   campos candidatos do efeito **naquele ataque** — reaproveita
   `resolve_candidate_paths` para calcular o teto, o mesmo portão que o
   compilador e o `IntentAgent` usam. Varia por catálogo: 12 em
   `masquerade_fault`/`lower_recall`, 15 em `random_replay`, só 2 em
   `injection` (ver `docs/attack_capabilities.md`). O teto do contrato
   (`domain.intent_spec.MAX_FIELDS_CHANGED_CEILING`, hoje 16) é só uma trava
   de sanidade acima desse número — nunca o número em si.

A `seed` não é alavanca: `intent_compiler._select_field_paths` embaralha os
candidatos com `random.Random(intent.seed)` e corta em `[:limite]`, então
manter a seed fixa faz de `limite+1` um superconjunto do conjunto anterior
— a escalada fica monotônica ("os mesmos campos, mais um") e a campanha
inteira reproduzível.

`desired_effect` também não é alavanca: `lower_f1`, `lower_recall` e
`mimic_normal_traffic` compartilham os mesmos campos candidatos e a mesma
direção no compilador — a política nunca reescreve `objective`,
`base_attack` nem `desired_effect`.

Toda intenção derivada é revalidada com `IntentSpec.model_validate(...)`
(não `model_copy(update=...)`, que no Pydantic v2 não re-roda validadores)
e passa por `validate_intent_capability` antes de virar `next_intent` —
defesa em profundidade: a política nunca devolve uma intenção que o
compilador rejeitaria.

## Condições de parada

Primeira que casar, nesta ordem:

1. `no_objective_metric` — efeito sem métrica de evasão.
2. `objective_reached` — `target_metric` informado e atingido.
3. `no_improvement` — platô: `ganho < min_delta` (comparado ao **melhor**
   valor já medido na campanha, não ao da rodada imediatamente anterior,
   porque cada compile reparte do baseline).
4. `levers_exhausted` — motivo científico ("não há mais o que escalar")
   vem antes do orçamento de rodadas.
5. `max_rounds_reached` — teto de rodadas.
6. `continue_campaign` — segue com `next_intent`.

## Linhagem da campanha

`LoopRecord` ganhou dois campos aditivos (E10): `round` (default `1`) e
`parent_run_id` (default `None`), com o invariante "só a rodada 1 não tem
pai". `schema_version` continua `1`: os dois campos são aditivos com
default, então um registro gravado antes do E10 carrega normalmente
(`tests/test_loop_record_store.py::test_records_persisted_before_the_e10_lineage_fields_still_load`)
e o formato em disco (`{"loop_records": [...]}`) não muda. Uma campanha de
N rodadas é N linhas em `outputs/loop_records.json`, encadeadas por
`parent_run_id` — não um contrato novo de campanha.

## Estágio INTENT nas rodadas 2+

Quando `run_campaign` fornece `intent_override`, o estágio `intent` não
chama o `IntentLike` — persiste a mesma `intent.json` e é registrado como
**`skipped`**, nunca `succeeded`: o status precisa continuar dizendo a
verdade sobre o que rodou. `succeeded` leria como "a LLM rodou de novo", o
que apagaria exatamente a propriedade que o E10 entrega (o lado Red é
Python determinístico da rodada 2 em diante).

## Segurança e escopo

- Nenhuma segunda chamada de LLM por rodada de campanha — o `IntentAgent`
  só roda na rodada 1.
- A política não aplica nada: `next_intent` é uma proposta que ainda passa
  pelo compilador e pelos portões normais na rodada seguinte.
- O `DefensePlan` continua sem campo de execução — o E10 não muda isso.

## Falha do estágio

Uma falha no FEEDBACK (ou em qualquer estágio anterior da rodada) marca o
`LoopStage` correspondente como `failed` e interrompe **a rodada** —
`run_campaign` então interrompe **a campanha**, sem levantar exceção: os
registros já concluídos (cada um persistido individualmente) são
devolvidos, e a causa fica no `LoopStage` que falhou.

## Limitação do modo cacheado

Em `generator_mode="cached"` o dataset do baseline e o do candidato
compilado são o mesmo arquivo — a métrica-objetivo não se move entre
rodadas, então uma campanha cacheada para na rodada 2 com
`no_improvement` (a menos que `feedback_min_delta=0.0`, usado nos testes
para exercitar mais rodadas sem depender de medição real). Use
`generator_mode="jar"` para uma campanha fisicamente mensurável.

## Fora de escopo e trabalho futuro

- **Alavanca dirigida por feature**: mapear `DetectionReport.top_features`
  para `allowed_fields`/`forbidden_fields` deixaria a escalada mirar (ou
  evitar) os campos ligados às features que mais pesam na decisão do
  detector. Não existe hoje um mapeamento feature-do-dataset → campo de
  config no repositório; adicioná-lo é a extensão natural deste épico.
- `LoopStage.error` segue como o único campo de texto livre, carregando
  tanto mensagens de falha quanto a mensagem "intenção herdada" — renomear
  para algo como `detail` é uma quebra de contrato sem ganho funcional por
  si só.
- **Métrica de pressão de recurso**: `increase_resource_pressure` não tem
  entrada em `_OBJECTIVE_METRIC_BY_EFFECT` e nenhum `AttackCapability` o
  anuncia (ver `docs/attack_capabilities.md`) — `DetectionReport` não carrega
  nada que meça pressão de recurso sem inventar uma medição arbitrária
  (`latency_ms` mede o processo Python local, não o barramento GOOSE). Uma
  métrica honesta viria de `DatasetBundle` (volume/taxa de linhas de ataque),
  não de `DetectionReport` — exigiria uma segunda fonte na assinatura de
  `decide_feedback`, mudança de contrato fora do escopo deste épico.

## Testes

- `tests/test_new_contracts.py` — `FeedbackDecision` (versionamento,
  congelamento, round-trip JSON, validadores de consistência) e a
  linhagem do `LoopRecord` (`round`/`parent_run_id`), junto dos demais
  contratos congelados.
- `tests/test_feedback_policy.py` — métrica-objetivo por efeito; escada
  intensity→largura; teto pela allowlist de campos candidatos;
  determinismo (mesma entrada → mesma decisão); a intenção escalada ainda
  compila; precedência dos cinco motivos de parada, incluindo os casos em
  que dois motivos valem ao mesmo tempo.
- `tests/test_intent_loop_orchestrator.py` — os sete estágios completos
  numa rodada avulsa; `feedback.json` persistido; `run_campaign` parando no
  platô em modo cacheado; rodando até o teto quando o epsilon é zero;
  chamando o `IntentLike` uma única vez por campanha; abortando quando um
  estágio falha; escalando a intensidade da rodada 2; falha do próprio
  estágio FEEDBACK.
- `tests/test_cli_intent_engine.py` — `--rounds` encaminhado, rejeição de
  `--rounds` não positivo, `--iterations` (do loop legado) ignorado no
  motor intent, resumo por rodada, saída não-zero quando qualquer rodada
  falha.
- `tests/test_loop_record_store.py` — um `LoopRecord` gravado antes do E10
  (sem `round`/`parent_run_id`) continua carregando com os defaults.
