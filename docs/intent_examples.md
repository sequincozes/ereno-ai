# Exemplos qualitativos do `--engine intent` (TCC/TSG)

Dois exemplos do pipeline dirigido por intenção, para ilustrar o caminho
prompt → `IntentSpec` → compilador → JAR → detector → Defensor → feedback.
Todos rodaram com `--generator-mode jar` e `PROTOCOL_FEATURES_MODE=deltas`.
Os artefatos ficam em `outputs/intent_loop/<run_id>/`, e o registro da
campanha em `outputs/loop_records.json`.

## Exemplo 1a: high_stnum com todos os campos ditados (03/11/2026)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "evada o IDS com o ataque high_stnum usando salto de stNum entre 2 e 5 e probabilidade de reinício do sqNum igual a 0"
```

Rodadas `994bb292…` (1) e `0fcbbfc0…` (2). Log em
`outputs/experiments/intent_ex1_high_stnum.log`. Saída 0, 16,4k tokens no total.

- **Os valores ditados chegaram intactos.** O LLM extraiu três `target_values`
  (`jump.min = 2`, `jump.max = 5`, `sqnumResetProb = 0`), e o compilador os
  aplicou sem mexer, fora da escada de intensidade. Não houve clamp.
- **Não evade.** Treino interno: recall 0,997. Variante contra o baseline:
  recall 0,990 e F1 0,994, com `stDiff` dominando a importância (0,68). Um
  salto de 2 a 5 no stNum ainda é um salto que `stDiff` enxerga.
- **A escalada não teve onde agir.** O feedback subiu a intensidade de medium
  para high, mas os três campos que o compilador muda (`max_fields_changed = 3`)
  já estavam todos fixados no pedido. A config da rodada 2 é idêntica à da
  rodada 1, e o trace também, byte a byte (mesmo seed 42). Por isso a
  campanha parou em `no_improvement` (ganho 0,0000) na rodada 2 de 3, antes de
  alargar `max_fields_changed`. Isso está correto pelas regras (um valor ditado
  nunca é sobrescrito), mas mostra uma limitação da política: ela gasta uma
  rodada numa escalada que já se sabe vazia antes de gerar o trace.
- **Os planos do Defensor saíram genéricos e de prioridade low:**
  `network_segmentation` na rodada 1, e `detector_retraining` mais
  `network_segmentation` na rodada 2. Nenhum cita feature, o mesmo padrão
  dos planos do E1 e do E3-LLM.

## Exemplo 1b: high_stnum com só o salto ditado (03/11/2026)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "evada o IDS com o ataque high_stnum usando salto de stNum entre 2 e 5"
```

Rodadas `9b20557c…` (1) e `f032ffd7…` (2). Log em
`outputs/experiments/intent_ex1b_high_stnum.log`. Saída 0, 16,0k tokens.

Repetiu o defeito do 1a. O LLM pôs `max_fields_changed = 2`, o mesmo número de
campos ditados, então a intensidade de novo não tinha campo livre. O trace da
rodada 2 saiu idêntico ao da 1, e a campanha parou em `no_improvement`. A
variante evade ainda menos que a do 1a: recall 0,996 e F1 0,997. Sem fixar
`sqnumResetProb = 0`, vale o 0,4 da baseline, e os reinícios de sqNum provavelmente também
denunciam o ataque.

Com dois casos seguidos ficou claro que o problema é estrutural, não do
prompt. A política passou a pular o degrau de intensidade quando toda a cota
está nos valores ditados, e vai direto para alargar `max_fields_changed`
(`core/feedback_policy.py::_escalate`, ver `docs/feedback_policy.md`).

## Exemplo 1c: o prompt do 1a, com a correção (03/11/2026)

Mesmo prompt do 1a, depois do commit `ae300678`. Rodadas `9afdc23e…` (1) e
`ae8d609b…` (2). Log em `outputs/experiments/intent_ex1c_high_stnum.log`.
Saída 0, 17,3k tokens.

| rodada | intensidade | campos | diff compilado | recall | F1 |
|---|---|---|---|---|---|
| 1 | medium | 3 | jump 20–50 → 2–5 (ditado), sqnumResetProb 0,4 → 0 (ditado) | 0,9899 | 0,9944 |
| 2 | medium | 4 | os mesmos, mais `sqNumDelta.max` 5 → 2 (escada) | 0,9772 | 0,9877 |

- **A escada agora anda.** A rodada 1 é byte a byte a mesma do 1a (mesma
  intenção, mesmo seed). Na rodada 2 a política alargou os campos de 3 para 4
  (teto 5) sem tocar na intensidade. O campo novo saiu da escada, com os
  valores ditados intactos.
- **A campanha parou em `no_improvement`, e agora com motivo.** A queda do F1
  (0,0066) ficou abaixo do mínimo de 0,01 (`FEEDBACK_MIN_DELTA`). Desta vez a
  variante mudou de fato; ela só não melhorou o bastante para justificar a
  rodada 3.
- **A evasão continua pequena,** porque `stDiff` segue enxergando qualquer
  salto de stNum. Os falsos negativos subiram de 503 para 1139 em 50k.
- **Os planos do Defensor continuam genéricos e com prioridade low.** Na
  rodada 1: `detector_threshold_tuning` (falsos positivos) e
  `network_segmentation` (latência). Na rodada 2: `detector_retraining`
  (recall) e `traffic_rate_limiting` (latência).

Este é o exemplo 1 que vai para o artigo. O 1a e o 1b ficam como registro
do defeito que ele expôs.

## Exemplo 2: programmable, atraso seletivo (03/11/2026)

```bash
uv run adversarial-ids --engine intent --generator-mode jar --rounds 3 --prompt \
  "Atrase em 20 ms só as mensagens com o disjuntor fechado."
```

Rodadas `10247d66…` (1), `3b82f168…` (2) e `d0ac8093…` (3). Log em
`outputs/experiments/intent_ex2_programmable.log`. Saída 0, 23,5k tokens.

**O LLM autorou a regra certa na primeira tentativa.** Foram 7 `target_values`
em `rules.r0`: `when cbStatus eq 1`, `op = add`, `field = t`, `value = 0.02`,
`fraction = 1`. O compilador os aplicou literalmente.

| rodada | intensidade | campos | o que a escada mudou | recall | F1 |
|---|---|---|---|---|---|
| 1 | medium | 7 | nada (todos os 7 ditados) | 0,637 | 0,702 |
| 2 | medium | 8 | `rules.r1.fraction` 0,5 → 0,275 | 0,560 | 0,628 |
| 3 | high | 8 | `rules.r1.fraction` 0,5 → 0,1175 | 0,504 | 0,572 |

- **A escada completa apareceu, e já com a correção do `ae300678`.** Com os
  7 campos todos ditados, a rodada 2 alargou de 7 para 8 sem passar pela
  intensidade (antes da correção, ela teria repetido o trace). A rodada 3
  subiu medium → high. A campanha parou em `levers_exhausted`, com o teto de
  8 campos atingido.
- **Ressalva: o ataque não é só o atraso.** A baseline do `uc11_programmable`
  traz uma segunda regra, `r1` (soma 1 ao stNum em 50% das mensagens), que o
  prompt não mencionou e que continuou ativa. O campo que a escada mexeu foi
  justamente a fração dela. A queda do F1 vem de diluir o salto de stNum que
  o detector reconhece, não de o atraso de 20 ms ter ficado mais furtivo.
- **Ressalva: o detector foi treinado na baseline do `uc11`.** Essa baseline é
  outro ataque (`set cbStatus = 1` mais stNum + 1), com recall interno de
  apenas 0,775. A variante, portanto, é avaliada fora da distribuição de
  treino.
- **Pela primeira vez, um plano cita feature.** O `device_quarantine` das
  rodadas 1 e 2 cita `stDiff` (a de maior importância, 0,27). A prioridade
  ficou medium nas três rodadas. O resto continua genérico: ajuste de
  limiar, enriquecimento do dataset e segmentação de rede.

Para o artigo, o exemplo 2 mostra que a DSL e o LLM produzem o comportamento
pedido e que a escada percorre os dois degraus. Ele **não** serve como
evidência de que o atraso evade, por causa da regra `r1` que sobrou.
