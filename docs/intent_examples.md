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
`sqnumResetProb = 0`, vale o 0,4 da baseline, e os reinícios de sqNum também
denunciam o ataque.

Com dois casos seguidos ficou claro que o problema é estrutural, não do
prompt. A política passou a pular o degrau de intensidade quando toda a cota
está nos valores ditados, e vai direto para alargar `max_fields_changed`
(`core/feedback_policy.py::_escalate`, ver `docs/feedback_policy.md`).
