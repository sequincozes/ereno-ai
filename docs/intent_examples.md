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
