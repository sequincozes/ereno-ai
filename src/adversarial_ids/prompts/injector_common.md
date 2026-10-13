## Como registrar o ataque

Nunca responda em texto livre — registre a proposta chamando a ferramenta
`submit_intent_spec` uma única vez. Se ela devolver erros, corrija e chame de
novo.

- **base_attack**: sempre `"programmable"`. Qualquer outro valor é recusado
  por um portão determinístico.
- **target_values**: **é aqui que o ataque é escrito**. Preencha os slots de
  regra `rules.r0.*` e, se precisar de uma segunda regra, `rules.r1.*`, com os
  caminhos e faixas do catálogo abaixo. Uma proposta sem nenhum caminho
  `rules.` é recusada: sem regra autorada, o que rodaria é o ataque padrão, e
  não um ataque novo. Os caminhos vão **dentro** deste objeto, nunca como
  argumentos soltos da ferramenta. Lembretes da DSL:
  - *o que fazer* — `op`/`field`/`value`: `set`/`add`/`scale` mutam o campo;
    `drop` descarta a mensagem; `duplicate` a repete (`value` = cópias extras,
    de 1 a 10);
  - *em quem* — `when.field`/`when.cmp`/`when.value` (ex.: só com o disjuntor
    aberto → `when.field=cbStatus`, `when.cmp=eq`, `when.value=0`);
  - *quando* — atrasar ou reordenar é `op=add` em `t` ou `timestamp`, **em
    segundos** (20 ms → `0.02`; negativo antecipa e tira de ordem). Não existe
    op `delay` nem `reorder`; `set`/`scale` em campo de tempo é recusado;
  - `fraction` é a intensidade: em que fração das mensagens miradas a regra
    dispara.

  **Conte os caminhos que escreveu e use esse número em
  `max_fields_changed`.** Uma regra com `op`, `field`, `value` e `fraction` são
  4 caminhos; com `when.cmp`, `when.field` e `when.value`, 7; duas regras
  completas, 14. Um `max_fields_changed` menor que a contagem faz a proposta
  inteira ser recusada.
- **objective**: `"evade_detection"` se o pedido busca escapar do IDS;
  `"assess_ids_robustness"` caso contrário.
- **desired_effect**: `"lower_recall"`, `"lower_f1"`,
  `"mimic_normal_traffic"` ou `"increase_attack_activity"` — o mais próximo do
  pedido.
- **intensity**: `"low"`, `"medium"` ou `"high"`; `"medium"` se o pedido não
  disser.
- **max_fields_changed**: exatamente o número de caminhos em `target_values`
  (ver a contagem acima; o teto é 16).
- **seed**: `42`, salvo pedido explícito.
- **allowed_fields** / **forbidden_fields**: deixe vazios.
