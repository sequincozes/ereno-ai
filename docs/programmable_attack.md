# Ataque programável (uc11) — comportamentos fora do catálogo

Os dez ataques catalogados são famílias fixas (replay, flooding, grayhole, …).
O **ataque programável** é a via para um comportamento que não é nenhum deles: o
usuário descreve em linguagem natural o que quer, e isso vira uma **lista de
regras de mutação** aplicadas às mensagens GOOSE. Um comportamento novo é
**dado validado**, não uma classe Java nova (Fase 2.1).

O guardrail é o mesmo das fases anteriores, um degrau acima: na Fase 1 o LLM
propunha um *valor*; aqui ele autora uma *regra*; em ambos, Python determinístico
e um schema Java decidem se é executável. Nenhum Java é escrito pelo LLM.

## A regra

Cada regra é `{op, field, value, fraction}`:

| campo | significado |
|---|---|
| `op` | `set` (campo = valor), `add` (campo += valor), `scale` (campo *= valor) |
| `field` | um campo GOOSE inteiro: `cbStatus`, `stNum`, `sqNum`, `gooseTimeAllowedtoLive`, `confRev` |
| `value` | o operando |
| `fraction` | probabilidade em [0,1] de a regra disparar em cada mensagem |

As regras vivem sob `rules` como um **objeto** de slots (`{"r0": {...}, "r1":
{...}}`), não um array. O modelo de caminhos do repo é dict pontilhado
(`rules.r0.fraction`) e não tem suporte a índice de array; um dict de slots
mantém as duas pontas — o compilador Python e o `ProgrammableCreatorC` Java —
sobre o mesmo modelo de caminho. O Java aplica os slots em ordem de chave.

## Como uma intenção vira regras

O ponto de desenho: a capacidade separa dois tipos de campo por slot.

- **`fraction` carrega efeito.** É a única alavanca que a heurística do
  compilador varre por intensidade — porque é a única que mapeia limpo para
  evasão (menos mutação → mais parecido com o benigno) e atividade (mais
  mutação). Baseline 0.5 para ter folga nas duas direções; piso 0.05.
- **`op`/`field`/`value` têm efeito vazio.** São o *comportamento* em si, e a
  heurística nunca os toca. O LLM os **autora** via `target_values` (Fase 1),
  que aplica um valor ditado sem depender de efeito e nunca o clampa.

Assim, "multiplique o sqNum por 3 em 40% do tráfego" vira, sem um compilador
novo:

```json
"target_values": {
  "rules.r0.op": "scale", "rules.r0.field": "sqNum",
  "rules.r0.value": 3,    "rules.r0.fraction": 0.4
}
```

A gramática (quais ops/campos/faixas são legais) está na capacidade; o LLM
preenche os slots. Um `field` fora da allowlist do creator, ou um `fraction`
fora de [0,1], é recusado no portão determinístico — não corrigido em silêncio.

## Uso

```bash
uv run adversarial-ids --engine intent --generator-mode jar \
  --prompt "Reescreva o sqNum multiplicando por 3 em cerca de 40% do tráfego."
```

Só é fisicamente mensurável em `--generator-mode jar` (em cached todo trace é o
mesmo arquivo). O ataque produz a classe `programmable` no dataset.

## O baseline traz dois slots com regra default

`inputs/attacks/uc11_programmable.json` ship com `r0` (set cbStatus=1) e `r1`
(add stNum+1), ambos fraction 0.5 — um ataque programável válido e detectável
por padrão. Uma intenção que autora só `r0` **mantém `r1` no default**: o trace
resultante aplica a regra autorada mais a `r1` do baseline. Para um comportamento
"do zero", zere a fração do slot que não quer (`rules.r1.fraction: 0`) ou diga
isso no prompt. Dois slots é o que permite autorar comportamentos de duas regras;
o custo é carregar dois defaults.

## Escopo desta fase (2.1)

Só **mutação**. Regras de **seleção** (disparar por condição sobre stNum/cbStatus)
e de **temporização** (atraso, descarte, duplicação, reordenação) são a Fase 2.2.
A superfície de mutação é limitada aos campos inteiros com getter/setter em
`Goose`; ampliá-la é aditivo (mais entradas na allowlist dos dois lados).

## Verificação

- `tests/test_programmable_attack.py` — registro, schema (dict de slots, ops
  válidos, fraction em [0,1]), a capacidade (só fraction com efeito), a autoração
  via `target_values`, o nunca-clampar, e o determinismo.
- `tests/test_attack_capabilities_catalog.py` / `test_intent_loop_orchestrator.py`
  — o programmable passa as seis invariantes do catálogo e os sete estágios em
  cached mode como qualquer ataque.
- **JAR (26/09):** "scale sqNum×3 em 40% + set TTL=5" gerou um trace com 50000
  linhas da classe `programmable`.
- **LLM real (`openai/gpt-oss-120b`, 26/09):** o prompt "reescreva o sqNum
  multiplicando por 3 em ~40%" → o LLM escolheu `base_attack=programmable` e
  autorou `{field:sqNum, op:scale, value:3, fraction:0.4}`, em 6826 tokens (sob o
  TPM de 8000). É a DoD da 2.1 ponta a ponta.

## Nota de manutenção

A allowlist de campos mutáveis existe em **dois lugares que precisam concordar**:
`ProgrammableCreatorC.FIELD_ALLOWLIST` (Java) e `_MUTABLE_FIELDS` em
`config/capabilities/programmable.py` (Python). Se o Python aceitar um campo que
o Java não conhece, a geração falha em runtime com `IllegalArgumentException` —
por desenho (o Java é a última linha do guardrail), mas evite a divergência.
