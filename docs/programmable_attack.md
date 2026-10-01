# Ataque programável (uc11) — comportamentos fora do catálogo

Os dez ataques catalogados são famílias fixas (replay, flooding, grayhole, …).
O **ataque programável** é a via para um comportamento que não é nenhum deles: o
usuário descreve em linguagem natural o que quer, e isso vira uma **lista de
regras** aplicadas às mensagens GOOSE. Um comportamento novo é **dado
validado**, não uma classe Java nova.

O guardrail é o mesmo das fases anteriores, um degrau acima: na Fase 1 o LLM
propunha um *valor*; aqui ele autora uma *regra*; em ambos, Python determinístico
e um schema Java decidem se é executável. Nenhum Java é escrito pelo LLM.

## A regra

Cada regra é `{op, field, value, fraction, when}`:

| campo | significado |
|---|---|
| `op` | `set`/`add`/`scale` (muta o campo), `drop` (descarta a mensagem), `duplicate` (repete a mensagem) |
| `field` | o campo que a mutação toca (`drop`/`duplicate` o ignoram) |
| `value` | o operando da mutação, ou o número de cópias **extras** do `duplicate` (1 a 10) |
| `fraction` | probabilidade em [0,1] de a regra disparar nas mensagens que ela mira |
| `when` | a **seleção**: `{field, cmp, value}`; `cmp="always"` mira toda mensagem |

Campos disponíveis: os inteiros `cbStatus`, `stNum`, `sqNum`,
`gooseTimeAllowedtoLive`, `confRev`, e os de tempo `t` e `timestamp` (em
**segundos**).

As regras vivem sob `rules` como um **objeto** de slots (`{"r0": {...}, "r1":
{...}}`), não um array. O modelo de caminhos do repo é dict pontilhado
(`rules.r0.when.cmp`) e não tem suporte a índice de array; um dict de slots
mantém as duas pontas — o compilador Python e o `ProgrammableCreatorC` Java —
sobre o mesmo modelo de caminho. O Java aplica os slots em ordem de chave.

### Três eixos, uma gramática só

- **Mutação** — `set`/`add`/`scale` sobre um campo do quadro.
- **Seleção** (`when`) — a regra só age nas mensagens em que a condição vale.
  A condição é lida da mensagem **original**, antes de qualquer mutação: qual
  mensagem uma regra mira não pode depender do que o slot anterior já mudou, ou
  a ordem dos slots viraria parte do comportamento. As mutações, essas sim,
  encadeiam em ordem de slot.
- **Temporização** — `add` nos campos de tempo, com sinal.

## Por que não existe uma op `delay` nem uma op `reorder`

Não é economia de gramática: é que não poderia funcionar. O fluxo escrito pelo
ERENO **não** está na ordem em que o creator emite — as mensagens saem por um
`PriorityQueue<EthernetFrame>` cujo `compareTo` ordena por `timestamp`, e as
features de delta (`timestampDiff`, `tDiff`, `delay`) são calculadas entre
mensagens consecutivas *nessa* ordem. Bufferizar ou trocar mensagens de lugar no
creator seria apagado pela fila.

Então o relógio **é** a ordem:

| o pedido | a regra |
|---|---|
| "atrase 20 ms" | `op=add`, `field=timestamp`, `value=0.02` |
| "tire de ordem / antecipe" | `op=add`, `field=timestamp`, `value=-0.02` |
| "descarte 40%" | `op=drop`, `fraction=0.4` |
| "duplique cada mensagem" | `op=duplicate`, `value=1` |

`set` e `scale` são **recusados** nos campos de tempo: eles são relógios
absolutos, e setar ou escalar um realoca a mensagem para um ponto arbitrário da
captura em vez de deslocá-la. Só `add` tem significado ali. A regra é do schema
(`domain/attack_configs/programmable.py`), não da allowlist — é uma coerência
*entre* dois campos da mesma regra, que nenhuma allowlist campo-a-campo
consegue expressar: o portão aprova `op=scale` e `field=timestamp`
separadamente, e é o schema que recusa a combinação, no compilador.

## Como uma intenção vira regras

O ponto de desenho: a capacidade separa dois tipos de campo por slot.

- **`fraction` carrega efeito.** É a única alavanca que a heurística do
  compilador varre por intensidade — porque é a única que mapeia limpo para
  evasão (menos mutação → mais parecido com o benigno) e atividade (mais
  mutação). Baseline 0.5 para ter folga nas duas direções; piso 0.05.
- **`op`/`field`/`value`/`when.*` têm efeito vazio.** São o *comportamento* em
  si, e a heurística nunca os toca. O LLM os **autora** via `target_values`
  (Fase 1), que aplica um valor ditado sem depender de efeito e nunca o clampa.

Assim, "atrase em 20 ms as mensagens com o disjuntor fechado" vira, sem um
compilador novo:

```json
"target_values": {
  "rules.r0.when.field": "cbStatus", "rules.r0.when.cmp": "eq",
  "rules.r0.when.value": 1,
  "rules.r0.op": "add", "rules.r0.field": "timestamp",
  "rules.r0.value": 0.02, "rules.r0.fraction": 1.0
}
```

A gramática (quais ops/campos/comparadores/faixas são legais) está na
capacidade; o LLM preenche os slots. Um `field` fora da allowlist do creator, um
`fraction` fora de [0,1] ou um comparador inventado é recusado no portão
determinístico — não corrigido em silêncio.

## Uso

```bash
uv run adversarial-ids --engine intent --generator-mode jar \
  --prompt "Atrase em 20 ms só as mensagens com o disjuntor fechado."
```

Só é fisicamente mensurável em `--generator-mode jar` (em cached todo trace é o
mesmo arquivo). O ataque produz a classe `programmable` no dataset.

## O baseline traz dois slots com regra default

`inputs/attacks/uc11_programmable.json` ship com `r0` (set cbStatus=1) e `r1`
(add stNum+1), ambos fraction 0.5 e `when.cmp="always"` — um ataque programável
válido e detectável por padrão.

O `when` **precisa** estar no baseline, e precisa ser inerte: `target_values`
aplica um valor a um caminho existente, não cria chave que falta, então um
`when` opcional nunca seria autorável; e um `when` com default não-neutro
mudaria o ataque default. `cmp="always"` resolve os dois — ignora
`when.field`/`when.value` e reproduz exatamente o comportamento anterior à
seleção.

Uma intenção que autora só `r0` **mantém `r1` no default**: o trace resultante
aplica a regra autorada mais a `r1` do baseline. Para um comportamento "do zero",
zere a fração do slot que não quer (`rules.r1.fraction: 0`) ou diga isso no
prompt. Dois slots é o que permite autorar comportamentos de duas regras; o
custo é carregar dois defaults.

## `drop` e o gate de prevalência do E4

`drop` reduz o volume da classe de ataque: o creator emite uma mensagem a menos
por disparo. Com `fraction` alto isso derruba a prevalência da classe
`programmable` e pode esbarrar no piso de 1% do gate E4
(`INTENT_LOOP_MIN_ATTACK_PREVALENCE`) — o que é o gate fazendo o trabalho dele,
não um bug: um ataque raro demais não produz métrica que signifique alguma
coisa. Medido: `fraction=0.4` deixa 30001 linhas de ataque das 50000 (60%), bem
acima do piso. Já `duplicate` vai na direção oposta e multiplica o trace — daí
o teto de 10 cópias extras.

## Escopo

A superfície é limitada aos campos com getter/setter em `Goose` que o creator
sabe ler e escrever; ampliá-la é aditivo (mais entradas na allowlist dos dois
lados). O que **não** existe: regras que dependem de mais de uma mensagem
(janelas, rajadas, estado entre mensagens) — cada regra decide olhando só a
mensagem corrente.

## Verificação

- `tests/test_programmable_attack.py` — registro, schema (dict de slots, ops
  válidos, fraction em [0,1], `when` obrigatório), a coerência op↔campo de tempo
  e o limite de cópias do `duplicate`, a capacidade (só fraction com efeito), a
  autoração de comportamento/seleção/atraso via `target_values`, o nunca-clampar,
  e o determinismo.
- `tests/test_attack_capabilities_catalog.py` / `test_intent_loop_orchestrator.py`
  — o programmable passa as seis invariantes do catálogo e os sete estágios em
  cached mode como qualquer ataque.
- **JAR (26/09, Fase 2.1):** "scale sqNum×3 em 40% + set TTL=5" gerou um trace
  com 50000 linhas da classe `programmable`.
- **LLM real (`openai/gpt-oss-120b`, 26/09):** o prompt "reescreva o sqNum
  multiplicando por 3 em ~40%" → o LLM escolheu `base_attack=programmable` e
  autorou `{field:sqNum, op:scale, value:3, fraction:0.4}`, em 6826 tokens (sob o
  TPM de 8000).
- **JAR (01/10, Fase 2.2), seed 4242, quatro configs:**
  - *mutação só* (regressão da 2.1): 50000 linhas de ataque — o `when` neutro
    não mudou nada;
  - *seleção + atraso* (`when cbStatus=1`, `add timestamp +0.02`, fraction 1.0):
    as linhas com `cbStatus=1` ficaram com `timestamp - t` **mínimo exatamente
    0.02** e as com `cbStatus=0` com mínimo 0.00 — o atraso caiu só na seleção;
  - *duplicação* (`duplicate value=2`, fraction 1.0): 150002 linhas, exatamente
    3× a baseline, e os zeros de `timestampDiff` saltaram de 31512 para 128706
    (as cópias ficam adjacentes com o mesmo relógio);
  - *descarte* (`drop`, fraction 0.4): 30001 linhas, exatamente os 60% que
    sobram.
- **Regressão dos outros ataques:** comparados os CRCs das 281 classes
  `br/ufu/facom/ereno/**` entre o JAR da 2.1 e o da 2.2 — **só
  `ProgrammableCreatorC.class` difere**.

## Nota de manutenção

Três pares precisam concordar entre Python e Java, e o Java é sempre a última
linha do guardrail (se o Python aceitar o que o Java não conhece, a geração
falha alto com `IllegalArgumentException`, por desenho):

| Python | Java |
|---|---|
| `MUTABLE_FIELDS` (`domain/attack_configs/programmable.py`) | `ProgrammableCreatorC.FIELD_ALLOWLIST` |
| `TIME_FIELDS` (idem) | `ProgrammableCreatorC.TIME_FIELDS` |
| `MAX_EXTRA_COPIES` (idem) | `ProgrammableCreatorC.MAX_EXTRA_COPIES` |

Do lado Python essas constantes moram no **domínio** e a capacidade
(`config/capabilities/programmable.py`) as importa — o schema precisa delas para
as regras de coerência, e `config/` já depende de `domain/`, nunca o contrário.
Uma lista só, nas duas pontas do Python.
