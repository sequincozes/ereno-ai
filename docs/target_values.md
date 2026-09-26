# Valores ditados pelo pedido (`target_values`)

Antes, o prompt escolhia **qual** ataque e **quanta** intensidade; o valor de
cada campo saía sempre da heurística do compilador. Agora o pedido pode dizer o
valor exato de um campo:

> *"Reduza o recall com a duração da falha entre 50 e 80 ms."*

vira `{"fault.durationMs.min": 50, "fault.durationMs.max": 80}` em
`IntentSpec.restrictions.target_values`, e a config compilada carrega esses dois
números tal e qual.

O guardrail central não muda: a LLM continua produzindo **apenas** a intenção
tipada. Ela agora propõe também o número, mas quem decide se o número existe é o
catálogo de capacidades, em Python determinístico.

## A regra única

**Um valor ditado nunca é corrigido em silêncio.** Ou ele vale exatamente como
pedido, ou a compilação falha dizendo por quê.

É o que separa um campo fixado de um campo da heurística. A heurística *negocia*:
`_clamp_paired` aproxima um limite do irmão para preservar `min < max`, porque o
valor dela é um meio para um efeito, não um fim. Um valor ditado é um fim — o
usuário disse 80. Ajustá-lo para 79 produziria uma config que afirma um valor que
ninguém pediu, e o `diff` e a justificativa (que o Defensor lê como evidência)
passariam a mentir sobre a origem dele.

Consequência prática: um pedido impossível **para o pipeline** naquele estágio,
com a causa no `LoopStage`, em vez de virar um ataque que ninguém pediu.

## Um campo fixado sai da heurística inteira

| | campo da heurística | campo fixado |
|---|---|---|
| valor | calculado pelo passo da `intensity` | o do pedido |
| escolha do campo | sorteio reprodutível pela `seed` | sempre aplicado |
| precisa carregar o `desired_effect`? | sim | não |
| clamp de par `{min, max}` | sim, cede ao irmão | não, o irmão é que cede |
| conta em `max_fields_changed` | sim | sim |

As duas linhas que merecem explicação:

- **Não precisa carregar o efeito.** A heurística só escolhe entre campos
  *capazes* do efeito pedido. Um valor ditado é uma instrução direta, mais forte
  que essa escolha — se o pedido nomeia o campo e o valor, o compilador aplica.
  Por isso um pedido inteiramente ditado também não cai no erro "as restrições
  removem todos os campos": ele não removeu campo nenhum, nomeou os campos um a
  um.
- **Conta na cota.** Um campo fixado é um campo alterado. Se o pedido fixa 4
  valores e `max_fields_changed` é 3, o portão recusa dizendo o número a usar —
  em vez de escolher sozinho quais 3 dos 4 obedecer.

## Onde cada coisa é recusada

A validação é dividida por **o que cada camada consegue saber**, não por
conveniência:

| camada | decide | exemplos |
|---|---|---|
| contrato (`domain/intent_spec.py`) | contradições internas do pedido, sem saber de que ataque ele fala | mesmo campo fixado duas vezes; campo fixado *e* proibido; campo fixado fora de `allowed_fields`; número entre aspas numa lista |
| portão (`config/attack_capabilities.py::validate_target_values`) | o que o catálogo decide sem abrir a baseline | caminho fora da allowlist; tipo errado; fora de `minimum`/`maximum`/`choices`; par `{min, max}` invertido com **os dois** limites fixados; mais fixados que a cota |
| compilador (`core/intent_compiler.py::_assert_pinned_ranges`) | o que só a baseline decide | limite fixado contra o irmão que ficou na baseline (`min=900` contra `max=800`) |

O portão roda dentro de `IntentAgent.interpret`, então quase tudo é recusado
**antes** de qualquer geração — a LLM recebe o erro e corrige na mesma chamada.
Só o último caso chega ao estágio GENERATOR.

## Duas decisões de tipo que não são óbvias

- **`True` nunca passa como inteiro.** `isinstance(True, int)` é verdadeiro em
  Python; sem a checagem explícita, `cbStatus: true` chegaria ao JSON do ERENO.
- **`1` passa onde o campo espera `number`.** O pedido diz "probabilidade 1", não
  "1.0" — exigir o ponto seria exigir sintaxe de ponto flutuante do texto em
  linguagem natural. Já `80.5` num campo inteiro é recusado: aí a conversão
  mudaria a magnitude, não a escrita.

A linha entre os dois é essa: **converter a representação, nunca a magnitude.**
Pela mesma razão, `["20", "40"]` é recusado em vez de convertido — a conversão
preservaria o número, mas a assimetria com `"0.42"` (que o Pydantic recusa num
campo escalar) não se explicaria para quem lê o contrato.

## Intervalos

O contrato não tem tipo de intervalo porque a config do ataque também não tem:
`{min, max}` são dois campos. "Entre 50 e 80 ms" são **dois** valores fixados. As
duas invariantes valem:

- `min < max` **estrito** — `min == max` passa por todo gate deste repo e o ERENO
  recusa a execução inteira (*"The lower limit must be less than the upper
  limit"*); é a mesma invariante que o piloto E0 descobriu, agora aplicada a
  valores ditados (ver `docs/pilot_e0.md`);
- fixar **um** limite e deixar o outro na baseline é válido, desde que a ordem
  sobreviva. Se a heurística escolher o irmão, ela cede ao valor fixado: os
  fixados são escritos na config **antes**, então o clamp já os enxerga.

## O que isto custou no prompt

O catálogo injetado passou a mostrar a faixa aceita de cada campo entre
colchetes (`fault.prob — number [0..1]`, `cbStatus — integer [0|1]`) — sem ela a
LLM não teria como escolher um valor que o portão aceite. Com as instruções
novas, são **+485 tokens por chamada** do estágio `intent`: de ~6,7k para ~7,1k,
contra um TPM de 8.000 na conta do projeto. Continua cabendo uma chamada por
minuto, mas a folga encolheu — ver `docs/pilot_e0.md` e a nota sobre o TPM.

## Compatibilidade

`target_values` é aditivo com default `()`, como `round`/`parent_run_id` no
`LoopRecord`: uma `IntentSpec` gravada antes desta versão carrega normalmente, e
`schema_version` continua `1`. Um pedido sem valores ditados compila exatamente
como antes — a heurística é o caminho comum, não o caminho legado.

## Testes

- `tests/test_intent_target_values.py` — as três camadas de recusa, o valor
  chegando intacto à config, a intensidade não movendo um campo fixado, a cota
  compartilhada com a heurística, o irmão do par cedendo ao valor fixado, a
  lista virando JSON, a escada do E10 sobre uma intenção com valores fixados, e
  o ponta a ponta até o `attack_candidate.json` em disco.
- `tests/test_intent_tool_schema.py` — o schema de `target_values` expõe
  escalares, não objetos vazios (mesmo bug de `list[object]` do Estrategista).
