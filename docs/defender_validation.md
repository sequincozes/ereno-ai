# Validação da saída do agente Defensor (E5)

## Objetivo

O agente Defensor atua como o estágio DEFENDER do pipeline intent-driven.
Ele recebe o `DetectionReport` produzido pelo estágio DETECTOR e propõe um
`DefensePlan`: ações de detecção, contenção e hardening, cada uma amarrada
a evidência real do relatório e a um método de verificação.

A resposta da LLM não é aceita diretamente. Toda saída passa por validação
tipada e por regras determinísticas antes de ser persistida no
`LoopRecord` — a DoD do épico E5: "toda ação referencia métrica/feature e
método de verificação".

## Contrato de saída (`schema_version: 2`)

A resposta deve respeitar o modelo `DefensePlan`, composto por:

- prioridade (`low`, `medium`, `high` ou `critical`);
- uma referência opcional ao `DetectionReport` de origem
  (`detection_report_ref`), no topo do plano;
- listas de ações de detecção, contenção e hardening (ao menos uma no
  total).

Cada ação (`DefenseAction`) contém:

- descrição;
- uma **técnica defensiva** nomeada (`technique`);
- ao menos uma evidência;
- um **teste de validação** (`validation_test`).

Cada evidência (`Evidence`) contém:

- a métrica ou feature citada;
- o valor citado;
- uma referência opcional ao `DetectionReport` de origem.

### A v1 e por que ela mudou

Até a janela D36-46 o contrato estava na v1, e a ação carregava um
`validation_method: str` de texto livre: qualquer string não vazia passava,
inclusive `"revalidar"`. O gate de saída do D47-54 é "100% das recomendações
ligadas a evidência **e teste de validação**", e metade dele não era
verificável por tipo nenhum. A v2 troca aquele campo por `ValidationTest`.
A prosa não se perdeu — virou `ValidationTest.procedure`.

### Técnicas e baldes

`DefenseTechnique` é o vocabulário fechado de técnicas defensivas, e vive em
`domain/defense_plan.py` (assim como `DetectorKey` vive em
`domain/detector_manifest.py`): o catálogo feature→técnica em `config/` se
declara em sincronia com ele, nunca o contrário.

Cada técnica é legal em um ou mais baldes. Várias são de duplo uso — uma
allowlist de publisher bloqueia o intruso *e* é controle estrutural —, então
forçar um balde único rejeitaria plano correto. O que o mapa impede é o erro
que a LLM de fato comete: propor `network_segmentation` como ação de
*detecção*, onde ela não mede nada.

O prompt recebe a lista por balde em `legal_techniques`, montada por
`agents/defender/tools.py::legal_techniques_by_bucket` a partir do mesmo mapa
que o contrato consulta para recusar — o que é oferecido e o que é aceito não
têm como divergir.

### O teste de validação

`ValidationTest` responde "como saberemos se isto funcionou?" em termos
remedíveis:

| campo | significado |
|---|---|
| `metric` | a métrica a medir de novo; só valores de `ValidationMetric` |
| `direction` | `increase`/`decrease` cobram mudança; `at_least`/`at_most` são piso/teto |
| `target` | o número a alcançar; finito, não negativo, e em [0, 1] para taxas |
| `split` | opcional; `null` significa o mesmo split do relatório |
| `procedure` | como a medição será feita, em uma frase |

`ValidationMetric` é um subconjunto estrito da allowlist de evidência:
`model_name` e `split` são texto (não há "aumentar o split"), e importância de
feature ficou de fora porque **não é resultado defensivo** — provar que uma
feature ficou mais importante não prova que o ataque passou a ser detectado.

## Validação por schema

O modelo Pydantic `DefensePlan` (`extra="forbid"`, `frozen=True`) rejeita:

- campos extras — inclusive o `validation_method` da v1;
- prioridade fora dos quatro valores permitidos;
- ação sem nenhuma evidência, sem técnica ou sem teste de validação;
- plano sem nenhuma ação nos três baldes;
- técnica fora do vocabulário, ou legítima mas no balde errado;
- métrica de validação que não seja um escalar remedível do relatório;
- alvo NaN, infinito, negativo, ou acima de 1.0 numa métrica de taxa;
- descrição, procedimento ou evidência vazios;
- evidência que referencia uma execução diferente da declarada pelo plano.

## Validação contra o DetectionReport

A função `agents/defender/tools.py::validate_plan_against_report` verifica
se a resposta está sustentada pelos dados reais da execução. A allowlist de
evidências citáveis é montada por `select_evidence_candidates`: os campos
escalares do relatório (`model_name`, `split`, `accuracy`, `precision`,
`recall`, `f1`, `latency_ms`), os quatro campos da matriz de confusão sob
`confusion_matrix.<campo>`, e o nome de cada feature em `top_features`
mapeado para sua importância.

São rejeitados:

1. evidência (métrica ou feature) que não aparece na allowlist;
2. valor de evidência alterado (comparação numérica tolerante a
   arredondamento de JSON, com guarda explícita para não confundir `bool`
   com `int`);
3. evidência duplicada dentro da mesma ação, em qualquer um dos três
   baldes;
4. teste de validação cujo alvo não cobra melhora: "subir o recall para
   0.50" quando o recall já é 0.60 é uma regressão vendida como correção, e
   passaria em qualquer checagem de tipo. Só `increase`/`decrease` são
   comparados contra o valor medido — `at_least`/`at_most` são guardas
   contra regressão, legitimamente já satisfeitas quando a ação existe para
   *proteger* uma métrica que está boa enquanto outra é atacada;
5. prioridade incompatível com `priority_from_report`;
6. `detection_report_ref` presente e diferente do relatório desta
   execução — fecha a lacuna de o Defensor citar uma evidência real mas
   apontar para um relatório inventado.

A divisão entre contrato e portão segue a mesma linha do resto do repo: o
contrato recusa o que é verificável sozinho (vocabulário, tipo, domínio
numérico, coerência interna do plano), e o portão recusa o que só se sabe com
o `DetectionReport` na mão (a evidência é real? o alvo significa alguma
coisa?).

Um guard adicional em `select_evidence_candidates` rejeita a construção da
própria allowlist se o nome de uma feature colidir com uma chave de
métrica reservada (por exemplo, uma feature chamada `f1`) — nunca ocorre
com dados reais do ERENO (features são CamelCase, como `TrapAreaSum`), mas
mantém a allowlist sem ambiguidade quando ocorre.

## Regra de prioridade

A prioridade é calculada deterministicamente a partir do F1 e do recall do
`DetectionReport` (`agents/defender/tools.py::priority_from_report`):

- nível-base pelo F1: `high` se F1 menor que 0.50; `medium` se F1 maior ou
  igual a 0.50 e menor que 0.80; `low` se F1 maior ou igual a 0.80;
- em seguida, se o recall for menor que 0.50, o nível sobe um degrau na
  ordem `low` → `medium` → `high` → `critical`.

A LLM recebe a prioridade esperada no campo `required_priority`, mas a
resposta ainda é validada novamente pelo código.

## Segurança e escopo

O agente:

- analisa somente dados sintéticos;
- não acessa sistemas reais;
- não produz código ofensivo;
- não inventa métrica, feature, valor ou referência ao relatório;
- não afirma causalidade sem evidência;
- produz somente recomendações defensivas — o `DefensePlan` não tem campo
  de execução, e nunca é aplicado automaticamente.

## Falha do estágio

Diferente do Analista legado (que degrada para uma análise ausente em caso
de falha, sem interromper o loop — `AdversarialWorkflow._safe_analyze`), no
pipeline intent-driven uma falha do Defensor — erro de chamada ou plano
rejeitado pelo portão — marca o `LoopStage` `defender` como `failed` com a
causa e interrompe o pipeline naquele ponto
(`IntentLoopOrchestrator.run`). O `LoopRecord` parcial ainda é persistido:
a causa nunca se perde.

## Testes

Os testes automatizados cobrem:

- contrato `DefensePlan` v2 — vocabulário de técnicas, técnica no balde
  errado, domínio do alvo, NaN/infinito, rastreabilidade e os invariantes
  herdados da v1 (`tests/test_defense_plan_contract.py`), além do
  versionamento/congelamento/round-trip comum a todos os contratos
  (`tests/test_new_contracts.py`);
- `priority_from_report` nos quatro níveis e nas fronteiras 0.50/0.80/recall
  0.50;
- montagem da allowlist, truncamento por `top_n` e o guard de colisão de
  nomes;
- a costura entre `ValidationMetric` e a allowlist: toda métrica de validação
  precisa existir num `DetectionReport` real, e o que sobra da allowlist é
  texto ou importância de feature;
- detecção de evidência inventada, valor alterado, evidência duplicada,
  alvo que não cobra melhora e prioridade incompatível;
- o exemplo JSON do prompt, parseado através do `DefensePlan` — um exemplo
  inválido ensinaria o modelo a produzir exatamente o que o portão recusa;
- detecção de `detection_report_ref` divergente;
- conversão de dict, JSON (com e sem cerca ```` ``` ````) e modelo
  Pydantic;
- execução sem chamadas externas usando agente falso
  (`tests/test_defender_agent.py`);
- o estágio DEFENDER de ponta a ponta no orquestrador, incluindo o caminho
  de falha (`tests/test_intent_loop_orchestrator.py`).
