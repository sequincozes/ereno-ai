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
   apontar para um relatório inventado;
7. ação cuja técnica não responde a nenhuma evidência que ela própria cita —
   a avaliação por regras, descrita na seção seguinte.

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

## Avaliação por regras (`core/defense_rules.py`)

Os seis itens acima provam que o plano **não mente**: a métrica existe, o valor
bate, o alvo cobra melhora. Nenhum deles prova que a recomendação **responde** ao
problema. Um plano que cita `recall` e propõe `goose_authentication` passava
inteiro — o balde é legal, a evidência é real, o teste é remedível —, e
autenticar publisher simplesmente não é o que um recall baixo pede.

`core/defense_rules.py::evaluate_plan_rules` fecha isso consultando o catálogo
feature→técnica (`config/defense_techniques.py`, ver a seção "Técnicas e
baldes"). Uma ação está **sustentada** quando sua `technique` aparece no
catálogo para pelo menos uma das evidências que ela cita — pelo menos uma, e não
todas: um plano sério cita a métrica que dói *e* a feature que a explica, e
exigir que a técnica respondesse às duas rejeitaria justamente a ação mais bem
fundamentada.

A avaliação é determinística e pura (nenhuma LLM, mesma disciplina do E10), e
nunca levanta exceção por conteúdo: ela devolve um `DefenseRuleReport`, e quem
recusa é o portão. Um plano ruim precisa ser legível antes de ser recusado.

### As cinco regras

| regra | severidade | o que ela vê |
|---|---|---|
| `technique_not_grounded` | **blocking** | a técnica não é resposta catalogada a nenhuma evidência citada |
| `evidence_not_actionable` | advisory | a ação cita `model_name`/`split`: identificam a execução, mas não há o que remediar neles |
| `uncatalogued_feature` | advisory | a evidência é uma feature real que o catálogo ainda não cobre — lacuna do catálogo, não defeito do plano |
| `playbook_technique_missing` | advisory | o playbook IEC-61850 daquela família prescreve técnicas que o plano não propõe |
| `playbook_signature_ignored` | advisory | o relatório destacou uma feature-assinatura do playbook e nenhuma ação a citou |

Só a primeira recusa. As duas do playbook falam do **cenário**, não desta
rodada: um playbook de replay pede validação de sequência mesmo sob um
`PROTOCOL_FEATURES_MODE` em que nenhum delta chega ao relatório (ver
`docs/preprocessing.md`). Bloquear por elas transformaria a resposta certa para
*esta* execução em erro; silenciá-las jogaria fora a única leitura que o
catálogo tem do cenário.

O eixo do playbook só liga quando o chamador informa `attack_key` — o
orquestrador passa `IntentSpec.base_attack`, o mesmo ataque que a intenção
compilada usou. Sem ele o relatório sai com `playbook_key=None`, dizendo que
aquele eixo não foi avaliado em vez de deixar a ausência de achados parecer
aprovação. `playbook_signature_ignored` só cobra features que o relatório
**de fato** destacou em `top_features`: cobrar uma ausente seria cobrar do plano
algo que o relatório não tinha como mostrar.

### O prompt e o portão leem o mesmo mapa

`techniques_by_evidence` chega no contexto do Defensor com as técnicas
catalogadas para cada chave de `citable_evidence`, sob o mesmo `top_n`. Os dois
lados consultam `config/defense_techniques.py::techniques_for_evidence`, então o
que o prompt oferece é literalmente o que o portão exige — mesma disciplina de
`select_evidence_candidates` para evidência e de `legal_techniques_by_bucket`
para balde. Sem isso, a LLM escolheria a técnica às cegas e descobriria a regra
só na recusa. `model_name` e `split` ficam de fora do mapa: mostrá-los com lista
vazia convidaria a citá-los sozinhos.

### O artefato `defense_rules.json`

O estágio DEFENDER persiste o `DefenseRuleReport` ao lado do plano
(`IntentLoopOrchestrator._run_defender`). `grounded_actions`/`total_actions` é,
literalmente, a fração de recomendações ligadas a evidência que o gate de saída
da janela D47-54 cobra — e os achados advisory ficam registrados por execução,
que é o que permite ver ao longo de uma campanha qual playbook o Defensor
sistematicamente ignora.

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
- a avaliação por regras: lastro por ação e por balde, a técnica alheia à
  evidência, as duas evidências que não sustentam nada (descritiva e fora do
  catálogo), os dois achados de playbook e os invariantes do próprio
  `DefenseRuleReport` — contagem que contradiz os achados, severidade
  rebaixada, achado de ação sem localização (`tests/test_defense_rules.py`);
- a recusa do portão pela regra de lastro e a fração citada na mensagem, mais
  `techniques_by_evidence` contra a mesma consulta que o portão usa
  (`tests/test_defender_tools.py`);
- o estágio DEFENDER de ponta a ponta no orquestrador, incluindo o caminho
  de falha e o `defense_rules.json` persistido
  (`tests/test_intent_loop_orchestrator.py`).
