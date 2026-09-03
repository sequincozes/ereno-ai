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

## Contrato de saída

A resposta deve respeitar o modelo `DefensePlan`, composto por:

- prioridade (`low`, `medium`, `high` ou `critical`);
- listas de ações de detecção, contenção e hardening (ao menos uma no
  total).

Cada ação (`DefenseAction`) contém:

- descrição;
- ao menos uma evidência;
- método de verificação.

Cada evidência (`Evidence`) contém:

- a métrica ou feature citada;
- o valor citado;
- uma referência opcional ao `DetectionReport` de origem.

## Validação por schema

O modelo Pydantic `DefensePlan` (`extra="forbid"`, `frozen=True`) rejeita:

- campos extras;
- prioridade fora dos quatro valores permitidos;
- ação sem nenhuma evidência;
- plano sem nenhuma ação nos três baldes;
- descrição, evidência ou método de verificação vazios.

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
4. prioridade incompatível com `priority_from_report`;
5. `detection_report_ref` presente e diferente do relatório desta
   execução — fecha a lacuna de o Defensor citar uma evidência real mas
   apontar para um relatório inventado.

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

- contrato `DefensePlan` (`tests/test_new_contracts.py`);
- `priority_from_report` nos quatro níveis e nas fronteiras 0.50/0.80/recall
  0.50;
- montagem da allowlist, truncamento por `top_n` e o guard de colisão de
  nomes;
- detecção de evidência inventada, valor alterado, evidência duplicada e
  prioridade incompatível;
- detecção de `detection_report_ref` divergente;
- conversão de dict, JSON (com e sem cerca ```` ``` ````) e modelo
  Pydantic;
- execução sem chamadas externas usando agente falso
  (`tests/test_defender_agent.py`);
- o estágio DEFENDER de ponta a ponta no orquestrador, incluindo o caminho
  de falha (`tests/test_intent_loop_orchestrator.py`).
