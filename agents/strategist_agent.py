from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq


class StrategistAgent:
    """
    Agente Estrategista.

    Usa uma LLM via Groq para propor variantes do ataque Masquerade
    a partir do JSON atual do ataque e das métricas obtidas pelo IDS.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
    ) -> None:
        self.agent = Agent(
            model=Groq(
                id=model_id,
                temperature=temperature,
            ),
            markdown=False,
        )

    def build_prompt(
        self,
        base_prompt: str,
        attack_json: dict[str, Any],
        performance_results: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        return f"""
{base_prompt}

==============================
DADOS ATUAIS DO EXPERIMENTO
==============================

1. JSON atual do ataque Masquerade:

{attack_json}

2. Resultados atuais do IDS Random Forest:

{performance_results}

3. Histórico das iterações anteriores:

{history}

==============================
CONTEXTO DO PIPELINE
==============================

Este experimento é executado em ambiente controlado e sintético.

O fluxo é:

1. O agente recebe o JSON atual do ataque Masquerade.
2. O agente recebe as métricas do IDS baseado em Random Forest.
3. A LLM propõe uma nova configuração do ataque.
4. O orquestrador salva o JSON atualizado.
5. O ERENO gera um novo dataset sintético GOOSE/IEC-61850.
6. O Random Forest, treinado no baseline, testa a nova variante.
7. As métricas retornam para a próxima iteração.

O Random Forest é treinado apenas uma vez no dataset baseline.
Nas iterações seguintes, o modelo fica congelado e apenas testa os datasets variantes.

Portanto, o objetivo não é melhorar o classificador, mas gerar variantes do ataque que reduzam a capacidade de generalização do IDS treinado no baseline.

==============================
IMPORTANTE SOBRE O AVALIADOR
==============================

Algumas colunas foram removidas do treinamento e teste por serem constantes, temporais, sequenciais ou muito associadas ao processo de geração.

Entre elas, quando disponíveis:

- Time
- t
- GooseTimestamp
- receivedTimestamp
- timestampDiff
- tDiff
- timeFromLastChange
- delay
- SqNum
- StNum
- sqDiff
- stDiff
- frameLen
- ethDst
- ethSrc
- ethType
- gooseTimeAllowedtoLive
- gooseAppid
- gooseLen
- TPID
- gocbRef
- datSet
- goID
- test
- confRev
- ndsCom
- numDatSetEntries
- APDUSize
- protocol
- gooseLengthDiff
- apduSizeDiff
- frameLengthDiff
- e2eLatency

Assim, o classificador tende a depender principalmente de grandezas físicas e operacionais, como:

- isbA, isbB, isbC
- vsbA, vsbB, vsbC
- isbARmsValue, isbBRmsValue, isbCRmsValue
- vsbARmsValue, vsbBRmsValue, vsbCRmsValue
- isbATrapAreaSum, isbBTrapAreaSum, isbCTrapAreaSum
- vsbATrapAreaSum, vsbBTrapAreaSum, vsbCTrapAreaSum
- cbStatus
- cbStatusDiff

Use o campo "top_feature_importances" das métricas para identificar quais atributos estão facilitando a detecção.

Se o F1-score estiver alto, isso significa que a variante ainda está muito fácil de detectar.

==============================
OBJETIVO DA ITERAÇÃO
==============================

Com base no JSON atual, nas métricas do IDS e nas features mais importantes, proponha uma nova variante do ataque Masquerade.

O objetivo principal é reduzir:

- f1_score_masquerade

Objetivos secundários:

- reduzir recall_masquerade, para que o IDS deixe de detectar parte dos ataques;
- reduzir precision_masquerade, se possível, aumentando a confusão entre ataque e tráfego normal;
- manter a plausibilidade do cenário GOOSE/IEC-61850;
- manter o tipo de ataque Masquerade.

==============================
ESTRATÉGIA ESPERADA
==============================

Se o F1-score continuar alto, tente aproximar o ataque do comportamento normal nas features mais importantes.

Se as features físicas forem as mais importantes, tente propor alterações graduais que reduzam diferenças abruptas nas grandezas físicas.

Se cbStatus ou cbStatusDiff aparecerem como importantes, tente reduzir mudanças muito evidentes associadas ao estado operacional, sem descaracterizar o ataque.

Se a variante anterior não reduziu o F1-score, tente uma alteração diferente da anterior.

Evite mudanças extremas, inválidas ou sem plausibilidade experimental.

==============================
REGRAS OBRIGATÓRIAS
==============================

- Responda somente com JSON válido.
- Não coloque texto fora do JSON.
- Não use Markdown.
- Não use bloco ```json.
- Não invente chaves novas.
- Mantenha a estrutura original do JSON.
- O campo "updated_attack_json" deve conter o JSON completo do ataque atualizado.
- Não retorne apenas um patch.
- Não altere o tipo do ataque.
- Não descaracterize o cenário GOOSE/IEC-61850.
- Não altere campos que não estejam relacionados aos parâmetros do ataque.
- Não remova campos existentes do JSON original.
- Use alterações graduais e justificáveis.

==============================
FORMATO OBRIGATÓRIO DA RESPOSTA
==============================

A resposta deve seguir exatamente esta estrutura:

{{
  "updated_attack_json": {{
    "cole_aqui_o_json_completo_do_ataque_atualizado": true
  }},
  "summary": "Explique brevemente quais alterações foram feitas.",
  "expected_effect": "Explique o efeito esperado sobre precision, recall e f1-score.",
  "changed_fields": [
    {{
      "field": "nome_do_campo_alterado",
      "previous_value": "valor_anterior",
      "new_value": "novo_valor",
      "reason": "motivo da alteração"
    }}
  ]
}}

Lembre-se: o conteúdo de "updated_attack_json" deve ser o JSON completo do ataque atualizado, mantendo a mesma estrutura do JSON original recebido.
"""

    def suggest_changes(
        self,
        base_prompt: str,
        attack_json: dict[str, Any],
        performance_results: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        prompt = self.build_prompt(
            base_prompt=base_prompt,
            attack_json=attack_json,
            performance_results=performance_results,
            history=history,
        )

        response = self.agent.run(prompt)

        return response.content