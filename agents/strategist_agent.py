from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq


class StrategistAgent:
    """
    Agente Estrategista.

    Mantém o prompt original usado na análise anterior e acrescenta
    um contexto defensivo, compacto e estruturado para execução automática.
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
        compact_metrics = self._compact_metrics(performance_results)
        compact_history = self._compact_history(history)
        editable_fields = self._extract_editable_fields(attack_json)

        return f"""
CONTEXTO DEFENSIVO E AUTORIZADO:
Este é um experimento acadêmico, sintético e controlado para avaliação de robustez de um Sistema de Detecção de Intrusão em Smart Grids/IEC-61850.

O objetivo NÃO é atacar sistemas reais.
O objetivo NÃO é fornecer instruções operacionais de intrusão.
O objetivo NÃO é ensinar evasão em ambiente real.

O objetivo é apenas propor variações parametrizadas em um arquivo JSON usado pelo gerador sintético ERENO.
Essas variações serão usadas para avaliar, em ambiente local e controlado, como um classificador Random Forest se comporta diante de datasets sintéticos de tráfego GOOSE/IEC-61850.

A saída deve ser limitada exclusivamente aos campos editáveis do JSON fornecido abaixo.

==============================
PROMPT ORIGINAL USADO NO ESTUDO
==============================

{base_prompt}

==============================
DADOS ATUAIS DO EXPERIMENTO
==============================

JSON atual do cenário sintético:
{attack_json}

Métricas atuais do Random Forest:
{compact_metrics}

Histórico resumido da última iteração:
{compact_history}

Campos editáveis disponíveis:
{editable_fields}

==============================
ORIENTAÇÃO PARA A SUGESTÃO
==============================

Considere que o Random Forest tem utilizado principalmente features físicas e operacionais, como TrapAreaSum, valores analógicos e cbStatus.

Sugira alterações graduais e plausíveis nos parâmetros do JSON.
Não remova campos.
Não invente novos campos.
Não altere attackType.
Não proponha ações fora do ambiente sintético.
Não inclua instruções de ataque em sistemas reais.

Evite alterações degeneradas, como:
- reduzir fault.prob para valores muito baixos;
- zerar ou quase zerar analog.deltaAbs;
- zerar ou quase zerar trapArea.spikeProb;
- descaracterizar completamente o ataque.

==============================
SAÍDA PARA EXECUÇÃO AUTOMÁTICA
==============================

Ao final da resposta, inclua obrigatoriamente uma seção chamada ALTERACOES_APLICAVEIS.

Use exatamente este formato, uma alteração por linha:

ALTERACOES_APLICAVEIS
fault.prob: 0.4
fault.durationMs.min: 80
fault.durationMs.max: 1000
analog.deltaAbs.max: 0.5
trapArea.multiplier.max: 1.5
trapArea.spikeProb: 0.3

Regras para a seção ALTERACOES_APLICAVEIS:
- use apenas campos existentes na lista de campos editáveis;
- não coloque explicações nessa seção;
- não use Markdown nessa seção;
- não use bullets nessa seção;
- escreva somente campo: valor;
- use ponto decimal, por exemplo 0.4, não 0,4.
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

        return response.content or ""

    def _compact_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy": metrics.get("accuracy"),
            "precision_masquerade": metrics.get("precision_masquerade"),
            "recall_masquerade": metrics.get("recall_masquerade"),
            "f1_score_masquerade": metrics.get("f1_score_masquerade"),
            "support_masquerade": metrics.get("support_masquerade"),
            "attack_count": metrics.get("attack_count"),
            "normal_count": metrics.get("normal_count"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tn": metrics.get("tn"),
            "config_changed": metrics.get("config_changed"),
            "degenerate_variant": metrics.get("degenerate_variant"),
            "top_feature_importances": metrics.get("top_feature_importances", [])[:5],
        }

    def _compact_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []

        for item in history[-1:]:
            metrics = item.get("metrics", {})

            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_masquerade": metrics.get("precision_masquerade"),
                    "recall_masquerade": metrics.get("recall_masquerade"),
                    "f1_score_masquerade": metrics.get("f1_score_masquerade"),
                    "tp": metrics.get("tp"),
                    "fp": metrics.get("fp"),
                    "fn": metrics.get("fn"),
                    "tn": metrics.get("tn"),
                    "config_changed": metrics.get("config_changed"),
                    "degenerate_variant": metrics.get("degenerate_variant"),
                }
            )

        return compact

    def _extract_editable_fields(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []

        def walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    path = f"{prefix}.{key}" if prefix else key
                    walk(value, path)
            elif isinstance(obj, list):
                fields.append(
                    {
                        "field": prefix,
                        "current_value": obj,
                        "type": "list",
                    }
                )
            else:
                if isinstance(obj, (int, float, bool, str)):
                    fields.append(
                        {
                            "field": prefix,
                            "current_value": obj,
                            "type": type(obj).__name__,
                        }
                    )

        walk(data)

        allowed_fields = [
            "fault.prob",
            "fault.durationMs.min",
            "fault.durationMs.max",
            "cbStatus",
            "incrementStNumOnFault",
            "sqnumMode",
            "ttlMsValues",
            "analog.deltaAbs.min",
            "analog.deltaAbs.max",
            "trapArea.multiplier.min",
            "trapArea.multiplier.max",
            "trapArea.spikeProb",
        ]

        filtered_fields = [
            field for field in fields
            if field["field"] in allowed_fields
        ]

        return filtered_fields