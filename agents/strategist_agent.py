from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq


class StrategistAgent:
    """
    Agente Estrategista.

    Mantém o prompt original usado na análise anterior e acrescenta apenas
    um contexto compacto para não estourar o limite da Groq.
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
{base_prompt}

==============================
DADOS COMPACTOS DO EXPERIMENTO
==============================

Métricas atuais do Random Forest:
{compact_metrics}

Histórico resumido:
{compact_history}

Campos editáveis disponíveis no JSON atual:
{editable_fields}

IMPORTANTE:
Use os nomes exatos dos campos acima ao sugerir alterações.
Não é necessário repetir o JSON inteiro.
Sugira somente quais campos alterar, quais valores usar e por quê.
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

    def _compact_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy": metrics.get("accuracy"),
            "precision_masquerade": metrics.get("precision_masquerade"),
            "recall_masquerade": metrics.get("recall_masquerade"),
            "f1_score_masquerade": metrics.get("f1_score_masquerade"),
            "support_masquerade": metrics.get("support_masquerade"),
            "top_feature_importances": metrics.get("top_feature_importances", [])[:5],
        }

    def _compact_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact = []

        for item in history[-1:]:
            metrics = item.get("metrics", {})

            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "f1_score_masquerade": metrics.get("f1_score_masquerade"),
                    "precision_masquerade": metrics.get("precision_masquerade"),
                    "recall_masquerade": metrics.get("recall_masquerade"),
                    "config_changed": metrics.get("config_changed"),
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
                return
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

        return fields[:30]