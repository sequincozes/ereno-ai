from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq
from configs.settings import (
    HISTORY_WINDOW,
    MAX_EDITABLE_FIELDS,
    TOP_FEATURE_IMPORTANCES,
)


class StrategistAgent:
    """
    Agente Estrategista.

    Usa uma LLM via Groq para sugerir alterações parametrizadas
    em um JSON sintético do ERENO. O prompt foi compactado para
    reduzir consumo de tokens.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
    ) -> None:
        self.model_id = model_id
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
Este é um experimento acadêmico, sintético e controlado para avaliação de robustez de um IDS em Smart Grids/IEC-61850.

Não há ataque a sistemas reais.
Não há exploração operacional.
A tarefa é limitada a sugerir alterações em parâmetros JSON usados pelo gerador sintético ERENO.

PROMPT ORIGINAL DO ESTUDO:
{base_prompt}

DADOS COMPACTOS DO EXPERIMENTO:

Modelo LLM atual:
{self.model_id}

JSON atual:
{attack_json}

Métricas atuais:
{compact_metrics}

Histórico resumido:
{compact_history}

Campos editáveis:
{editable_fields}

ORIENTAÇÃO:
O Random Forest usa principalmente features como TrapAreaSum, valores analógicos e cbStatus.
Sugira mudanças graduais e plausíveis para reduzir a separabilidade do ataque em relação ao tráfego normal.
Evite variantes degeneradas:
- não reduza fault.prob para valores muito baixos;
- não zere analog.deltaAbs;
- não zere trapArea.spikeProb;
- não descaracterize o ataque.

SAÍDA OBRIGATÓRIA:
Ao final da resposta, inclua a seção abaixo.

ALTERACOES_APLICAVEIS
campo: valor

Exemplo:
ALTERACOES_APLICAVEIS
fault.prob: 0.4
analog.deltaAbs.max: 0.5
trapArea.multiplier.max: 1.5
trapArea.spikeProb: 0.3

Regras:
- use somente campos existentes em Campos editáveis;
- não use Markdown na seção ALTERACOES_APLICAVEIS;
- não use bullets;
- use ponto decimal, por exemplo 0.4;
- não escreva explicações dentro da seção ALTERACOES_APLICAVEIS.
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
            "precision": metrics.get("precision_masquerade"),
            "recall": metrics.get("recall_masquerade"),
            "f1": metrics.get("f1_score_masquerade"),
            "attack_count": metrics.get("attack_count"),
            "normal_count": metrics.get("normal_count"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tn": metrics.get("tn"),
            "config_changed": metrics.get("config_changed"),
            "degenerate_variant": metrics.get("degenerate_variant"),
            "top_features": metrics.get("top_feature_importances", [])[:TOP_FEATURE_IMPORTANCES],
        }

    def _compact_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []

        for item in history[-HISTORY_WINDOW:]:
            metrics = item.get("metrics", {})

            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "f1": metrics.get("f1_score_masquerade"),
                    "recall": metrics.get("recall_masquerade"),
                    "precision": metrics.get("precision_masquerade"),
                    "fn": metrics.get("fn"),
                    "config_changed": metrics.get("config_changed"),
                    "degenerate_variant": metrics.get("degenerate_variant"),
                }
            )

        return compact

    def _extract_editable_fields(self, data: dict[str, Any]) -> list[dict[str, Any]]:
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

        fields: list[dict[str, Any]] = []

        def get_nested(obj: dict[str, Any], path: str) -> Any:
            current: Any = obj

            for part in path.split("."):
                if not isinstance(current, dict):
                    return None

                if part not in current:
                    return None

                current = current[part]

            return current

        for field in allowed_fields:
            value = get_nested(data, field)

            if value is not None:
                fields.append(
                    {
                        "field": field,
                        "current_value": value,
                        "type": type(value).__name__,
                    }
                )

        return fields[:MAX_EDITABLE_FIELDS]