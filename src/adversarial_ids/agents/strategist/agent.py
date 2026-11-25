from __future__ import annotations

import json
import uuid
from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq

from adversarial_ids.agents.strategist.personas import get_persona
from adversarial_ids.agents.strategist.tools import submit_strategist_output
from adversarial_ids.config.settings import (
    HISTORY_WINDOW,
    MAX_EDITABLE_FIELDS,
    TOP_FEATURE_IMPORTANCES,
)
from adversarial_ids.domain.strategist_output import StrategistOutput
from adversarial_ids.shared.editable_fields import derive_editable_fields

_SESSION_STATE_HISTORY_KEY = "iteration_history"


class StrategistAgent:
    """
    Agente Estrategista.

    Usa uma LLM via Groq para sugerir alterações parametrizadas
    em um JSON sintético do ERENO. A persona controla temperatura,
    agressividade das alterações e instruções. O histórico entre
    iterações é armazenado via session_state do Agno, substituindo
    a injeção textual de ``_compact_history()``.
    """

    def __init__(
        self,
        model_id: str,
        persona: str = "conservative",
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id
        self._persona = get_persona(persona)
        self._session_id = str(uuid.uuid4())

        effective_temperature = (
            temperature if temperature is not None else self._persona.temperature
        )

        self.agent = Agent(
            model=Groq(
                id=model_id,
                temperature=effective_temperature,
            ),
            tools=[submit_strategist_output],
            markdown=False,
        )

    # ------------------------------------------------------------------ #
    # Prompt                                                              #
    # ------------------------------------------------------------------ #

    def build_prompt(
        self,
        base_prompt: str,
        attack_json: dict[str, Any],
        performance_results: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        compact_metrics = self._compact_metrics(performance_results)
        editable_fields = self._extract_editable_fields(attack_json)

        attack_json_str = json.dumps(
            attack_json, ensure_ascii=False, indent=2
        )
        metrics_str = json.dumps(
            compact_metrics, ensure_ascii=False, indent=2
        )
        editable_str = json.dumps(
            editable_fields, ensure_ascii=False, indent=2
        )

        return (
            f"{base_prompt}\n\n"
            f"{self._persona.system_prompt_extra}\n\n"
            "## Dados da iteração atual\n\n"
            f"Modelo LLM atual:\n"
            f"{self.model_id}\n\n"
            "JSON da configuração do ataque:\n"
            f"```json\n{attack_json_str}\n```\n\n"
            "Métricas atuais:\n"
            f"```json\n{metrics_str}\n```\n\n"
            "Campos editáveis disponíveis:\n"
            f"```json\n{editable_str}\n```\n\n"
            "Analise os dados acima e chame a ferramenta "
            "submit_strategist_output para registrar sua proposta.\n"
        )

    # ------------------------------------------------------------------ #
    # Interface pública                                                   #
    # ------------------------------------------------------------------ #

    def suggest_changes(
        self,
        base_prompt: str,
        attack_json: dict[str, Any],
        performance_results: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> StrategistOutput:
        prompt = self.build_prompt(
            base_prompt=base_prompt,
            attack_json=attack_json,
            performance_results=performance_results,
            history=history,
        )

        compact_history = self._compact_history(history)

        response = self.agent.run(
            prompt,
            session_id=self._session_id,
            session_state={_SESSION_STATE_HISTORY_KEY: compact_history},
            add_session_state_to_context=True,
        )

        tool_result = self._extract_tool_result(response)

        if tool_result is not None:
            return StrategistOutput.model_validate_json(tool_result)

        content = response.content or ""
        return self._fallback_parse(content)

    # ------------------------------------------------------------------ #
    # Helpers de extração                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tool_result(response: Any) -> str | None:
        tools = getattr(response, "tools", None) or []

        for tool_exec in tools:
            if getattr(tool_exec, "tool_name", None) == "submit_strategist_output":
                result = getattr(tool_exec, "result", None)
                if result:
                    return result

        return None

    @staticmethod
    def _fallback_parse(content: str) -> StrategistOutput:
        try:
            return StrategistOutput.model_validate_json(content)
        except Exception:
            pass

        try:
            data = json.loads(content)
            return StrategistOutput.model_validate(data)
        except Exception:
            pass

        return StrategistOutput(
            reasoning=content.strip() or "Sem reasoning.",
            persona="conservative",
            changes=[],
        )

    # ------------------------------------------------------------------ #
    # Compactação                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision_attack"),
            "recall": metrics.get("recall_attack"),
            "f1": metrics.get("f1_score_attack"),
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

    @staticmethod
    def _compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []

        for item in history[-HISTORY_WINDOW:]:
            metrics = item.get("metrics", {})

            compact.append(
                {
                    "iteration": item.get("iteration"),
                    "f1": metrics.get("f1_score_attack"),
                    "recall": metrics.get("recall_attack"),
                    "precision": metrics.get("precision_attack"),
                    "fn": metrics.get("fn"),
                    "config_changed": metrics.get("config_changed"),
                    "degenerate_variant": metrics.get("degenerate_variant"),
                }
            )

        return compact

    @staticmethod
    def _extract_editable_fields(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Campos editáveis derivados do próprio JSON do ataque (data-driven).

        Funciona para qualquer tipo de ataque do ERENO — os parâmetros são
        achatados em caminhos ``dot`` a partir da configuração atual, sem
        depender de uma lista fixa por ataque. Ver ``shared.editable_fields``.
        """
        return derive_editable_fields(data, max_fields=MAX_EDITABLE_FIELDS)