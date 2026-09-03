"""IntentAgent — compila prompts livres em ``IntentSpec`` validadas (épico E2).

Usa uma LLM via Groq para interpretar o pedido do usuário e propor os campos
estruturados de uma intenção via tool calling (``submit_intent_spec``). A LLM
nunca grava ``source_prompt``: o texto original é anexado pelo próprio agente
depois da chamada. A saída passa por dois portões determinísticos antes de
ser aceita — validação do contrato ``IntentSpec`` (Pydantic) e
``validate_intent_capability`` (allowlist de campos por ataque) — mantendo a
LLM fora do caminho de escrita direto no runtime do ERENO, conforme o
guardrail central do plano de 60 dias.
"""

from __future__ import annotations

import json
from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq
from pydantic import ValidationError

from adversarial_ids.agents.intent.tools import submit_intent_spec
from adversarial_ids.config.attack_capabilities import validate_intent_capability
from adversarial_ids.config.settings import PROMPTS_DIR
from adversarial_ids.domain.intent_spec import IntentSpec

_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "intent.md"


class IntentCompilationError(ValueError):
    """Erro acionável: o prompt não produziu uma intenção válida e autorizada."""


def compile_intent(tool_result_json: str, source_prompt: str) -> IntentSpec:
    """Aplica os dois portões determinísticos a uma saída bruta da tool.

    Compartilhado entre ``IntentAgent.interpret`` (via LLM real) e
    ``FakeIntentAgent`` (golden prompts/testes), para que os dois caminhos
    apliquem exatamente a mesma validação — nenhum deles pode devolver uma
    ``IntentSpec`` que não tenha passado por ambos os portões.
    """
    payload = json.loads(tool_result_json)
    if "errors" in payload:
        raise IntentCompilationError(
            "Ferramenta rejeitou os campos propostos: " + "; ".join(payload["errors"])
        )

    payload["source_prompt"] = source_prompt
    try:
        intent = IntentSpec.model_validate(payload)
    except ValidationError as exc:
        raise IntentCompilationError(f"IntentSpec inválida: {exc}") from exc

    try:
        validate_intent_capability(intent)
    except ValueError as exc:
        raise IntentCompilationError(str(exc)) from exc

    return intent


class IntentAgent:
    """Compila um prompt livre em uma ``IntentSpec`` validada e autorizada."""

    def __init__(self, model_id: str, temperature: float = 0.1) -> None:
        self.model_id = model_id
        self.agent = Agent(
            model=Groq(id=model_id, temperature=temperature),
            tools=[submit_intent_spec],
            instructions=_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            markdown=False,
        )

    def interpret(self, prompt: str) -> IntentSpec:
        """Interpreta ``prompt`` e devolve uma ``IntentSpec`` já autorizada.

        Levanta ``IntentCompilationError`` sempre que a saída da LLM não
        chegar a uma intenção válida e dentro da allowlist de capacidades —
        o chamador nunca recebe uma ``IntentSpec`` que não tenha passado por
        ambos os portões.
        """
        response = self.agent.run(prompt)
        tool_result = self._extract_tool_result(response)
        if tool_result is None:
            raise IntentCompilationError(
                "A LLM não chamou submit_intent_spec; nenhuma intenção foi produzida."
            )

        return compile_intent(tool_result, prompt)

    @staticmethod
    def _extract_tool_result(response: Any) -> str | None:
        tools = getattr(response, "tools", None) or []

        for tool_exec in tools:
            if getattr(tool_exec, "tool_name", None) == "submit_intent_spec":
                result = getattr(tool_exec, "result", None)
                if result:
                    return result

        return None
