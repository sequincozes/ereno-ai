"""ContextualInjectorAgent — o LLM dos injetores model-driven e data-driven.

Mesma mecânica do ``IntentAgent``: Groq via Agno, uma tool
(``submit_intent_spec``), e a saída passa por ``compile_intent`` — os mesmos
dois portões de qualquer ``IntentSpec``. A diferença está nas instruções (o
contexto que a estratégia lê, ver ``context.py``) e num terceiro portão,
``gate.enforce_injector_output``, que confere se o que voltou é um ataque novo
na DSL do programmable.

Importa ``agno`` no topo, como ``agents/intent/agent.py``: só é carregado
quando ``registry.build_injector`` constrói um injetor real.
"""

from __future__ import annotations

from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq

from adversarial_ids.agents.injector.gate import enforce_injector_output
from adversarial_ids.agents.intent.agent import (
    IntentAgent,
    IntentCompilationError,
    compile_intent,
)
from adversarial_ids.agents.intent.tools import submit_intent_spec
from adversarial_ids.agents.usage import usage_from_response
from adversarial_ids.domain.injector_manifest import InjectorManifest
from adversarial_ids.domain.intent_spec import IntentSpec
from adversarial_ids.domain.run_usage import AgentUsage


class ContextualInjectorAgent:
    """Propõe um ataque novo a partir de um contexto (código ou trace)."""

    def __init__(
        self,
        *,
        model_id: str,
        instructions: str,
        manifest: InjectorManifest,
        extra_artifacts: dict[str, dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id
        self.manifest = manifest
        self.instructions = instructions
        self._extra_artifacts = dict(extra_artifacts or {})
        self.last_usage: AgentUsage | None = None
        self.agent = Agent(
            model=Groq(id=model_id, temperature=temperature),
            tools=[submit_intent_spec],
            instructions=instructions,
            markdown=False,
        )

    def interpret(self, prompt: str) -> IntentSpec:
        response = self.agent.run(prompt)
        # Conta antes dos portões: a chamada foi paga mesmo quando a saída é
        # recusada logo abaixo (mesma regra do IntentAgent, E11).
        self.last_usage = usage_from_response(response)
        tool_result = IntentAgent._extract_tool_result(response)
        if tool_result is None:
            raise IntentCompilationError(
                "A LLM não chamou submit_intent_spec; nenhum ataque foi proposto."
            )
        intent = compile_intent(tool_result, prompt)
        return enforce_injector_output(intent, self.manifest.injector)

    def injector_artifacts(self) -> dict[str, dict[str, Any]]:
        """Arquivos que o orquestrador grava no diretório da rodada."""

        return {
            "injector_manifest.json": self.manifest.model_dump(mode="json"),
            **self._extra_artifacts,
        }
