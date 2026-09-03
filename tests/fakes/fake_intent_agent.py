"""FakeIntentAgent — compila payloads estruturados sem depender da Groq.

Usado pelos golden prompts (ação 72h #4 do plano de 60 dias): simula o que a
tool call ``submit_intent_spec`` produziria a partir de um prompt, e
reaproveita ``compile_intent`` — os mesmos dois portões determinísticos que o
``IntentAgent`` real usa — para validar a cadeia completa sem chamar a API.
"""

from __future__ import annotations

from typing import Any

from adversarial_ids.agents.intent.agent import compile_intent
from adversarial_ids.agents.intent.tools import submit_intent_spec
from adversarial_ids.domain.intent_spec import IntentSpec


class FakeIntentAgent:
    """Stand-in determinístico do IntentAgent para golden prompts/testes."""

    @staticmethod
    def compile(payload: dict[str, Any], source_prompt: str) -> IntentSpec:
        tool_result = submit_intent_spec.entrypoint(
            objective=payload["objective"],
            base_attack=payload["base_attack"],
            desired_effect=payload["desired_effect"],
            intensity=payload.get("intensity", "medium"),
            allowed_fields=payload.get("allowed_fields"),
            forbidden_fields=payload.get("forbidden_fields"),
            max_fields_changed=payload.get("max_fields_changed", 3),
            seed=payload.get("seed", 42),
        )
        return compile_intent(tool_result, source_prompt)
