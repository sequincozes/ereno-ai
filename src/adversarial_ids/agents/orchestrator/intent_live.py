"""orchestrator/intent_live.py — liga o ``IntentLoopOrchestrator`` ao IntentAgent real.

``IntentLoopOrchestrator`` (intent_loop.py) é agente-agnóstico: fala com
qualquer objeto que satisfaça ``IntentLike``. Este módulo é a costura que
liga o agente real (``IntentAgent``, épico E1/E2) a esse contrato e expõe
``run_intent_loop`` — o ponto de entrada público que a CLI liga via
``--engine intent`` (a feature flag da entrega E3/E4).

Caminho opt-in: exige ``GROQ_API_KEY`` (o ``IntentAgent`` chama a Groq) e, em
``generator_mode="jar"``, o JAR do ERENO. Por isso o import de ``agno`` (via
``agents.intent.agent``) fica confinado a este módulo — o mesmo motivo que
mantém ``agents/orchestrator/live.py`` separado de ``workflow.py``.
"""

from __future__ import annotations

from adversarial_ids.agents.intent.agent import IntentAgent
from adversarial_ids.agents.orchestrator.intent_loop import IntentLoopOrchestrator
from adversarial_ids.config.settings import MODEL_ID
from adversarial_ids.domain.loop_record import LoopRecord


def build_intent_loop(
    *,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
) -> IntentLoopOrchestrator:
    """Monta o orquestrador v2 com o ``IntentAgent`` real."""

    return IntentLoopOrchestrator(
        intent_agent=IntentAgent(model_id=model_id),
        generator_mode=generator_mode,
    )


def run_intent_loop(
    *,
    prompt: str,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
) -> LoopRecord:
    """Roda o pipeline intent-driven ponta a ponta e devolve o ``LoopRecord``.

    Assinatura estável usada por ``interfaces/cli.py`` (``--engine intent``).
    Sem parâmetro ``attack``: o ataque-base vem do ``IntentSpec.base_attack``
    que o ``IntentAgent`` extrai de ``prompt`` (ver ``IntentLoopOrchestrator.run``).
    """

    orchestrator = build_intent_loop(model_id=model_id, generator_mode=generator_mode)
    return orchestrator.run(prompt)
