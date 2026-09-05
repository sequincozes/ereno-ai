"""orchestrator/intent_live.py — liga o ``IntentLoopOrchestrator`` aos agentes reais.

``IntentLoopOrchestrator`` (intent_loop.py) é agente-agnóstico: fala com
qualquer objeto que satisfaça ``IntentLike``/``DefenderLike``. Este módulo é
a costura que liga os agentes reais (``IntentAgent`` do E1/E2, ``DefenderAgent``
do E5) a esses contratos e expõe ``run_intent_loop`` — o ponto de entrada
público que a CLI liga via ``--engine intent`` (a feature flag da entrega
E3/E4).

Caminho opt-in: exige ``GROQ_API_KEY`` (o ``IntentAgent`` chama a Groq uma
vez por campanha — só na primeira rodada, já que da segunda em diante a
intenção vem da política de feedback do E10; o ``DefenderAgent`` chama uma
vez por rodada) e, em ``generator_mode="jar"``, o JAR do ERENO.
Por isso o import de ``agno`` (via ``agents.intent.agent`` e
``agents.defender.agent``) fica confinado a este módulo — o mesmo motivo que
mantém ``agents/orchestrator/live.py`` separado de ``workflow.py``.
"""

from __future__ import annotations

from adversarial_ids.agents.defender.agent import DefenderAgent
from adversarial_ids.agents.intent.agent import IntentAgent
from adversarial_ids.agents.orchestrator.intent_loop import IntentLoopOrchestrator
from adversarial_ids.config.settings import (
    DETECTOR_MODE,
    INTENT_LOOP_DEFAULT_ROUNDS,
    MODEL_ID,
)
from adversarial_ids.domain.loop_record import LoopRecord


def build_intent_loop(
    *,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
    detector: str = DETECTOR_MODE,
) -> IntentLoopOrchestrator:
    """Monta o orquestrador v2 com o ``IntentAgent`` e o ``DefenderAgent`` reais."""

    return IntentLoopOrchestrator(
        intent_agent=IntentAgent(model_id=model_id),
        defender_agent=DefenderAgent(model_id=model_id),
        generator_mode=generator_mode,
        detector=detector,
    )


def run_intent_loop(
    *,
    prompt: str,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
    rounds: int = INTENT_LOOP_DEFAULT_ROUNDS,
    detector: str = DETECTOR_MODE,
) -> tuple[LoopRecord, ...]:
    """Roda a campanha intent-driven ponta a ponta e devolve um ``LoopRecord``
    por rodada.

    Assinatura estável usada por ``interfaces/cli.py`` (``--engine intent``).
    Sem parâmetro ``attack``: o ataque-base vem do ``IntentSpec.base_attack``
    que o ``IntentAgent`` extrai de ``prompt`` na primeira rodada (ver
    ``IntentLoopOrchestrator.run_campaign``). ``rounds=1`` (default) resolve
    uma única rodada — o mesmo comportamento de antes do E10.

    ``detector`` (E8) escolhe qual modelo o estágio DETECTOR treina; o default
    ``random_forest`` mantém o comportamento anterior. Ver ``docs/detectors.md``.
    """

    orchestrator = build_intent_loop(
        model_id=model_id, generator_mode=generator_mode, detector=detector
    )
    return orchestrator.run_campaign(prompt, rounds=rounds)
