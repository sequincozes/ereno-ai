"""Agente 3 · Orquestrador — encadeia Estrategista → Analista (M3, #16).

``intent_loop.py`` é o Orchestrator v2 (E3): controlador agente-agnóstico do
pipeline intent-driven novo (INTENT→...→DETECTOR), distinto deste loop
legado Strategist↔Analyst.
"""

from adversarial_ids.agents.orchestrator.intent_loop import (
    IntentLike,
    IntentLoopOrchestrator,
)
from adversarial_ids.agents.orchestrator.workflow import (
    AdversarialWorkflow,
    AnalystLike,
    StrategistLike,
    build_agno_team,
    changes_to_patch,
)

__all__ = [
    "AdversarialWorkflow",
    "StrategistLike",
    "AnalystLike",
    "changes_to_patch",
    "build_agno_team",
    "IntentLoopOrchestrator",
    "IntentLike",
]
