"""Agente 3 · Orquestrador — encadeia Estrategista → Analista (M3, #16)."""

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
]
