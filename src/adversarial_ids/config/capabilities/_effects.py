"""Conjuntos de efeitos compartilhados entre os módulos de capacidade por ataque.

``increase_resource_pressure`` deliberadamente não aparece em nenhum conjunto
aqui: nenhum catálogo o anuncia (ver ``core/feedback_policy.py`` —
``DetectionReport`` não tem uma métrica que meça pressão de recurso sem
inventar uma medição arbitrária).
"""

from __future__ import annotations

from adversarial_ids.domain.intent_spec import DesiredEffect

DETECTION_EFFECTS = frozenset(
    {
        DesiredEffect.LOWER_F1,
        DesiredEffect.LOWER_RECALL,
        DesiredEffect.MIMIC_NORMAL_TRAFFIC,
    }
)
DETECTION_AND_ACTIVITY_EFFECTS = DETECTION_EFFECTS | {DesiredEffect.INCREASE_ATTACK_ACTIVITY}
