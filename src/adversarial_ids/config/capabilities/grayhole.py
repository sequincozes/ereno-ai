"""Capacidade intent-driven de ``grayhole`` (uc08).

Exclui, pela auditoria campo a campo do E0 (13/10/2026,
``scripts/run_e0_sensitivity.py``), tudo que gerava trace idêntico byte a byte:

- ``extraDelayMs.{min,max}``, ``delayBurstProb``, ``delayBurstLen.{min,max}`` —
  o ``GrayHoleVictimCreatorC`` nunca lê esses campos (as mensagens que
  sobrevivem saem sem atraso). Atraso no caminho é plausível, mas modelá-lo é
  decisão do estudo do próprio grayhole, não do catálogo.
- ``protectStatusChanges`` e ``statusChangeDropProb`` — a probabilidade só é
  consultada com o gate ligado (aí as mensagens de mudança de estado caem com
  ``statusChangeDropProb`` em vez de ``dropRate``). O gate fica ``false`` na
  baseline e nunca esteve no catálogo, então a probabilidade era inalcançável.

``dropRate.{min,max}`` é semanticamente uma probabilidade em ``[0, 1]`` cujas
folhas se chamam ``min``/``max`` — o clamp genérico legado (que casa por
substring ``"prob"`` na própria chave) não a alcança; aqui os limites são
explícitos, com um piso ``0.05`` (em vez de ``0.0``) para não deixar o
compilador reduzir a taxa de descarte a um ponto onde o ataque deixa de
existir fisicamente.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

GRAYHOLE_CAPABILITY = AttackCapability(
    capability_id="grayhole.v2",
    attack_key="grayhole",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="dropRate.min",
            value_type="number",
            description="Taxa mínima de descarte de mensagens.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.05,
            maximum=1.0,
        ),
        FieldCapability(
            path="dropRate.max",
            value_type="number",
            description="Taxa máxima de descarte de mensagens.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.05,
            maximum=1.0,
        ),
        FieldCapability(
            path="burstDropProb",
            value_type="number",
            description="Probabilidade de descartar em rajada.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="burstDropLen.min",
            value_type="integer",
            description="Tamanho mínimo da rajada de descarte.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=100,
        ),
        FieldCapability(
            path="burstDropLen.max",
            value_type="integer",
            description="Tamanho máximo da rajada de descarte.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=100,
        ),
    ),
)
