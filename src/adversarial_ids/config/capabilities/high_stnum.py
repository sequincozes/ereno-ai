"""Capacidade intent-driven de ``high_stnum`` (uc06).

Exclui ``sqNumDelta.min`` e ``padBytes.min`` — ambos com baseline ``0``, morto
nas duas direções: ``decrease`` não desce de um piso ``0``, e ``increase`` sem
``maximum`` é multiplicativo (``0 * fator = 0``). Só os ``.max`` de cada par
entram no catálogo.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

HIGH_STNUM_CAPABILITY = AttackCapability(
    capability_id="high_stnum.v1",
    attack_key="high_stnum",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="count.lambda",
            value_type="integer",
            description="Taxa média (Poisson) de mensagens com stNum elevado.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=20,
            maximum=4000,
        ),
        FieldCapability(
            path="jump.min",
            value_type="integer",
            description="Salto mínimo de stNum acima do valor normal (0-1).",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=1000,
        ),
        FieldCapability(
            path="jump.max",
            value_type="integer",
            description="Salto máximo de stNum acima do valor normal (0-1).",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=1000,
        ),
        FieldCapability(
            path="sqnumResetProb",
            value_type="number",
            description="Probabilidade de reiniciar o sqNum após o salto.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="sqNumDelta.max",
            value_type="integer",
            description="Delta máximo de sqNum aplicado após o salto de stNum.",
            effects=DETECTION_EFFECTS,
            minimum=0,
            maximum=100,
        ),
        FieldCapability(
            path="randomizeTimestamp",
            value_type="boolean",
            description="Indica se o timestamp da mensagem é aleatorizado.",
            effects=DETECTION_EFFECTS,
            choices=(False, True),
        ),
        FieldCapability(
            path="ttlOverride.valuesMs",
            value_type="integer_list",
            description="Valores de TTL sobrescritos nas mensagens injetadas.",
            effects=DETECTION_EFFECTS,
        ),
        FieldCapability(
            path="ttlOverride.prob",
            value_type="number",
            description="Probabilidade de sobrescrever o TTL da mensagem injetada.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="padBytes.max",
            value_type="integer",
            description="Padding máximo (bytes) adicionado ao frame.",
            effects=DETECTION_EFFECTS,
            minimum=0,
            maximum=256,
        ),
    ),
)
