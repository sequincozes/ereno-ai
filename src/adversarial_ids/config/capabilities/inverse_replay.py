"""Capacidade intent-driven de ``inverse_replay`` (uc02).

Cobertura total dos 12 campos editáveis do baseline — nenhuma exclusão.
Mesma família estrutural do ``random_replay``: ``burst.gapMs.{min,max}`` é
``inverse`` pelo mesmo motivo (gap menor = rajada mais apertada e detectável).
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

INVERSE_REPLAY_CAPABILITY = AttackCapability(
    capability_id="inverse_replay.v1",
    attack_key="inverse_replay",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="count.lambda",
            value_type="integer",
            description="Taxa média (Poisson) de mensagens reenviadas em ordem invertida.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=50,
            maximum=5000,
        ),
        FieldCapability(
            path="blockLen.min",
            value_type="integer",
            description="Tamanho mínimo do bloco de mensagens invertido.",
            effects=DETECTION_EFFECTS,
            minimum=2,
            maximum=200,
        ),
        FieldCapability(
            path="blockLen.max",
            value_type="integer",
            description="Tamanho máximo do bloco de mensagens invertido.",
            effects=DETECTION_EFFECTS,
            minimum=2,
            maximum=200,
        ),
        FieldCapability(
            path="delayMs.min",
            value_type="integer",
            description="Atraso mínimo antes do reenvio, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=2000,
        ),
        FieldCapability(
            path="delayMs.max",
            value_type="integer",
            description="Atraso máximo antes do reenvio, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=2000,
        ),
        FieldCapability(
            path="burst.prob",
            value_type="number",
            description="Probabilidade de reenviar em rajada.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="burst.min",
            value_type="integer",
            description="Tamanho mínimo da rajada de reenvio.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=100,
        ),
        FieldCapability(
            path="burst.max",
            value_type="integer",
            description="Tamanho máximo da rajada de reenvio.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=100,
        ),
        FieldCapability(
            path="burst.gapMs.min",
            value_type="number",
            description="Gap mínimo entre mensagens da rajada, em milissegundos.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            polarity="inverse",
            minimum=0.05,
            maximum=1000.0,
        ),
        FieldCapability(
            path="burst.gapMs.max",
            value_type="number",
            description="Gap máximo entre mensagens da rajada, em milissegundos.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            polarity="inverse",
            minimum=0.05,
            maximum=1000.0,
        ),
        FieldCapability(
            path="ttlOverride.valuesMs",
            value_type="integer_list",
            description="Valores de TTL sobrescritos nas mensagens reenviadas.",
            effects=DETECTION_EFFECTS,
        ),
        FieldCapability(
            path="ttlOverride.prob",
            value_type="number",
            description="Probabilidade de sobrescrever o TTL da mensagem reenviada.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)
