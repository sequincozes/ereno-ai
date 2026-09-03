"""Capacidade intent-driven de ``flooding`` (uc07).

Cobertura total dos 8 campos editáveis do baseline. ``gapMs.{min,max}`` é a
alavanca principal do ataque e tem ``polarity="inverse"``: gap **menor** entre
pacotes é o que caracteriza flooding — diminuí-lo é o oposto de evadir.
``sqnumStrideValues`` só carrega ``increase_attack_activity``: escalar a lista
elementwise por um fator ``&lt;1`` (evasão) produz um multiset degenerado
(ex.: ``[1,2,4]`` × 0.5 → ``[1,1,2]``, duplicando o stride 1 em vez de reduzir
o conjunto), então a direção de evasão não é uma alavanca segura para esse
campo.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import DETECTION_AND_ACTIVITY_EFFECTS, DETECTION_EFFECTS
from adversarial_ids.domain.intent_spec import DesiredEffect, IntentObjective

FLOODING_CAPABILITY = AttackCapability(
    capability_id="flooding.v1",
    attack_key="flooding",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="burst.min",
            value_type="integer",
            description="Tamanho mínimo da rajada de flooding.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=2000,
        ),
        FieldCapability(
            path="burst.max",
            value_type="integer",
            description="Tamanho máximo da rajada de flooding.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=2000,
        ),
        FieldCapability(
            path="gapMs.min",
            value_type="number",
            description="Gap mínimo entre pacotes, em milissegundos.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            polarity="inverse",
            minimum=0.01,
            maximum=10.0,
        ),
        FieldCapability(
            path="gapMs.max",
            value_type="number",
            description="Gap máximo entre pacotes, em milissegundos.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            polarity="inverse",
            minimum=0.01,
            maximum=10.0,
        ),
        FieldCapability(
            path="stnumEveryPacket",
            value_type="boolean",
            description="Indica se o stNum avança a cada pacote (assinatura do flooding).",
            effects=DETECTION_EFFECTS,
            choices=(False, True),
        ),
        FieldCapability(
            path="sqnumStrideValues",
            value_type="integer_list",
            description="Conjunto de strides de sqNum usados entre pacotes.",
            effects=frozenset({DesiredEffect.INCREASE_ATTACK_ACTIVITY}),
        ),
        FieldCapability(
            path="ttlMs",
            value_type="integer",
            description="TTL das mensagens de flooding, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=1000,
        ),
        FieldCapability(
            path="ethSrcSpoofProb",
            value_type="number",
            description="Probabilidade de forjar o endereço MAC de origem.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)
