"""Capacidade intent-driven de ``grayhole`` (uc08).

Exclui ``protectStatusChanges`` — um gate booleano de ``statusChangeDropProb``
(com o gate ``true``, mensagens de mudança de estado nunca são descartadas, e
a probabilidade fica inerte). Catalogar os dois permitiria ao compilador
selecionar ambos no mesmo round e ligar o gate justo quando muda a
probabilidade que ele desativa. Mantém-se só ``statusChangeDropProb`` — a
alavanca fisicamente mais forte, já que mensagens de mudança de estado são as
de maior valor para o IDS.

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
    capability_id="grayhole.v1",
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
        FieldCapability(
            path="extraDelayMs.min",
            value_type="number",
            description="Atraso extra mínimo aplicado às mensagens, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1.0,
            maximum=2000.0,
        ),
        FieldCapability(
            path="extraDelayMs.max",
            value_type="number",
            description="Atraso extra máximo aplicado às mensagens, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1.0,
            maximum=2000.0,
        ),
        FieldCapability(
            path="delayBurstProb",
            value_type="number",
            description="Probabilidade de atrasar em rajada.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="delayBurstLen.min",
            value_type="integer",
            description="Tamanho mínimo da rajada de atraso.",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=100,
        ),
        FieldCapability(
            path="delayBurstLen.max",
            value_type="integer",
            description="Tamanho máximo da rajada de atraso.",
            effects=DETECTION_EFFECTS,
            minimum=1,
            maximum=100,
        ),
        FieldCapability(
            path="statusChangeDropProb",
            value_type="number",
            description="Probabilidade de descartar mensagens de mudança de estado.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)
