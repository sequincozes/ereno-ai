"""Capacidade intent-driven de ``random_replay`` (uc01).

Cobertura total dos 15 campos editáveis do baseline — nenhuma exclusão.
``burst.gapMs.{min,max}`` tem ``polarity="inverse"``: um gap **menor** entre
mensagens da rajada é o que torna o replay mais agressivo/detectável, não o
oposto.

Incerteza registrada (não bloqueia o uso, mas vale revisão futura):
``delayMs.{min,max}`` está tagueado ``direct`` (menos atraso ≈ temporização
mais próxima do tráfego normal), mas um replay com atraso quase zero também é
uma duplicata mais óbvia — plausivelmente ``inverse``. Decidir com um
experimento A/B (`--generator-mode jar`, comparar recall) em vez de garantir.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

RANDOM_REPLAY_CAPABILITY = AttackCapability(
    capability_id="random_replay.v1",
    attack_key="random_replay",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="count.lambda",
            value_type="integer",
            description="Taxa média (Poisson) de mensagens reenviadas.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=50,
            maximum=5000,
        ),
        FieldCapability(
            path="windowS.min",
            value_type="number",
            description="Limite inferior da janela de idade das mensagens, em segundos.",
            effects=DETECTION_EFFECTS,
            minimum=0.1,
            maximum=30.0,
        ),
        FieldCapability(
            path="windowS.max",
            value_type="number",
            description="Limite superior da janela de idade das mensagens, em segundos.",
            effects=DETECTION_EFFECTS,
            minimum=0.1,
            maximum=30.0,
        ),
        FieldCapability(
            path="delayMs.min",
            value_type="number",
            description="Atraso mínimo antes do reenvio, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1.0,
            maximum=2000.0,
        ),
        FieldCapability(
            path="delayMs.max",
            value_type="number",
            description="Atraso máximo antes do reenvio, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=1.0,
            maximum=2000.0,
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
            maximum=50,
        ),
        FieldCapability(
            path="burst.max",
            value_type="integer",
            description="Tamanho máximo da rajada de reenvio.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=1,
            maximum=50,
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
            path="reorderProb",
            value_type="number",
            description="Probabilidade de reordenar mensagens reenviadas.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
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
        FieldCapability(
            path="ethSpoof.srcProb",
            value_type="number",
            description="Probabilidade de forjar o endereço MAC de origem.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="ethSpoof.dstProb",
            value_type="number",
            description="Probabilidade de forjar o endereço MAC de destino.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)
