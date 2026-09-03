"""Capacidade intent-driven de ``masquerade_fault`` (uc03).

Movido de ``config/attack_capabilities.py`` — conteúdo inalterado, exceto o
``polarity`` implícito (todos os campos são "direct", já era o comportamento
default do compilador antes deste campo existir).
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

MASQUERADE_FAULT_CAPABILITY = AttackCapability(
    capability_id="masquerade_fault.v1",
    attack_key="masquerade_fault",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="fault.prob",
            value_type="number",
            description="Probabilidade de injetar a falha forjada.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="fault.durationMs.min",
            value_type="integer",
            description="Limite inferior da duração da falha, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=0,
        ),
        FieldCapability(
            path="fault.durationMs.max",
            value_type="integer",
            description="Limite superior da duração da falha, em milissegundos.",
            effects=DETECTION_EFFECTS,
            minimum=0,
        ),
        FieldCapability(
            path="cbStatus",
            value_type="integer",
            description="Estado forjado do disjuntor.",
            effects=DETECTION_EFFECTS,
            choices=(0, 1),
        ),
        FieldCapability(
            path="incrementStNumOnFault",
            value_type="boolean",
            description="Indica se stNum avança durante a falha forjada.",
            effects=DETECTION_EFFECTS,
            choices=(False, True),
        ),
        FieldCapability(
            path="sqnumMode",
            value_type="string",
            description="Estratégia de evolução do sqNum durante o ataque.",
            effects=DETECTION_EFFECTS,
            # O gerador ERENO só distingue "fast" (branch equalsIgnoreCase) de
            # qualquer outro valor — confirmado no bytecode do JAR
            # (MasqueradeFakeFaultCreatorC). "normal" é o alternante estável.
            choices=("fast", "normal"),
        ),
        FieldCapability(
            path="ttlMsValues",
            value_type="integer_list",
            description="Valores de TTL usados nas mensagens GOOSE forjadas.",
            effects=DETECTION_EFFECTS,
        ),
        FieldCapability(
            path="analog.deltaAbs.min",
            value_type="number",
            description="Perturbação analógica absoluta mínima.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="analog.deltaAbs.max",
            value_type="number",
            description="Perturbação analógica absoluta máxima.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.multiplier.min",
            value_type="number",
            description="Multiplicador mínimo da área do trapézio.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.multiplier.max",
            value_type="number",
            description="Multiplicador máximo da área do trapézio.",
            effects=DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.spikeProb",
            value_type="number",
            description="Probabilidade de produzir spike na trap area.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)
