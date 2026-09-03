"""Capacidades intent-driven da família ``delayed_replay`` (uc10 — 4 variantes).

Quatro ``capability_id`` distintos (``delayed_replay.v1``,
``delayed_replay_backoff.v1``, ``delayed_replay_batch_dump.v1``,
``delayed_replay_double_drop.v1``) — mesmo ``delayed_replay_double_drop`` tendo
baseline campo-a-campo idêntico ao ``delayed_replay`` base. Compartilhar um id
faria ``AttackCapability.attack_key`` mentir para três das quatro specs.

Exclui ``orderBy`` nas quatro variantes: string sem enum documentado em
nenhum lugar do repo, e ``FieldCapability._string_fields_need_choices``
rejeitaria a declaração sem ``choices`` — teria que se recuperar via
arqueologia no bytecode do JAR, como foi feito para ``sqnumMode``.

``burstInterval.{min,max}`` tem ``polarity="inverse"``: um intervalo menor
entre rajadas é o que torna o replay atrasado mais agressivo.
``selectionProb.value`` é uma probabilidade cuja chave-folha é ``value`` — o
clamp genérico legado não a alcança.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import (
    DETECTION_AND_ACTIVITY_EFFECTS,
    DETECTION_EFFECTS,
)
from adversarial_ids.domain.intent_spec import IntentObjective

_COMMON_FIELDS = (
    FieldCapability(
        path="burstInterval.min",
        value_type="integer",
        description="Intervalo mínimo entre rajadas de reenvio, em milissegundos.",
        effects=DETECTION_AND_ACTIVITY_EFFECTS,
        polarity="inverse",
        minimum=10,
        maximum=5000,
    ),
    FieldCapability(
        path="burstInterval.max",
        value_type="integer",
        description="Intervalo máximo entre rajadas de reenvio, em milissegundos.",
        effects=DETECTION_AND_ACTIVITY_EFFECTS,
        polarity="inverse",
        minimum=10,
        maximum=5000,
    ),
    FieldCapability(
        path="burstMax",
        value_type="integer",
        description="Tamanho máximo da rajada de reenvio.",
        effects=DETECTION_AND_ACTIVITY_EFFECTS,
        minimum=1,
        maximum=2000,
    ),
    FieldCapability(
        path="selectionProb.value",
        value_type="number",
        description="Probabilidade de selecionar uma mensagem retida para reenvio.",
        effects=DETECTION_AND_ACTIVITY_EFFECTS,
        minimum=0.05,
        maximum=1.0,
    ),
    FieldCapability(
        path="networkDelayMs.min",
        value_type="number",
        description="Atraso de rede mínimo simulado, em milissegundos.",
        effects=DETECTION_EFFECTS,
        minimum=1.0,
        maximum=5000.0,
    ),
    FieldCapability(
        path="networkDelayMs.max",
        value_type="number",
        description="Atraso de rede máximo simulado, em milissegundos.",
        effects=DETECTION_EFFECTS,
        minimum=1.0,
        maximum=5000.0,
    ),
    FieldCapability(
        path="shiftSendTimestamp",
        value_type="boolean",
        description="Indica se o timestamp de envio é deslocado para mascarar o atraso.",
        effects=DETECTION_EFFECTS,
        choices=(False, True),
    ),
    FieldCapability(
        path="replaceWithFake",
        value_type="boolean",
        description="Indica se a mensagem retida é substituída por uma forjada.",
        effects=DETECTION_EFFECTS,
        choices=(False, True),
    ),
)

_OBJECTIVES = frozenset({IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION})

DELAYED_REPLAY_CAPABILITY = AttackCapability(
    capability_id="delayed_replay.v1",
    attack_key="delayed_replay",
    supported_objectives=_OBJECTIVES,
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=_COMMON_FIELDS,
)

DELAYED_REPLAY_DOUBLE_DROP_CAPABILITY = AttackCapability(
    capability_id="delayed_replay_double_drop.v1",
    attack_key="delayed_replay_double_drop",
    supported_objectives=_OBJECTIVES,
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=_COMMON_FIELDS,
)

DELAYED_REPLAY_BACKOFF_CAPABILITY = AttackCapability(
    capability_id="delayed_replay_backoff.v1",
    attack_key="delayed_replay_backoff",
    supported_objectives=_OBJECTIVES,
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=_COMMON_FIELDS
    + (
        FieldCapability(
            path="rateMultiplier",
            value_type="number",
            description="Multiplicador de taxa aplicado no backoff do reenvio.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            # 1.0 é "backoff desligado" — o piso semântico correto, não 0.0.
            minimum=1.0,
            maximum=10.0,
        ),
    ),
)

DELAYED_REPLAY_BATCH_DUMP_CAPABILITY = AttackCapability(
    capability_id="delayed_replay_batch_dump.v1",
    attack_key="delayed_replay_batch_dump",
    supported_objectives=_OBJECTIVES,
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=_COMMON_FIELDS
    + (
        FieldCapability(
            path="microGapMs",
            value_type="number",
            description="Micro-gap entre mensagens despejadas em lote, em milissegundos.",
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            polarity="inverse",
            minimum=0.1,
            maximum=100.0,
        ),
    ),
)
