"""Capacidade intent-driven de ``inverse_replay`` (uc02).

Exclui, pela auditoria campo a campo do E0 (13/10/2026,
``scripts/run_e0_sensitivity.py``), os campos que não mudam o trace gerado:

- ``count.lambda`` — o creator nunca lê; a quantidade de mensagens de ataque
  vem do orçamento por segmento da ação de dataset.
- ``ttlOverride.{valuesMs,prob}`` — o TTL é aplicado na mensagem, mas
  ``Goose.copy()`` não o carrega e o creator emite a cópia: o CSV sai sempre
  com o default. Consertar no gerador criaria um artefato (o benigno tem TTL
  constante, e ``frameLen`` soma o TTL como texto), então o campo sai do
  catálogo em vez de o gerador mudar.

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
    capability_id="inverse_replay.v2",
    attack_key="inverse_replay",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
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
    ),
)
