"""Capacidade intent-driven de ``high_stnum`` (uc06).

Exclui ``sqNumDelta.min`` e ``padBytes.min`` — ambos com baseline ``0``, morto
nas duas direções: ``decrease`` não desce de um piso ``0``, e ``increase`` sem
``maximum`` é multiplicativo (``0 * fator = 0``).

Exclui também, pela auditoria campo a campo do E0 (13/10/2026,
``scripts/run_e0_sensitivity.py``), os campos que não mudam o trace gerado:

- ``count.lambda`` — o creator nunca lê; a quantidade de mensagens de ataque
  vem do orçamento por segmento da ação de dataset.
- ``ttlOverride.{valuesMs,prob}`` — o TTL é aplicado na mensagem, mas
  ``Goose.copy()`` não o carrega: o CSV sai sempre com o default. Consertar no
  gerador criaria um artefato (o benigno tem TTL constante, e ``frameLen``
  soma o TTL como texto).
- ``padBytes.max`` — o creator nunca lê; e bytes de enchimento dentro de uma
  PDU BER fariam um frame malformado, não uma injeção mais furtiva.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import DETECTION_EFFECTS
from adversarial_ids.domain.intent_spec import IntentObjective

HIGH_STNUM_CAPABILITY = AttackCapability(
    capability_id="high_stnum.v2",
    attack_key="high_stnum",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_EFFECTS,
    fields=(
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
    ),
)
