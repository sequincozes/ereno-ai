"""Capacidade intent-driven de ``random_replay`` (uc01).

Exclui, pela auditoria campo a campo do E0 (13/10/2026,
``scripts/run_e0_sensitivity.py``), os campos que não mudam o trace gerado:

- ``count.lambda`` — o creator nunca lê; a quantidade de mensagens de ataque
  vem do orçamento por segmento da ação de dataset.
- ``ttlOverride.{valuesMs,prob}`` — o TTL é aplicado na mensagem, mas
  ``Goose.copy()`` não o carrega e o creator emite a cópia: o CSV sai sempre
  com o default. Consertar no gerador criaria um artefato (o benigno tem TTL
  constante, e ``frameLen`` soma o TTL como texto).
- ``ethSpoof.{srcProb,dstProb}`` — o creator nunca lê; e trocar o MAC de origem
  de um replay o deixaria menos parecido com o publisher, não mais.

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
    capability_id="random_replay.v2",
    attack_key="random_replay",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
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
    ),
)
