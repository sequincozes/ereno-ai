"""Schema estrito do ataque ``grayhole`` (uc08).

Baseline sem a chave ``enabled`` (como a ``injection``) e com
``description``/``comment`` no topo. ``dropRate`` é semanticamente uma
probabilidade em ``[0, 1]`` mas suas folhas se chamam ``min``/``max`` — o
clamp genérico legado (``shared/validator.py``, que casa por substring
``"prob"`` na própria chave) não pega esse caso; aqui o tipo é explícito via
``ProbabilityRange``.
"""

from __future__ import annotations

from typing import Literal

from adversarial_ids.domain.attack_configs.ranges import (
    FloatRange,
    FrozenConfig,
    IntRange,
    Probability,
    ProbabilityRange,
)


class GrayholeConfig(FrozenConfig):
    """Configuração completa do ataque ``grayhole`` (uc08)."""

    attackType: Literal["grayhole"]
    description: str
    comment: str

    dropRate: ProbabilityRange
    burstDropProb: Probability
    burstDropLen: IntRange
    extraDelayMs: FloatRange
    delayBurstProb: Probability
    delayBurstLen: IntRange
    protectStatusChanges: bool
    statusChangeDropProb: Probability
