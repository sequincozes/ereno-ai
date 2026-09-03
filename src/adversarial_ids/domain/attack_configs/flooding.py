"""Schema estrito do ataque ``flooding`` (uc07).

``burst`` aqui é só ``{min, max}`` (contagem de pacotes) — sem ``prob``/``gapMs``
como no ``random_replay``/``inverse_replay``. ``gapMs`` é um objeto no
**nível raiz** (não aninhado em ``burst``). ``ttlMs`` é um **inteiro solto**
— no ``injection`` (uc05) é um objeto ``{min, max}``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from adversarial_ids.domain.attack_configs.ranges import FloatRange, FrozenConfig, IntRange, Probability


class FloodingConfig(FrozenConfig):
    """Configuração completa do ataque ``flooding`` (uc07)."""

    attackType: Literal["flooding"]
    enabled: bool = True

    burst: IntRange
    gapMs: FloatRange
    stnumEveryPacket: bool
    sqnumStrideValues: list[int] = Field(min_length=1)
    ttlMs: int = Field(ge=1)
    ethSrcSpoofProb: Probability
