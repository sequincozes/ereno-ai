"""Schema estrito do ataque ``high_stnum`` (uc06, ``attackType`` = ``high_stnum_injection``)."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from adversarial_ids.domain.attack_configs.ranges import FrozenConfig, IntRange, Probability


class CountConfig(FrozenConfig):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lambda_: int = Field(alias="lambda", ge=1)


class TtlOverrideConfig(FrozenConfig):
    valuesMs: list[int] = Field(min_length=1)
    prob: Probability


class HighStNumInjectionConfig(FrozenConfig):
    """Configuração completa do ataque ``high_stnum`` (uc06)."""

    attackType: Literal["high_stnum_injection"]
    enabled: bool = True
    description: str

    count: CountConfig
    jump: IntRange
    sqnumResetProb: Probability
    sqNumDelta: IntRange
    randomizeTimestamp: bool
    ttlOverride: TtlOverrideConfig
    padBytes: IntRange
