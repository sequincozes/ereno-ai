"""Schema estrito do ataque ``inverse_replay`` (uc02).

Mesma família estrutural do ``random_replay`` (uc01) — ``count.lambda``,
``burst``, ``ttlOverride`` — mas ``delayMs`` é **inteiro** aqui (``10``/``100``
no baseline) enquanto no uc01 é ponto flutuante (``50.0``/``500.0``). Nenhum
caminho pode ser compartilhado entre os dois modelos por isso.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from adversarial_ids.domain.attack_configs.ranges import (
    FloatRange,
    FrozenConfig,
    IntRange,
    Probability,
)


class CountConfig(FrozenConfig):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lambda_: int = Field(alias="lambda", ge=1)


class BurstConfig(FrozenConfig):
    prob: Probability
    min: int = Field(ge=1)
    max: int = Field(ge=1)
    gapMs: FloatRange

    @model_validator(mode="after")
    def _min_le_max(self) -> "BurstConfig":
        if self.min > self.max:
            raise ValueError(f"burst: min={self.min} não pode ser maior que max={self.max}")
        return self


class TtlOverrideConfig(FrozenConfig):
    valuesMs: list[int] = Field(min_length=1)
    prob: Probability


class InverseReplayConfig(FrozenConfig):
    """Configuração completa do ataque ``inverse_replay`` (uc02)."""

    attackType: Literal["inverse_replay"]
    enabled: bool = True

    count: CountConfig
    blockLen: IntRange
    delayMs: IntRange
    burst: BurstConfig
    ttlOverride: TtlOverrideConfig
