"""Schema estrito do ataque ``random_replay`` (uc01)."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from adversarial_ids.domain.attack_configs.ranges import FloatRange, FrozenConfig, Probability


class CountConfig(FrozenConfig):
    # ``lambda`` é palavra reservada em Python — o atributo Python é
    # ``lambda_``, mas o JSON (e o dump por alias) continuam usando ``lambda``.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lambda_: int = Field(alias="lambda", ge=1)


class BurstConfig(FrozenConfig):
    """``burst`` mistura o par ``{min, max}`` (tamanho da rajada) com ``prob``
    e ``gapMs`` no mesmo objeto — por isso não é um ``_OrderedRange`` puro."""

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


class EthSpoofConfig(FrozenConfig):
    srcProb: Probability
    dstProb: Probability


class RandomReplayConfig(FrozenConfig):
    """Configuração completa do ataque ``random_replay`` (uc01)."""

    attackType: Literal["random_replay"]
    enabled: bool = True

    count: CountConfig
    windowS: FloatRange
    delayMs: FloatRange
    burst: BurstConfig
    reorderProb: Probability
    ttlOverride: TtlOverrideConfig
    ethSpoof: EthSpoofConfig
