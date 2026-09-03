"""Schemas estritos da família ``delayed_replay`` (uc10 — 4 variantes).

``delayed_replay`` e ``delayed_replay_double_drop`` têm baselines idênticos
campo a campo, exceto ``attackType`` — a variante "double drop" é dispatch-only,
sem alavanca própria. ``delayed_replay_backoff`` adiciona ``rateMultiplier``;
``delayed_replay_batch_dump`` adiciona ``microGapMs``. Quatro classes
concretas (em vez de uma com campos opcionais) porque ``extra="forbid"`` só
barra ``rateMultiplier`` numa config ``batch_dump`` se as duas variantes forem
schemas distintos.

``selectionProb.value`` é uma probabilidade cuja chave-folha é ``value``, não
algo que contenha "prob" — o clamp genérico legado não a alcança.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from adversarial_ids.domain.attack_configs.ranges import (
    FloatRange,
    FrozenConfig,
    IntRange,
    ProbabilityValue,
)


class _DelayedReplayCommon(FrozenConfig):
    enabled: bool = True

    burstInterval: IntRange
    burstMax: int = Field(ge=1)
    selectionProb: ProbabilityValue
    networkDelayMs: FloatRange
    shiftSendTimestamp: bool
    orderBy: str
    replaceWithFake: bool


class DelayedReplayConfig(_DelayedReplayCommon):
    attackType: Literal["delayed_replay"]


class DelayedReplayDoubleDropConfig(_DelayedReplayCommon):
    attackType: Literal["delayed_replay_double_drop"]


class DelayedReplayBackoffConfig(_DelayedReplayCommon):
    attackType: Literal["delayed_replay_backoff"]
    rateMultiplier: float = Field(gt=0.0)


class DelayedReplayBatchDumpConfig(_DelayedReplayCommon):
    attackType: Literal["delayed_replay_batch_dump"]
    microGapMs: float = Field(ge=0.0)
