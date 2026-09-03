"""Primitivas compartilhadas das configurações de ataque do ERENO.

Substituem ``DurationMs``/``DeltaAbs``/``Multiplier`` — três classes quase
idênticas que existiam só para o ``masquerade_fault`` — e servem qualquer
objeto ``{min, max}`` dos 11 baselines em ``inputs/attacks/``.

O tipo (``int`` vs ``float``) é **normativo, não estético**: escolha a classe
pelo literal que está no JSON baseline daquele ataque, nunca por semelhança
com outro. Para o ``masquerade_fault`` o modelo também é o serializador do
golden byte-estável ``data/iteration_history.json``
(``scripts/generate_golden_history.py`` grava ``model_dump(mode="json")`` via
``IterationRecord``) — declarar ``float`` onde o arquivo tem ``50`` viraria
``50.0`` e o golden mudaria. É por isso que ``fault.durationMs`` usa
``IntRange`` e ``uc02.delayMs`` (que o arquivo grava como ``10``/``100``)
também, ainda que ``uc01.delayMs`` (``50.0``/``500.0``) seja ``FloatRange``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    """Base com shape fechado: qualquer chave desconhecida é um erro.

    Congela o contrato — se a LLM ou o gerador introduzirem um campo novo, a
    validação falha em vez de deixar passar silenciosamente.
    """

    model_config = ConfigDict(extra="forbid")


# Probabilidade em [0, 1] — usada nas folhas ``*Prob`` (e em algumas, como
# ``dropRate``/``selectionProb``, cujo nome não denuncia a semântica).
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _OrderedRange(FrozenConfig):
    """Par ``{min, max}`` ordenado. Subclasses fixam o tipo e os limites."""

    min: float
    max: float

    @model_validator(mode="after")
    def _min_le_max(self) -> "_OrderedRange":
        if self.min > self.max:
            raise ValueError(
                f"{type(self).__name__}: min={self.min} não pode ser maior que max={self.max}"
            )
        return self


class IntRange(_OrderedRange):
    """Par inteiro não negativo — durações em ms, contagens, saltos, pad bytes."""

    min: int = Field(ge=0)
    max: int = Field(ge=0)


class FloatRange(_OrderedRange):
    """Par de ponto flutuante não negativo — janelas em s, atrasos, amplitudes."""

    min: float = Field(ge=0.0)
    max: float = Field(ge=0.0)


class ProbabilityRange(FloatRange):
    """Par cujos dois limites são probabilidades (ex.: ``uc08.dropRate``)."""

    min: Probability
    max: Probability


class ProbabilityValue(FrozenConfig):
    """Envelope de chave única ``{"value": p}`` — só o ``uc10.selectionProb`` usa."""

    value: Probability
