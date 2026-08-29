from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Base com shape fechado: qualquer chave desconhecida é um erro.

    Congela o contrato — se a LLM ou o gerador introduzirem um campo novo, a
    validação falha em vez de deixar passar silenciosamente.
    """

    model_config = ConfigDict(extra="forbid")


class DurationMs(_Frozen):
    """Janela de duração da falha, em milissegundos (inteiros)."""

    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> "DurationMs":
        if self.min > self.max:
            raise ValueError("fault.durationMs.min não pode ser maior que max")
        return self


class FaultConfig(_Frozen):
    prob: float = Field(ge=0.0, le=1.0)
    durationMs: DurationMs


class DeltaAbs(_Frozen):
    """Amplitude absoluta da perturbação no valor analógico."""

    min: float = Field(ge=0.0)
    max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _min_le_max(self) -> "DeltaAbs":
        if self.min > self.max:
            raise ValueError("analog.deltaAbs.min não pode ser maior que max")
        return self


class AnalogConfig(_Frozen):
    deltaAbs: DeltaAbs


class Multiplier(_Frozen):
    """Multiplicador aplicado à área do trapézio (intensidade do spike)."""

    min: float = Field(ge=0.0)
    max: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _min_le_max(self) -> "Multiplier":
        if self.min > self.max:
            raise ValueError("trapArea.multiplier.min não pode ser maior que max")
        return self


class TrapAreaConfig(_Frozen):
    multiplier: Multiplier
    spikeProb: float = Field(ge=0.0, le=1.0)


class AttackConfig(_Frozen):
    """Configuração completa do ataque ``masquerade_fault`` (uc03)."""

    # --- estruturais (não editáveis pela persona) ---
    attackType: str
    enabled: bool = True

    # --- 12 campos editáveis ---
    fault: FaultConfig
    cbStatus: Literal[0, 1]
    incrementStNumOnFault: bool
    sqnumMode: str
    ttlMsValues: list[int] = Field(min_length=1)
    analog: AnalogConfig
    trapArea: TrapAreaConfig
