"""Contratos tipados de saída do agente Analista (Blue Team)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Severity = Literal["low", "medium", "high"]
MitigationType = Literal["threshold", "feature", "retrain"]


class DeceptiveFeature(BaseModel):
    """Feature relevante para a decisão do Random Forest."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(min_length=1)
    importance: float = Field(ge=0.0)
    explanation: str = Field(min_length=10)


class Mitigation(BaseModel):
    """Recomendação defensiva produzida pelo Analista."""

    model_config = ConfigDict(extra="forbid")

    type: MitigationType
    recommendation: str = Field(min_length=10)


class AnalystOutput(BaseModel):
    """Saída estruturada do agente Analista."""

    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=0)

    deceptive_features: list[DeceptiveFeature] = Field(
        default_factory=list,
        max_length=5,
    )

    diagnosis: str = Field(min_length=20)

    mitigations: list[Mitigation] = Field(
        min_length=1,
        max_length=5,
    )

    severity: Severity
