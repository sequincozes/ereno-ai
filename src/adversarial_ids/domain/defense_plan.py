"""DefensePlan — recomendação defensiva rastreável até evidência.

Contrato congelado (ação 72h #2). Endereça o gap P0 "defesa é recomendação
curta" do diagnóstico: toda ação exige evidência (métrica ou feature
concreta) e um método de verificação — nunca uma alegação solta. O contrato
não tem campo de execução: a validação/aceite do plano é explícita
("sem aplicar defesas automaticamente"), então não há como uma
``DefenseAction`` disparar nada sozinha.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_or_feature: str = Field(min_length=1)
    value: float | int | str
    detection_report_ref: str | None = None


class DefenseAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1, max_length=1000)
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    validation_method: str = Field(min_length=1)


class DefensePlan(BaseModel):
    """Plano defensivo fundamentado; nunca aplicado automaticamente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    priority: Literal["low", "medium", "high", "critical"]
    detection_actions: tuple[DefenseAction, ...] = ()
    containment_actions: tuple[DefenseAction, ...] = ()
    hardening_actions: tuple[DefenseAction, ...] = ()

    @model_validator(mode="after")
    def _has_at_least_one_action(self) -> "DefensePlan":
        if not (
            self.detection_actions
            or self.containment_actions
            or self.hardening_actions
        ):
            raise ValueError(
                "DefensePlan precisa de ao menos uma ação de detecção, "
                "contenção ou hardening."
            )
        return self
