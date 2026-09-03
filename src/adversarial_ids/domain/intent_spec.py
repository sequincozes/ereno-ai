"""Contrato versionado da intenção operacional fornecida pelo usuário."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentObjective(str, Enum):
    """Objetivo de alto nível do experimento adversarial."""

    ASSESS_IDS_ROBUSTNESS = "assess_ids_robustness"
    EVADE_DETECTION = "evade_detection"


class DesiredEffect(str, Enum):
    """Efeito mensurável que o compilador deve tentar produzir."""

    LOWER_F1 = "lower_f1"
    LOWER_RECALL = "lower_recall"
    MIMIC_NORMAL_TRAFFIC = "mimic_normal_traffic"
    INCREASE_ATTACK_ACTIVITY = "increase_attack_activity"
    INCREASE_RESOURCE_PRESSURE = "increase_resource_pressure"


class IntentIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IntentRestrictions(BaseModel):
    """Restrições determinísticas aplicadas ao futuro compilador."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preserve_attack_semantics: bool = True
    max_fields_changed: int = Field(default=3, ge=1, le=12)
    allowed_fields: tuple[str, ...] | None = None
    forbidden_fields: tuple[str, ...] = ()

    @field_validator("allowed_fields", "forbidden_fields")
    @classmethod
    def _normalize_field_paths(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(field.strip() for field in value)
        if any(not field for field in normalized):
            raise ValueError("caminhos de campos não podem ser vazios")
        if len(set(normalized)) != len(normalized):
            raise ValueError("caminhos de campos não podem ser repetidos")
        return normalized

    @model_validator(mode="after")
    def _allowed_and_forbidden_must_not_overlap(self) -> "IntentRestrictions":
        overlap = set(self.allowed_fields or ()) & set(self.forbidden_fields)
        if overlap:
            fields = ", ".join(sorted(overlap))
            raise ValueError(
                f"campos não podem ser permitidos e proibidos ao mesmo tempo: {fields}"
            )
        return self


class IntentSpec(BaseModel):
    """Representa uma intenção já interpretada, antes de gerar configuração.

    O modelo é congelado e rejeita campos desconhecidos para que o artefato JSON
    possa ser persistido e reproduzido sem mutações silenciosas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_prompt: str = Field(min_length=1, max_length=4000)
    objective: IntentObjective
    base_attack: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    desired_effect: DesiredEffect
    restrictions: IntentRestrictions = Field(default_factory=IntentRestrictions)
    intensity: IntentIntensity = IntentIntensity.MEDIUM
    seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @field_validator("source_prompt", "base_attack")
    @classmethod
    def _strip_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("texto não pode ser vazio")
        return normalized
