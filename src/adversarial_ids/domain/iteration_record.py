from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from adversarial_ids.domain.analyst_output import AnalystOutput
from adversarial_ids.domain.metrics import Metrics
from adversarial_ids.domain.strategist_output import StrategistOutput  # M1 · issue #4


def _now_iso() -> str:
    """Timestamp ISO-8601 em UTC (string, compatível com json_io.save_json)."""

    return datetime.now(timezone.utc).isoformat()


class IterationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(ge=0)
    # A configuração do ataque é dados de schema variável (cada tipo de ataque do
    # ERENO tem seus próprios campos), então guardamos o dict cru em vez de um
    # schema tipado por ataque. Ver ``domain/attack_configs/`` (um schema
    # estrito por ataque registrado, usado pelo compilador intent-driven).
    attack_config: dict[str, Any]
    strategist_output: StrategistOutput | None = None
    metrics: Metrics
    analyst_output: AnalystOutput | None = None
    timestamp: str = Field(default_factory=_now_iso)

    @field_validator("attack_config", mode="before")
    @classmethod
    def _coerce_attack_config(cls, value: Any) -> Any:
        """Aceita tanto ``dict`` quanto qualquer ``BaseModel`` (ex.: ``MasqueradeFaultConfig``)."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        return value
