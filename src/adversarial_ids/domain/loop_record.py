"""LoopRecord — registro append-only de uma execução ponta a ponta.

Contrato congelado (ação 72h #2). Amarra todos os IDs, o prompt de origem, o
status e a duração de cada estágio do pipeline INTENT→...→FEEDBACK, mais
custo/duração agregados — a base da telemetria de ``run_id`` e da
reprodutibilidade por seed exigida no roadmap (D6-14 em diante). Distinto do
``IterationRecord`` legado (loop Strategist→Analyst existente): este contrato
cobre o pipeline intent-driven novo, com um estágio por etapa da arquitetura
alvo, não uma única iteração de ajuste de ataque.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _now_iso() -> str:
    """Timestamp ISO-8601 em UTC (string, compatível com json_io.save_json)."""

    return datetime.now(timezone.utc).isoformat()


class LoopStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class LoopStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal[
        "intent",
        "generator",
        "ereno",
        "preprocess",
        "detector",
        "defender",
        "feedback",
    ]
    status: LoopStageStatus
    artifact_ref: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    error: str | None = None


class LoopRecord(BaseModel):
    """Registro de uma execução completa do pipeline, um item por estágio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    source_prompt: str = Field(min_length=1, max_length=4000)
    seed: int = Field(ge=0, le=4_294_967_295)
    stages: tuple[LoopStage, ...] = Field(min_length=1)
    cost_usd: float | None = Field(default=None, ge=0.0)
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    created_at: str = Field(default_factory=_now_iso)
