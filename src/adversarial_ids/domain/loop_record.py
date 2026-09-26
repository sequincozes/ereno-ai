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
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now_iso() -> str:
    """Timestamp ISO-8601 em UTC (string, compatível com json_io.save_json)."""

    return datetime.now(timezone.utc).isoformat()


class LoopStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


# O nome do estágio é vocabulário compartilhado, não detalhe de um campo: o
# `LoopEvent` (E11) fala dos mesmos sete estágios, e uma segunda cópia da lista
# envelheceria em silêncio no dia em que o pipeline ganhasse uma etapa.
LoopStageName = Literal[
    "intent",
    "generator",
    "ereno",
    "preprocess",
    "detector",
    "defender",
    "feedback",
]
LOOP_STAGE_NAMES: tuple[str, ...] = get_args(LoopStageName)


class LoopStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: LoopStageName
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
    # Tokens consumidos pelas chamadas de LLM da rodada (E11). Aditivo e com
    # default, como `round`/`parent_run_id`: um registro gravado antes do E11
    # carrega normalmente com None. É em token, e não em dólar, que o orçamento
    # operacional se mede — ver `domain/run_usage.py` sobre por que o repo não
    # carrega tabela de preço.
    total_tokens: int | None = Field(default=None, ge=0)
    total_duration_seconds: float | None = Field(default=None, ge=0.0)
    created_at: str = Field(default_factory=_now_iso)

    # Linhagem da campanha multi-rodada (E10). Campos aditivos com default:
    # ``schema_version`` continua 1 porque um registro gravado antes do E10
    # carrega normalmente (round=1, sem pai) e o formato em disco
    # ({"loop_records": [...]}) não muda.
    round: int = Field(default=1, ge=1)
    parent_run_id: str | None = None

    # Retomada da campanha (critério "retomada não duplica registros"). Mesma
    # disciplina aditiva: um registro antigo carrega com os dois em None.
    # ``max_rounds`` é o teto que a campanha pediu — sem ele, retomar uma
    # campanha interrompida por orçamento teria de adivinhar até onde ela ia.
    # ``retry_of`` aponta a rodada que falhou e que esta refaz: as duas têm o
    # mesmo ``round`` e o mesmo pai, e o registro que falhou continua no ledger
    # (append-only), então o elo explícito é o que distingue uma nova tentativa
    # de uma duplicata.
    max_rounds: int | None = Field(default=None, ge=1)
    retry_of: str | None = None

    # Lote de réplicas (Fase 2.R): réplicas do mesmo experimento sob seeds
    # diferentes compartilham este id. Ortogonal à linhagem de campanha — uma
    # réplica é uma rodada repetida sob outra seed, não uma campanha multi-rodada
    # — então convive com round/parent_run_id sem restrição cruzada. Aditivo com
    # default; ``schema_version`` continua 1. O ``seed`` de cada réplica já
    # distingue uma da outra; este campo só as agrupa no ledger.
    replicate_batch_id: str | None = None

    @model_validator(mode="after")
    def _first_round_has_no_parent(self) -> "LoopRecord":
        if (self.round == 1) != (self.parent_run_id is None):
            raise ValueError(
                "A rodada 1 é a única sem parent_run_id "
                f"(round={self.round}, parent_run_id={self.parent_run_id!r})."
            )
        return self

    @model_validator(mode="after")
    def _resume_lineage_is_consistent(self) -> "LoopRecord":
        if self.max_rounds is not None and self.round > self.max_rounds:
            raise ValueError(
                f"A rodada {self.round} passa do teto da campanha "
                f"(max_rounds={self.max_rounds})."
            )
        if self.retry_of is not None and self.retry_of == self.run_id:
            raise ValueError("Uma rodada não pode ser a nova tentativa de si mesma.")
        return self
