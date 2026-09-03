"""FeedbackDecision — saída do estágio FEEDBACK do pipeline intent-driven (E10).

Fecha o loop Red×Blue do pipeline novo: dado o ``DetectionReport`` e o
``DefensePlan`` de uma rodada, o estágio FEEDBACK registra o que foi medido,
decide se a campanha continua e, quando continua, qual é a **próxima
intenção**. A decisão é tomada por política determinística
(``core/feedback_policy.py``) — nenhuma LLM participa deste estágio, o que
mantém o guardrail central do pipeline: o modelo só produz a intenção
(``IntentSpec``) ou o plano de defesa (``DefensePlan``); tudo a jusante é
Python determinístico e auditável.

O contrato não tem campo de execução: a decisão diz qual intenção rodar em
seguida, mas quem roda é o orquestrador (``IntentLoopOrchestrator.run_campaign``)
— um ``FeedbackDecision`` persistido nunca dispara nada sozinho.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.intent_spec import IntentSpec


class FeedbackStopReason(str, Enum):
    """Por que a campanha continua ou para — sempre explícito, nunca implícito."""

    CONTINUE_CAMPAIGN = "continue_campaign"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    OBJECTIVE_REACHED = "objective_reached"
    NO_IMPROVEMENT = "no_improvement"
    LEVERS_EXHAUSTED = "levers_exhausted"
    NO_OBJECTIVE_METRIC = "no_objective_metric"


class RoundOutcome(BaseModel):
    """Resultado mínimo de uma rodada já concluída, para a política comparar.

    Só o necessário para calcular o melhor-até-agora: carregar ``LoopRecord``
    inteiros no cálculo tornaria a política dependente do formato do ledger.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    round: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    objective_value: float | None = None


class FeedbackDecision(BaseModel):
    """Decisão de retroalimentação de uma rodada da campanha adversarial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    should_continue: bool
    stop_reason: FeedbackStopReason

    # Métrica que a intenção pede para minimizar. ``None`` quando o efeito
    # desejado não é de evasão (não existe métrica de evasão a minimizar no
    # DetectionReport — inventar uma seria fingir medição).
    objective_metric: Literal["f1", "recall"] | None = None
    objective_value: float | None = Field(default=None, ge=0.0, le=1.0)

    best_value_so_far: float | None = Field(default=None, ge=0.0, le=1.0)
    best_round: int | None = Field(default=None, ge=1)
    # Positivo = esta rodada melhorou sobre o melhor valor anterior da campanha.
    improvement: float | None = None

    round: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    parent_run_id: str | None = None

    # Prioridade do DefensePlan desta rodada: o elo Blue→ledger. Entra como
    # contexto rastreável, não decide a continuidade da campanha.
    defense_priority: Literal["low", "medium", "high", "critical"]

    next_intent: IntentSpec | None = None
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _continuation_is_consistent(self) -> "FeedbackDecision":
        continues = self.stop_reason is FeedbackStopReason.CONTINUE_CAMPAIGN
        if self.should_continue != continues:
            raise ValueError(
                "should_continue precisa concordar com stop_reason "
                f"({self.should_continue!r} vs {self.stop_reason.value!r})."
            )
        if self.should_continue and self.next_intent is None:
            raise ValueError(
                "Uma campanha que continua precisa da próxima intenção em "
                "next_intent."
            )
        if not self.should_continue and self.next_intent is not None:
            raise ValueError(
                "Uma campanha encerrada não pode carregar next_intent — a "
                f"parada foi {self.stop_reason.value!r}."
            )
        return self

    @model_validator(mode="after")
    def _first_round_has_no_parent(self) -> "FeedbackDecision":
        if (self.round == 1) != (self.parent_run_id is None):
            raise ValueError(
                "A rodada 1 é a única sem parent_run_id "
                f"(round={self.round}, parent_run_id={self.parent_run_id!r})."
            )
        return self
