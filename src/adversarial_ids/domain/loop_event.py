"""LoopEvent — o que o pipeline está fazendo *agora* (épico E11).

O ``LoopRecord`` é o veredito: existe quando a execução acabou, e conta uma
história completa e imutável. Isso basta para auditoria e não basta para
observabilidade — enquanto a execução roda, o `LoopRecord` ainda não existe, e
quem olha não vê nada.

Era esse buraco que a UI vinha tapando por conta própria:
``interfaces/dashboard/studio/bridge.py::progress()`` deduzia a fase corrente
procurando as substrings ``"=== ITERAÇÃO"``, ``"[GEN:BENIGN] AUSENTE"`` e
``"concluída"`` no ``stdout`` capturado do núcleo. Progresso por engenharia
reversa de texto em português: reescrever uma mensagem de log quebrava a barra
de progresso, e nenhum teste ficava vermelho.

Este contrato substitui aquilo. O orquestrador emite um evento tipado por
transição de estágio, e quem observa — a UI ao vivo, um arquivo JSONL, um teste
— consome evento, nunca texto.

## Ordem é `sequence`, não timestamp

``created_at`` existe para leitura humana e correlação com logs externos.
Ordenar por ele seria frágil: dois estágios rápidos podem cair no mesmo
timestamp, e o relógio pode andar para trás numa sincronização de NTP.
``sequence`` é monotônico dentro da execução e é ele que define a ordem.

## Evento de execução e evento de estágio

Quatro tipos, e a distinção é estrutural, não cosmética: ``run_started`` e
``run_finished`` falam da execução (``stage=None``), ``stage_started`` e
``stage_finished`` falam de um dos sete estágios do pipeline. ``status`` e
``duration_seconds`` só existem no par ``*_finished`` — um evento de início que
afirmasse duração estaria mentindo, e validadores recusam exatamente isso em vez
de deixar a UI decidir se acredita.

O vocabulário de estágio e de status vem de ``loop_record`` (``LoopStageName``,
``LoopStageStatus``): o evento e o registro falam dos mesmos sete estágios, ou a
timeline ao vivo e o histórico contariam histórias diferentes sobre a mesma
execução.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.loop_record import LoopStageName, LoopStageStatus


def _now_iso() -> str:
    """Timestamp ISO-8601 em UTC, como em ``loop_record``/``iteration_record``."""

    return datetime.now(timezone.utc).isoformat()


LoopEventKind = Literal[
    "run_started",
    "stage_started",
    "stage_finished",
    "run_finished",
]
LOOP_EVENT_KINDS: tuple[str, ...] = get_args(LoopEventKind)

# Os dois eixos que os validadores abaixo consultam. Declarados como conjuntos
# em vez de espalhados em `if`s para que a regra seja legível de uma vez: quem
# fala de estágio, e quem fecha alguma coisa.
_STAGE_KINDS: frozenset[str] = frozenset({"stage_started", "stage_finished"})
_FINISHING_KINDS: frozenset[str] = frozenset({"stage_finished", "run_finished"})


class LoopEvent(BaseModel):
    """Uma transição observável do pipeline intent-driven."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    # A rodada da campanha (E10), para uma timeline de `run_campaign` não
    # misturar as rodadas num monte só.
    round: int = Field(default=1, ge=1)
    sequence: int = Field(ge=0)
    kind: LoopEventKind
    stage: LoopStageName | None = None
    status: LoopStageStatus | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    artifact_ref: str | None = None
    message: str | None = Field(default=None, max_length=2000)
    created_at: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _stage_events_name_a_stage(self) -> "LoopEvent":
        """Evento de estágio sem estágio não localiza nada; o inverso mente."""

        names_a_stage = self.kind in _STAGE_KINDS
        if names_a_stage and self.stage is None:
            raise ValueError(f"{self.kind!r} exige o estágio que o produziu.")
        if not names_a_stage and self.stage is not None:
            raise ValueError(
                f"{self.kind!r} é evento de execução e não pode nomear o "
                f"estágio {self.stage!r}."
            )

        return self

    @model_validator(mode="after")
    def _only_a_finished_event_reports_an_outcome(self) -> "LoopEvent":
        """Começar não produz status nem duração — afirmar os dois seria mentir."""

        finishing = self.kind in _FINISHING_KINDS
        if finishing and self.status is None:
            raise ValueError(f"{self.kind!r} exige o status resultante.")
        if not finishing:
            if self.status is not None:
                raise ValueError(
                    f"{self.kind!r} não pode declarar status ({self.status})."
                )
            if self.duration_seconds is not None:
                raise ValueError(
                    f"{self.kind!r} não pode declarar duração "
                    f"({self.duration_seconds})."
                )

        return self

    @property
    def is_terminal(self) -> bool:
        """Verdadeiro no evento que fecha a execução — o fim da timeline."""

        return self.kind == "run_finished"
