"""orchestrator/campaign_resume.py — de onde uma campanha interrompida recomeça.

Critério de aceite da camada Loop: *"retomada não duplica registros"*. Uma
campanha (``IntentLoopOrchestrator.run_campaign``) pode parar sem ter acabado
por três motivos, e só um deles é uma decisão sobre o experimento:

- **um estágio falhou** (limite de taxa da Groq, timeout do JAR, portão de
  volume) — a última rodada tem um ``LoopStage`` ``failed`` e nenhum FEEDBACK;
- **o orçamento de tokens acabou** (E11) — a última rodada terminou e a política
  mandou continuar, mas o teto operacional cortou antes;
- **o processo morreu** no meio de uma rodada — ela nunca chegou ao ledger, e o
  que sobra é, de novo, uma última rodada que mandou continuar.

A política de feedback (E10) é quem encerra de verdade (``should_continue`` falso),
e esse veredito a retomada nunca desfaz: uma campanha encerrada retoma para
*nada*, que é justamente como ela não duplica registros.

Tudo aqui é leitura: o ledger e os artefatos que ele aponta (``feedback.json``,
``intent.json``, ``detector_manifest.json``). Nenhum agente, nenhum ``agno`` —
a CLI resolve o ponto de retomada antes de montar os agentes, e uma campanha que
não tem nada a retomar não custa uma chave da Groq para descobrir isso.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from adversarial_ids.domain.detector_manifest import DetectorManifest
from adversarial_ids.domain.feedback_decision import FeedbackDecision, RoundOutcome
from adversarial_ids.domain.intent_spec import IntentSpec
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.shared.json_io import load_json

ResumeAction = Literal["retry", "continue", "done"]


class CampaignResumeError(ValueError):
    """A campanha pedida não pode ser retomada — erro do chamador, não de estágio."""


@dataclass(frozen=True)
class ResumePoint:
    """O que a retomada vai rodar, e por quê.

    ``action``:

    - ``retry`` — refaz a rodada ``round_index`` que falhou (``retry_of`` é ela);
    - ``continue`` — roda a rodada ``round_index`` que a política pediu e ninguém
      rodou;
    - ``done`` — nada a rodar; ``reason`` diz por quê.

    ``intent`` é ``None`` só quando a rodada 1 falhou no próprio estágio INTENT:
    aí não existe intenção validada nenhuma a reaproveitar, e o ``IntentLike``
    precisa ser chamado de novo. Em qualquer outro caso a intenção já existe em
    disco e é reaproveitada — refazer uma rodada que caiu no DEFENDER por limite
    de taxa não deveria gastar de novo os ~7k tokens do estágio INTENT.
    """

    action: ResumeAction
    reason: str
    tip: LoopRecord
    round_index: int
    max_rounds: int
    parent_run_id: str | None = None
    retry_of: str | None = None
    intent: IntentSpec | None = None
    intent_source: str | None = None
    history: tuple[RoundOutcome, ...] = ()
    detector: str | None = None


def plan_resume(
    records: Sequence[LoopRecord],
    run_id: str,
    *,
    rounds: int | None = None,
) -> ResumePoint:
    """Resolve o ponto de retomada da campanha a que ``run_id`` pertence.

    ``run_id`` pode ser qualquer rodada da campanha, não só a última: a ponta é
    encontrada pelo ledger. Retomar a partir de uma rodada do meio criaria um
    segundo ramo da mesma campanha — exatamente a duplicata que o critério proíbe.

    ``rounds`` sobrescreve o teto gravado em ``LoopRecord.max_rounds``; é
    obrigatório só para registros gravados antes de o campo existir.
    """

    if rounds is not None and rounds < 1:
        raise CampaignResumeError(f"rounds precisa ser >= 1 (veio {rounds}).")

    campaign = _campaign_of(records, run_id)
    order = {record.run_id: index for index, record in enumerate(records)}
    tip = max(campaign, key=lambda record: (record.round, order[record.run_id]))
    by_id = {record.run_id: record for record in campaign}

    max_rounds = rounds if rounds is not None else tip.max_rounds
    if max_rounds is None:
        raise CampaignResumeError(
            f"A campanha de {run_id!r} foi gravada antes de LoopRecord.max_rounds "
            "existir; informe o teto de rodadas explicitamente."
        )

    detector = _campaign_detector(campaign, order)
    failed = _failed_stage(tip)

    if failed is not None:
        history = _history_until(tip.parent_run_id, by_id)
        base = dict(
            tip=tip,
            round_index=tip.round,
            max_rounds=max_rounds,
            parent_run_id=tip.parent_run_id,
            history=history,
            detector=detector,
        )
        if tip.round > max_rounds:
            return ResumePoint(
                action="done",
                reason=(
                    f"A rodada {tip.round} falhou, mas passa do teto pedido "
                    f"({max_rounds})."
                ),
                **base,
            )
        intent, source = _intent_for_retry(tip, by_id)
        return ResumePoint(
            action="retry",
            reason=(
                f"A rodada {tip.round} ({tip.run_id}) falhou no estágio "
                f"{failed.name}: {failed.error}"
            ),
            retry_of=tip.run_id,
            intent=intent,
            intent_source=source,
            **base,
        )

    decision = _decision_of(tip)
    history = _history_until(tip.run_id, by_id)
    base = dict(tip=tip, max_rounds=max_rounds, history=history, detector=detector)

    if not decision.should_continue:
        return ResumePoint(
            action="done",
            round_index=tip.round,
            reason=(
                f"A campanha foi encerrada pela política de feedback na rodada "
                f"{tip.round}: {decision.stop_reason.value}. Retomar não desfaz "
                "esse veredito."
            ),
            **base,
        )
    if tip.round >= max_rounds:
        return ResumePoint(
            action="done",
            round_index=tip.round,
            reason=f"A campanha já rodou {tip.round} de {max_rounds} rodadas.",
            **base,
        )
    return ResumePoint(
        action="continue",
        round_index=tip.round + 1,
        reason=(
            f"A rodada {tip.round} ({tip.run_id}) mandou continuar e a rodada "
            f"{tip.round + 1} nunca chegou ao ledger (orçamento de tokens ou "
            "processo interrompido)."
        ),
        parent_run_id=tip.run_id,
        intent=decision.next_intent,
        intent_source=tip.run_id,
        **base,
    )


# --------------------------------------------------------------------------- #
# A campanha no ledger                                                        #
# --------------------------------------------------------------------------- #
def _campaign_of(records: Sequence[LoopRecord], run_id: str) -> list[LoopRecord]:
    """Todas as rodadas ligadas a ``run_id`` por ``parent_run_id`` ou ``retry_of``.

    Componente conexo, não cadeia: uma nova tentativa é irmã da rodada que
    falhou (mesmo pai), e é só pelo ``retry_of`` que as duas ficam na mesma
    campanha quando a que falhou era a rodada 1 — que não tem pai nenhum.
    """

    by_id = {record.run_id: record for record in records}
    if run_id not in by_id:
        raise CampaignResumeError(f"run_id {run_id!r} não está no ledger.")

    neighbours: dict[str, set[str]] = {key: set() for key in by_id}
    for record in records:
        for linked in (record.parent_run_id, record.retry_of):
            if linked is not None and linked in by_id:
                neighbours[record.run_id].add(linked)
                neighbours[linked].add(record.run_id)

    seen = {run_id}
    pending = [run_id]
    while pending:
        for linked in neighbours[pending.pop()]:
            if linked not in seen:
                seen.add(linked)
                pending.append(linked)

    return [record for record in records if record.run_id in seen]


def _failed_stage(record: LoopRecord) -> LoopStage | None:
    return next(
        (stage for stage in record.stages if stage.status is LoopStageStatus.FAILED),
        None,
    )


def _stage(record: LoopRecord, name: str) -> LoopStage | None:
    return next((stage for stage in record.stages if stage.name == name), None)


def _artifact(record: LoopRecord, stage_name: str) -> Path:
    stage = _stage(record, stage_name)
    if stage is None or stage.artifact_ref is None:
        raise CampaignResumeError(
            f"A rodada {record.round} ({record.run_id}) não registrou o artefato "
            f"do estágio {stage_name}."
        )
    path = Path(stage.artifact_ref)
    if not path.exists():
        raise CampaignResumeError(
            f"O artefato {path} da rodada {record.round} ({record.run_id}) não "
            "existe mais — outputs/ foi limpo depois da execução?"
        )
    return path


def _decision_of(record: LoopRecord) -> FeedbackDecision:
    return FeedbackDecision.model_validate(load_json(_artifact(record, "feedback")))


def _history_until(
    run_id: str | None, by_id: dict[str, LoopRecord]
) -> tuple[RoundOutcome, ...]:
    """O ``history`` que ``decide_feedback`` recebe, reconstruído do ledger.

    Sobe pelos pais a partir de ``run_id`` (inclusive): cada rodada no caminho
    terminou e mandou continuar — senão não teria filho —, então todas têm um
    ``feedback.json`` com o valor-objetivo que a política compara.
    """

    outcomes: list[RoundOutcome] = []
    cursor = run_id
    while cursor is not None:
        record = by_id[cursor]
        outcomes.append(
            RoundOutcome(
                round=record.round,
                run_id=record.run_id,
                objective_value=_decision_of(record).objective_value,
            )
        )
        cursor = record.parent_run_id
    return tuple(reversed(outcomes))


def _intent_for_retry(
    tip: LoopRecord, by_id: dict[str, LoopRecord]
) -> tuple[IntentSpec | None, str | None]:
    """A intenção da rodada que falhou, se ela chegou a existir.

    Da rodada 2 em diante a intenção é a ``next_intent`` do pai — a mesma que a
    rodada que falhou gravou em ``intent.json``. Na rodada 1 ela só existe se o
    estágio INTENT passou; se foi ele que falhou, devolve ``None`` e o LLM roda
    de novo.
    """

    stage = _stage(tip, "intent")
    if stage is not None and stage.status in (
        LoopStageStatus.SUCCEEDED,
        LoopStageStatus.SKIPPED,
    ):
        return IntentSpec.model_validate(load_json(_artifact(tip, "intent"))), tip.run_id
    if tip.parent_run_id is not None:
        parent = by_id[tip.parent_run_id]
        return _decision_of(parent).next_intent, parent.run_id
    return None, None


def _campaign_detector(
    campaign: Sequence[LoopRecord], order: dict[str, int]
) -> str | None:
    """O detector que a campanha treinou, lido do manifest da rodada mais recente.

    Retomar com outro detector juntaria dois experimentos num ledger só, e a
    comparação entre rodadas deixaria de medir o ataque. ``None`` quando nenhuma
    rodada chegou a treinar (ou o manifest não está mais em disco).
    """

    for record in sorted(campaign, key=lambda item: -order[item.run_id]):
        stage = _stage(record, "detector")
        if stage is None or stage.status is not LoopStageStatus.SUCCEEDED:
            continue
        if stage.artifact_ref is None:
            continue
        manifest = Path(stage.artifact_ref).parent / "detector_manifest.json"
        if manifest.exists():
            return DetectorManifest.model_validate(load_json(manifest)).detector
    return None
