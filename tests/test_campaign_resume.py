"""Testes da retomada de campanha (critério de aceite da camada Loop).

"E2E de 3 iterações + retomada — retomada não duplica registros." Cobre os três
jeitos de uma campanha parar sem ter acabado (estágio que falhou, orçamento de
tokens, processo morto) e o único jeito de ela acabar de verdade (a política de
feedback encerra), que a retomada precisa respeitar sem gravar nada.

Mesmos stubs e o mesmo seed cacheado de ``test_intent_loop_orchestrator.py`` —
sem Groq e sem Java.
"""

from __future__ import annotations

import json
from io import StringIO

import pytest

from adversarial_ids.agents.orchestrator.campaign_resume import (
    CampaignResumeError,
    plan_resume,
)
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.domain.run_usage import AgentUsage
from adversarial_ids.interfaces.cli import run_cli
from adversarial_ids.shared.json_io import load_json
from adversarial_ids.shared.loop_record_store import (
    append_loop_record,
    load_loop_records,
)
from tests.test_intent_loop_orchestrator import (
    _FailingDefenderAgent,
    _FailingIntentAgent,
    _StubIntentAgent,
    _intent,
    _orchestrator,
)

PROMPT = "Reduza o recall."


class _CountingIntentAgent(_StubIntentAgent):
    """Informa consumo, para que o teto de tokens (E11) corte a campanha."""

    def __init__(self, intent, usage: AgentUsage) -> None:
        super().__init__(intent)
        self.last_usage = usage


def _ledger(tmp_path) -> list[LoopRecord]:
    return load_loop_records(tmp_path / "loop_records.json")


def _run_ids(tmp_path) -> list[str]:
    return [record.run_id for record in _ledger(tmp_path)]


def _outcomes(tmp_path, records) -> list[tuple[int, str, str]]:
    """(rodada, motivo de parada, intensidade) — o que a campanha decidiu."""

    rows = []
    for record in records:
        run_dir = tmp_path / "artifacts" / record.run_id
        feedback = load_json(run_dir / "feedback.json")
        intent = load_json(run_dir / "intent.json")
        rows.append((record.round, feedback["stop_reason"], intent["intensity"]))
    return rows


# --------------------------------------------------------------------------- #
# Uma rodada que falhou é refeita, sem apagar a que falhou                     #
# --------------------------------------------------------------------------- #
def test_a_round_that_failed_in_the_defender_is_retried_as_a_new_record(tmp_path):
    broken = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), defender_agent=_FailingDefenderAgent()
    )
    (failed,) = broken.run_campaign(PROMPT, rounds=1)
    assert failed.stages[-1].status is LoopStageStatus.FAILED

    agent = _StubIntentAgent(_intent())
    (retried,) = _orchestrator(tmp_path, agent).resume_campaign(failed.run_id)

    assert retried.run_id != failed.run_id
    assert retried.round == 1
    assert retried.parent_run_id is None
    assert retried.retry_of == failed.run_id
    assert retried.stages[-1].name == "feedback"
    assert retried.stages[-1].status is LoopStageStatus.SUCCEEDED

    # O registro que falhou continua no ledger: append-only, e ele é a verdade
    # sobre o que aconteceu na primeira tentativa.
    assert _run_ids(tmp_path) == [failed.run_id, retried.run_id]


def test_a_retry_reuses_the_validated_intent_instead_of_calling_the_llm(tmp_path):
    broken = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), defender_agent=_FailingDefenderAgent()
    )
    (failed,) = broken.run_campaign(PROMPT, rounds=1)

    agent = _StubIntentAgent(_intent())
    (retried,) = _orchestrator(tmp_path, agent).resume_campaign(failed.run_id)

    # Refazer uma rodada que caiu no DEFENDER não gasta de novo o estágio INTENT.
    assert agent.received_prompts == []
    intent_stage = retried.stages[0]
    assert intent_stage.status is LoopStageStatus.SKIPPED
    assert "reaproveitada" in intent_stage.error
    assert failed.run_id in intent_stage.error


def test_a_round_one_that_failed_in_the_intent_stage_calls_the_llm_again(tmp_path):
    (failed,) = _orchestrator(tmp_path, _FailingIntentAgent()).run_campaign(
        PROMPT, rounds=1
    )

    agent = _StubIntentAgent(_intent())
    (retried,) = _orchestrator(tmp_path, agent).resume_campaign(failed.run_id)

    # Não havia intenção validada nenhuma a reaproveitar.
    assert agent.received_prompts == [PROMPT]
    assert retried.stages[0].status is LoopStageStatus.SUCCEEDED
    assert retried.retry_of == failed.run_id


def test_a_failed_middle_round_is_retried_with_the_same_parent(tmp_path):
    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    cut = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
    cut.token_budget = 500
    (first,) = cut.run_campaign(PROMPT, rounds=3)
    # Rodada 2 à mão, falhando no DEFENDER: é o que uma campanha de 3 rodadas
    # deixa no ledger quando a Groq recusa a segunda chamada do Defender.
    plan = plan_resume(_ledger(tmp_path), first.run_id)
    assert plan.action == "continue"
    broken = _orchestrator(
        tmp_path, agent, defender_agent=_FailingDefenderAgent(), feedback_min_delta=0.0
    )
    failed_round_two, _, _ = broken._run_round(
        PROMPT,
        seed=first.seed,
        intent_override=plan.intent,
        round_index=2,
        parent_run_id=first.run_id,
        max_rounds=3,
        history=plan.history,
    )

    resumed = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).resume_campaign(
        failed_round_two.run_id
    )

    assert [r.round for r in resumed] == [2, 3]
    assert resumed[0].parent_run_id == first.run_id
    assert resumed[0].retry_of == failed_round_two.run_id
    assert resumed[1].parent_run_id == resumed[0].run_id
    assert resumed[1].retry_of is None
    # A rodada 3 é comum: herda da 2 pela política, não "reaproveita".
    assert "herdada" in resumed[1].stages[0].error


# --------------------------------------------------------------------------- #
# Uma campanha cortada pelo orçamento continua de onde parou                  #
# --------------------------------------------------------------------------- #
def test_a_campaign_cut_by_the_token_budget_continues_on_resume(tmp_path):
    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))

    def orchestrator():
        built = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
        built.token_budget = 500
        return built

    # Só a rodada 1 chama o LLM (600 tokens) e estoura o teto de 500 — a campanha
    # para nela. As rodadas 2+ reusam a intenção (custo de intent zero), então não
    # somam ao gasto conhecido: na retomada elas completam a campanha.
    (first,) = orchestrator().run_campaign(PROMPT, rounds=3)
    assert first.total_tokens == 600

    resumed = orchestrator().resume_campaign(first.run_id)

    assert [r.round for r in (first, *resumed)] == [1, 2, 3]
    assert resumed[0].parent_run_id == first.run_id
    assert resumed[1].parent_run_id == resumed[0].run_id
    assert {r.max_rounds for r in (first, *resumed)} == {3}
    # A intenção reusada não recontabiliza os 600 tokens da rodada 1: as rodadas
    # 2+ não têm consumo de intent (só o defender stub, que não informa nada).
    assert all(r.total_tokens is None for r in resumed)
    # O LLM foi chamado uma única vez na campanha inteira.
    assert agent.received_prompts == [PROMPT]


def test_a_resumed_campaign_decides_exactly_like_an_uninterrupted_one(tmp_path):
    """O ``history`` reconstruído do ledger é o mesmo que a campanha teria em memória."""

    straight_dir = tmp_path / "straight"
    straight_dir.mkdir()
    straight = _orchestrator(
        straight_dir, _StubIntentAgent(_intent()), feedback_min_delta=0.0
    ).run_campaign(PROMPT, rounds=3)

    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    cut = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
    cut.token_budget = 500
    (first,) = cut.run_campaign(PROMPT, rounds=3)
    rest = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).resume_campaign(
        first.run_id
    )

    assert _outcomes(tmp_path, (first, *rest)) == _outcomes(straight_dir, straight)


def test_a_round_lost_with_the_process_is_run_on_resume(tmp_path):
    agent = _StubIntentAgent(_intent())
    records = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).run_campaign(
        PROMPT, rounds=3
    )
    # O processo morreu durante a rodada 2: ela nunca chegou ao ledger.
    ledger = tmp_path / "loop_records.json"
    raw = json.loads(ledger.read_text(encoding="utf-8"))
    raw["loop_records"] = raw["loop_records"][:1]
    ledger.write_text(json.dumps(raw), encoding="utf-8")

    resumed = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).resume_campaign(
        records[0].run_id
    )

    assert [r.round for r in resumed] == [2, 3]
    assert resumed[0].parent_run_id == records[0].run_id


# --------------------------------------------------------------------------- #
# Retomada não duplica registros                                              #
# --------------------------------------------------------------------------- #
def test_resuming_a_campaign_the_policy_ended_records_nothing(tmp_path):
    agent = _StubIntentAgent(_intent())
    # Modo cacheado: a métrica não se move e a política encerra na rodada 2.
    records = _orchestrator(tmp_path, agent).run_campaign(PROMPT, rounds=5)
    before = _run_ids(tmp_path)

    assert _orchestrator(tmp_path, agent).resume_campaign(records[-1].run_id) == ()
    assert _run_ids(tmp_path) == before

    plan = plan_resume(_ledger(tmp_path), records[-1].run_id)
    assert plan.action == "done"
    assert "no_improvement" in plan.reason


def test_resuming_from_a_middle_round_does_not_fork_the_campaign(tmp_path):
    agent = _StubIntentAgent(_intent())
    records = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).run_campaign(
        PROMPT, rounds=3
    )
    before = _run_ids(tmp_path)

    # A rodada 1 tem filho; retomar dela criaria um segundo ramo da campanha.
    assert _orchestrator(tmp_path, agent).resume_campaign(records[0].run_id) == ()
    assert _run_ids(tmp_path) == before


def test_resuming_twice_only_runs_the_missing_round_once(tmp_path):
    broken = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), defender_agent=_FailingDefenderAgent()
    )
    (failed,) = broken.run_campaign(PROMPT, rounds=1)
    agent = _StubIntentAgent(_intent())

    first_resume = _orchestrator(tmp_path, agent).resume_campaign(failed.run_id)
    second_resume = _orchestrator(tmp_path, agent).resume_campaign(failed.run_id)

    assert len(first_resume) == 1
    assert second_resume == ()
    assert len(_ledger(tmp_path)) == 2


def test_the_ledger_refuses_the_same_run_id_twice(tmp_path):
    path = tmp_path / "loop_records.json"
    record = LoopRecord(
        run_id="run-1",
        source_prompt=PROMPT,
        seed=42,
        stages=(LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),),
    )
    append_loop_record(path, record)

    with pytest.raises(ValueError, match="run-1"):
        append_loop_record(path, record)
    assert len(load_loop_records(path)) == 1


# --------------------------------------------------------------------------- #
# Erros do chamador                                                           #
# --------------------------------------------------------------------------- #
def test_an_unknown_run_id_is_refused(tmp_path):
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()))
    orchestrator.run(PROMPT)

    with pytest.raises(CampaignResumeError, match="não está no ledger"):
        orchestrator.resume_campaign("nao-existe")


def test_resuming_without_a_ledger_is_refused(tmp_path):
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()))
    orchestrator.save_path = None

    with pytest.raises(CampaignResumeError, match="ledger"):
        orchestrator.resume_campaign("qualquer")


def test_resuming_with_another_detector_is_refused(tmp_path):
    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    cut = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
    cut.token_budget = 500
    (first,) = cut.run_campaign(PROMPT, rounds=3)

    other = _orchestrator(tmp_path, agent, detector="decision_tree")
    with pytest.raises(CampaignResumeError, match="random_forest"):
        other.resume_campaign(first.run_id)
    assert _run_ids(tmp_path) == [first.run_id]


def test_a_record_from_before_max_rounds_needs_an_explicit_ceiling(tmp_path):
    path = tmp_path / "loop_records.json"
    legacy = LoopRecord(
        run_id="antigo",
        source_prompt=PROMPT,
        seed=42,
        stages=(LoopStage(name="intent", status=LoopStageStatus.FAILED, error="x"),),
    )
    append_loop_record(path, legacy)

    with pytest.raises(CampaignResumeError, match="max_rounds"):
        plan_resume(load_loop_records(path), "antigo")
    assert plan_resume(load_loop_records(path), "antigo", rounds=2).action == "retry"


def test_a_deleted_artifact_is_reported_instead_of_guessed(tmp_path):
    agent = _StubIntentAgent(_intent())
    records = _orchestrator(tmp_path, agent).run_campaign(PROMPT, rounds=5)
    (tmp_path / "artifacts" / records[-1].run_id / "feedback.json").unlink()

    with pytest.raises(CampaignResumeError, match="não existe mais"):
        plan_resume(_ledger(tmp_path), records[-1].run_id)


def test_the_record_rejects_a_round_past_its_ceiling_or_a_retry_of_itself():
    base = dict(
        source_prompt=PROMPT,
        seed=42,
        stages=(LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),),
    )
    with pytest.raises(ValueError, match="teto"):
        LoopRecord(run_id="r", round=3, parent_run_id="p", max_rounds=2, **base)
    with pytest.raises(ValueError, match="si mesma"):
        LoopRecord(run_id="r", retry_of="r", **base)


# --------------------------------------------------------------------------- #
# CLI: --engine intent --resume RUN_ID                                        #
# --------------------------------------------------------------------------- #
def test_cli_refuses_resume_together_with_a_prompt(tmp_path):
    stderr = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x", "--resume", "r"],
        stderr=stderr,
        loop_records_path=tmp_path / "loop_records.json",
    )

    assert exit_code == 2
    assert "exclusivos" in stderr.getvalue()


def test_cli_says_there_is_nothing_to_resume_without_building_the_agents(tmp_path):
    agent = _StubIntentAgent(_intent())
    records = _orchestrator(tmp_path, agent).run_campaign(PROMPT, rounds=5)

    def must_not_run(**_kwargs):
        raise AssertionError("uma campanha encerrada não monta agente nenhum")

    stdout = StringIO()
    exit_code = run_cli(
        argv=["--engine", "intent", "--resume", records[-1].run_id],
        stdout=stdout,
        resume_intent_loop=must_not_run,
        loop_records_path=tmp_path / "loop_records.json",
    )

    assert exit_code == 0
    assert "Nada a retomar" in stdout.getvalue()


def test_cli_forwards_the_resume_and_keeps_the_campaigns_detector(tmp_path):
    broken = _orchestrator(
        tmp_path,
        _StubIntentAgent(_intent()),
        defender_agent=_FailingDefenderAgent(),
        detector="decision_tree",
    )
    (failed,) = broken.run_campaign(PROMPT, rounds=1)
    received = {}

    def fake_resume(**kwargs):
        received.update(kwargs)
        return _orchestrator(
            tmp_path, _StubIntentAgent(_intent()), detector=kwargs["detector"]
        ).resume_campaign(kwargs["run_id"], rounds=kwargs["rounds"])

    stdout = StringIO()
    exit_code = run_cli(
        argv=["--engine", "intent", "--resume", failed.run_id],
        stdout=stdout,
        resume_intent_loop=fake_resume,
        loop_records_path=tmp_path / "loop_records.json",
    )

    assert exit_code == 0
    assert received["run_id"] == failed.run_id
    assert received["rounds"] is None
    # Sem --detector, a retomada segue com o que a campanha treinava.
    assert received["detector"] == "decision_tree"
    output = stdout.getvalue()
    assert "falhou no estágio defender" in output
    assert f"Rodada 1/1, nova tentativa de {failed.run_id}" in output


def test_cli_reports_an_unknown_run_id_as_a_usage_error(tmp_path):
    stderr = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--resume", "nao-existe"],
        stderr=stderr,
        loop_records_path=tmp_path / "loop_records.json",
    )

    assert exit_code == 2
    assert "não está no ledger" in stderr.getvalue()
