"""Testes da contabilidade de consumo e do orçamento da campanha (épico E11)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adversarial_ids.agents.usage import usage_from_response
from adversarial_ids.domain.run_usage import AgentUsage


def _response(**metrics) -> SimpleNamespace:
    return SimpleNamespace(metrics=SimpleNamespace(**metrics))


# --------------------------------------------------------------------------- #
# O contrato                                                                  #
# --------------------------------------------------------------------------- #
def test_total_tokens_is_the_sum_of_both_directions():
    usage = AgentUsage(input_tokens=120, output_tokens=30)

    assert usage.total_tokens == 150
    assert not usage.is_empty


def test_adding_keeps_an_unknown_cost_unknown():
    """Tratar custo ausente como zero produziria um total que parece completo."""

    known = AgentUsage(input_tokens=10, output_tokens=5, cost_usd=0.002)
    unknown = AgentUsage(input_tokens=7, output_tokens=3)

    total = known + unknown

    assert total.input_tokens == 17
    assert total.output_tokens == 8
    assert total.cost_usd == pytest.approx(0.002)


def test_adding_two_unknown_costs_stays_unknown():
    total = AgentUsage(input_tokens=1) + AgentUsage(output_tokens=1)

    assert total.cost_usd is None


def test_cost_without_any_token_does_not_validate():
    """Custo sem token não corresponde a chamada nenhuma que aconteceu."""

    with pytest.raises(ValueError, match="sem token contabilizado"):
        AgentUsage(cost_usd=0.5)


def test_an_empty_usage_says_so():
    assert AgentUsage().is_empty
    assert not AgentUsage(input_tokens=1).is_empty


# --------------------------------------------------------------------------- #
# Leitura da resposta do agente                                               #
# --------------------------------------------------------------------------- #
def test_usage_is_read_from_the_provider_response():
    usage = usage_from_response(
        _response(input_tokens=1200, output_tokens=340, cost=0.0031)
    )

    assert usage is not None
    assert usage.total_tokens == 1540
    assert usage.cost_usd == pytest.approx(0.0031)


@pytest.mark.parametrize(
    "metrics",
    [
        {"prompt_tokens": 10, "completion_tokens": 4},
        {"input_tokens": 10, "output_tokens": 4},
    ],
)
def test_the_same_number_is_found_under_either_provider_name(metrics: dict):
    usage = usage_from_response(_response(**metrics))

    assert usage is not None
    assert usage.total_tokens == 14


def test_a_response_without_metrics_reports_nothing_instead_of_zero():
    """Zero token afirma algo sobre a chamada; None confessa que não se sabe."""

    assert usage_from_response(SimpleNamespace()) is None
    assert usage_from_response(SimpleNamespace(metrics=None)) is None
    assert usage_from_response(_response(coisa_irrelevante="x")) is None


def test_incoherent_metrics_never_take_the_run_down():
    """Contabilidade que derruba a execução que ela mede é pior que ausente."""

    assert usage_from_response(_response(input_tokens=-5)) is None
    assert usage_from_response(_response(input_tokens="muitos")) is None


def test_a_boolean_is_not_mistaken_for_a_token_count():
    assert usage_from_response(_response(input_tokens=True)) is None


# --------------------------------------------------------------------------- #
# Agregação na rodada e o teto da campanha                                    #
# --------------------------------------------------------------------------- #
class _CountingIntentAgent:
    """``IntentLike`` que também informa consumo, como o agente real informa."""

    def __init__(self, intent, usage: AgentUsage) -> None:
        self._intent = intent
        self.last_usage = usage

    def interpret(self, prompt: str):
        return self._intent


def test_the_record_carries_what_the_agents_reported(tmp_path):
    from tests.test_intent_loop_orchestrator import _intent, _orchestrator

    agent = _CountingIntentAgent(
        _intent(), AgentUsage(input_tokens=800, output_tokens=200, cost_usd=0.004)
    )
    record = _orchestrator(tmp_path, agent).run("Reduza o recall.")

    assert record.total_tokens == 1000
    assert record.cost_usd == pytest.approx(0.004)


def test_a_stub_that_reports_nothing_leaves_the_record_honest(tmp_path):
    """Exigir contabilidade do protocolo obrigaria todo stub a inventar número."""

    from tests.test_intent_loop_orchestrator import (
        _StubIntentAgent,
        _intent,
        _orchestrator,
    )

    record = _orchestrator(tmp_path, _StubIntentAgent(_intent())).run("Reduza.")

    assert record.total_tokens is None
    assert record.cost_usd is None


def test_the_campaign_stops_when_the_token_budget_runs_out(tmp_path):
    from tests.test_intent_loop_orchestrator import _intent, _orchestrator

    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    orchestrator = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
    orchestrator.token_budget = 500

    records = orchestrator.run_campaign("Reduza o recall.", rounds=3)

    # A rodada em curso sempre termina: um LoopRecord pela metade não é mais
    # barato, só menos útil. O teto corta a *próxima*.
    assert len(records) == 1
    assert records[0].total_tokens == 600


def test_a_budget_of_zero_never_interrupts(tmp_path):
    from tests.test_intent_loop_orchestrator import _intent, _orchestrator

    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=10_000))
    orchestrator = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)
    orchestrator.token_budget = 0

    assert len(orchestrator.run_campaign("Reduza o recall.", rounds=2)) == 2


def test_a_campaign_whose_agents_report_nothing_is_never_cut_by_the_budget(tmp_path):
    """Comparar um orçamento contra zero inventado aprovaria qualquer coisa —
    e cortar por ele puniria quem simplesmente não tem a informação."""

    from tests.test_intent_loop_orchestrator import (
        _StubIntentAgent,
        _intent,
        _orchestrator,
    )

    orchestrator = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), feedback_min_delta=0.0
    )
    orchestrator.token_budget = 1

    assert len(orchestrator.run_campaign("Reduza o recall.", rounds=2)) == 2


def test_a_negative_budget_is_rejected_at_construction(tmp_path):
    from tests.test_intent_loop_orchestrator import (
        _StubDefenderAgent,
        _StubIntentAgent,
        _intent,
    )
    from adversarial_ids.agents.orchestrator.intent_loop import IntentLoopOrchestrator

    with pytest.raises(ValueError, match="token_budget"):
        IntentLoopOrchestrator(
            intent_agent=_StubIntentAgent(_intent()),
            defender_agent=_StubDefenderAgent(),
            token_budget=-1,
        )
