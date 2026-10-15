"""E1: caixa, propositores determinísticos, ledger retomável e o injetor LLM com stub.

Nada aqui gera trace nem chama a Groq: a avaliação é substituída por um stub,
e o agente do LLM por objetos que imitam a resposta do agno.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from adversarial_ids.agents.attacker.context import (
    DAILY,
    MINUTE,
    TOO_LARGE,
    build_round_message,
    classify_provider_error,
)
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.experiments.metrics import stratified_paired_difference
from adversarial_ids.experiments.search import (
    NO_INTENT,
    VALID,
    OpenLoopLadder,
    ProposalContext,
    ProposerUnavailable,
    RandomSearch,
    SearchBox,
    best_so_far_curve,
    summarize_runs,
    read_ledger,
    run_campaign,
    select_best,
)
from adversarial_ids.experiments.sensitivity import reference_config

FAMILIES = ("masquerade_fault", "random_replay", "inverse_replay", "injection", "high_stnum")


def _box(family: str) -> SearchBox:
    return SearchBox.for_family(family, reference_config(family))


def _context(family: str, index: int, rep: int = 1, history=()) -> ProposalContext:
    return ProposalContext(family, rep, index, 10, _box(family), tuple(history))


# --------------------------------------------------------------------------- #
# Caixa                                                                       #
# --------------------------------------------------------------------------- #
def test_open_side_is_four_times_the_pair_max():
    box = _box("masquerade_fault")
    assert box.field("fault.durationMs.min").high == 4 * 800
    assert box.field("fault.durationMs.max").high == 4 * 800
    assert box.field("fault.prob").high == 1.0  # lado fechado no catálogo fica


def test_physical_bound_overrides_the_factor():
    box = SearchBox.for_family(
        "masquerade_fault", reference_config("masquerade_fault"), open_bounds={"fault.durationMs.max": 2000}
    )
    assert box.field("fault.durationMs.max").high == 2000
    assert box.field("fault.durationMs.min").high == 3200


def test_box_rejects_too_many_fields():
    box = SearchBox.for_family("masquerade_fault", reference_config("masquerade_fault"), max_fields=1)
    candidate = compile_attack_candidate(OpenLoopLadder().propose(_context("masquerade_fault", 3)).intent)
    assert len(candidate.diff) == 2
    assert box.violations(candidate)


# --------------------------------------------------------------------------- #
# B1 e B2                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAMILIES)
def test_random_search_proposals_compile_inside_the_box(family):
    box = _box(family)
    for index in range(20):
        proposal = RandomSearch().propose(_context(family, index))
        candidate = compile_attack_candidate(proposal.intent)
        assert box.violations(candidate) == []


def test_random_search_is_reproducible_per_rep_and_index():
    a = RandomSearch().propose(_context("random_replay", 7, rep=2)).intent
    b = RandomSearch().propose(_context("random_replay", 7, rep=2)).intent
    c = RandomSearch().propose(_context("random_replay", 7, rep=3)).intent
    assert a.restrictions.target_values == b.restrictions.target_values
    assert a.restrictions.target_values != c.restrictions.target_values


def test_ladder_climbs_intensity_then_width_and_exhausts():
    seen = []
    for index in range(10):
        proposal = OpenLoopLadder().propose(_context("injection", index))
        if proposal is None:
            break
        seen.append((proposal.intent.intensity.value, proposal.intent.restrictions.max_fields_changed))
    assert seen == [("low", 1), ("medium", 1), ("high", 1), ("high", 2)]


def test_ladder_stops_at_the_box_width():
    widths = []
    for index in range(10):
        proposal = OpenLoopLadder().propose(_context("masquerade_fault", index))
        if proposal is None:
            break
        widths.append(proposal.intent.restrictions.max_fields_changed)
    assert max(widths) == 4 and len(widths) == 6


# --------------------------------------------------------------------------- #
# Campanha, ledger, seleção                                                   #
# --------------------------------------------------------------------------- #
class _StubEvaluator:
    """Evasão = posição da proposta / 10; a 3 é inválida."""

    def __init__(self, family):
        self.box = _box(family)
        self.calls = 0

    def evaluate(self, intent, error=None):
        self.calls += 1
        if intent is None:
            return {"status": NO_INTENT, "reason": error}
        index = self.calls - 1
        if index == 3:
            return {"status": "oracle_failed", "reason": "stub"}
        return {
            "status": VALID,
            "reason": None,
            "diff": [],
            "config": {},
            "objective": index / 10,
            "search": {"ap": 0.5, "evasion": {"10.0": index / 10}},
        }


def test_campaign_resumes_without_repeating(tmp_path):
    ledger = tmp_path / "rep1.jsonl"
    evaluator = _StubEvaluator("random_replay")
    run_campaign(RandomSearch(), evaluator, family="random_replay", rep=1, ledger_path=ledger, budget=4)
    assert evaluator.calls == 4
    entries = run_campaign(RandomSearch(), evaluator, family="random_replay", rep=1, ledger_path=ledger, budget=6)
    assert evaluator.calls == 6
    assert [e["index"] for e in entries] == list(range(6))


def test_truncated_last_line_is_dropped(tmp_path):
    ledger = tmp_path / "rep1.jsonl"
    ledger.write_text(json.dumps({"index": 0, "status": VALID}) + "\n{\"index\": 1, \"sta", encoding="utf-8")
    assert [e["index"] for e in read_ledger(ledger)] == [0]
    assert read_ledger(ledger) == [{"index": 0, "status": VALID}]


def test_exhausted_ladder_is_recorded_once(tmp_path):
    ledger = tmp_path / "rep1.jsonl"
    evaluator = _StubEvaluator("injection")
    entries = run_campaign(OpenLoopLadder(), evaluator, family="injection", rep=1, ledger_path=ledger)
    assert entries[-1]["status"] == "exhausted" and evaluator.calls == 4
    again = run_campaign(OpenLoopLadder(), evaluator, family="injection", rep=1, ledger_path=ledger)
    assert len(again) == len(entries) and evaluator.calls == 4


def test_selection_ties_break_on_ap_then_age():
    entries = [
        {"index": 0, "status": VALID, "objective": 0.5, "search": {"ap": 0.4}},
        {"index": 1, "status": VALID, "objective": 0.5, "search": {"ap": 0.3}},
        {"index": 2, "status": VALID, "objective": 0.5, "search": {"ap": 0.3}},
        {"index": 3, "status": "oracle_failed"},
    ]
    assert select_best(entries)["index"] == 1
    assert select_best([entries[3]]) is None


def test_curve_carries_the_best_through_invalid_and_exhaustion():
    entries = [
        {"index": 0, "status": "compile_failed"},
        {"index": 1, "status": VALID, "objective": 0.2},
        {"index": 2, "status": VALID, "objective": 0.1},
        {"index": 3, "status": "exhausted"},
    ]
    assert best_so_far_curve(entries, 5) == [None, 0.2, 0.2, 0.2, 0.2]


# --------------------------------------------------------------------------- #
# Bootstrap estratificado                                                     #
# --------------------------------------------------------------------------- #
def test_stratified_difference_keeps_each_family_weight():
    first = {"a": [1.0, 1.0, 1.0], "b": [0.0] * 3}
    second = {"a": [0.0] * 3, "b": [0.0] * 3}
    interval = stratified_paired_difference(first, second)
    # sempre 3 unidades de cada família: o agregado não varia com a reamostragem
    assert interval.value == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5) and interval.high == pytest.approx(0.5)


def test_stratified_difference_drops_missing_tests():
    interval = stratified_paired_difference({"a": [1.0, np.nan]}, {"a": [0.0, 0.0]})
    assert interval.value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Injetor LLM (stub)                                                          #
# --------------------------------------------------------------------------- #
def test_provider_errors_are_told_apart():
    assert classify_provider_error("rate_limit_exceeded: Limit 8000, Used 7900 tokens per minute (TPM)") == MINUTE
    assert classify_provider_error("Rate limit reached ... tokens per day (TPD): Limit 200000") == DAILY
    assert classify_provider_error("Error code: 413 Request too large") == TOO_LARGE
    assert classify_provider_error("tool_use_failed: additionalProperties") is None
    assert classify_provider_error(None) is None


def test_round_message_shows_box_history_and_plan():
    history = [
        {"index": 0, "status": VALID, "diff": [{"path": "fault.prob", "new_value": 0.3}],
         "search": {"ap": 0.9, "soft_evasion": 0.52, "evasion": {"1.0": 0.2, "10.0": 0.1, "100.0": 0.05}}},
        {"index": 1, "status": "oracle_failed", "diff": [], "reason": "AP baixo"},
    ]
    message = build_round_message(_context("masquerade_fault", 2, history=history), plan_block="PLANO")
    assert "proposta 3 de 10" in message
    assert "fault.durationMs.max: [0, 3200]" in message
    assert "#1 [valid] fault.prob=0.300" in message and "evasão suave = 0.520" in message
    assert "#2 [oracle_failed]" in message and "AP baixo" in message
    assert "PLANO" in message


def _tool_response(**arguments):
    from adversarial_ids.agents.intent.tools import submit_intent_spec

    result = submit_intent_spec.entrypoint(**arguments)
    return SimpleNamespace(
        tools=[SimpleNamespace(tool_name="submit_intent_spec", result=result)],
        metrics=SimpleNamespace(input_tokens=100, output_tokens=20),
        status="COMPLETED",
        content=None,
    )


class _Agent:
    def __init__(self, *responses):
        self.responses = list(responses)

    def run(self, message):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _attacker(agent):
    from adversarial_ids.agents.attacker.agent import LlmAttacker, Pacer

    return LlmAttacker(
        "injection", with_defense=False, model_id="stub", agent=agent,
        pacer=Pacer(0, sleep=lambda s: None), minute_wait_s=0,
    )


def test_attacker_returns_a_gated_intent_and_counts_tokens():
    response = _tool_response(
        objective="evade_detection", base_attack="injection", desired_effect="lower_recall",
        target_values={"numInjectedMessages": 800}, max_fields_changed=1,
    )
    proposal = _attacker(_Agent(response)).propose(_context("injection", 0))
    assert proposal.intent.restrictions.target_values[0].value == 800
    assert proposal.meta["llm_calls"] == 1 and proposal.meta["input_tokens"] == 100


def test_attacker_outside_the_family_is_an_invalid_proposal():
    response = _tool_response(
        objective="evade_detection", base_attack="high_stnum", desired_effect="lower_recall",
    )
    proposal = _attacker(_Agent(response)).propose(_context("injection", 0))
    assert proposal.intent is None and "fora da família" in proposal.error


def test_minute_limit_waits_and_retries_without_counting():
    response = _tool_response(
        objective="evade_detection", base_attack="injection", desired_effect="lower_recall",
    )
    agent = _Agent(RuntimeError("429 rate_limit_exceeded tokens per minute (TPM)"), response)
    proposal = _attacker(agent).propose(_context("injection", 0))
    assert proposal.intent is not None


def test_daily_quota_stops_the_campaign():
    agent = _Agent(RuntimeError("rate_limit_exceeded: tokens per day (TPD)"))
    with pytest.raises(ProposerUnavailable):
        _attacker(agent).propose(_context("injection", 0))


def test_no_tool_call_is_an_invalid_proposal():
    response = SimpleNamespace(tools=[], metrics=None, status="COMPLETED", content="não vou chamar")
    proposal = _attacker(_Agent(response)).propose(_context("injection", 0))
    assert proposal.intent is None and "não chamou" in proposal.error


def test_soft_evasion_moves_when_every_threshold_still_fires():
    """O caso do piloto: evasão 0 em todo orçamento, mas confiança pela metade."""

    def run(score):
        return {
            "seed": 301, "rows": 100, "attack_rows": 40, "ap": 1.0, "attack_score_mean": score,
            "budgets": {"10.0": {"evasion": 0.0, "false_alerts_per_10k_normal": 0.0}},
        }

    summary = summarize_runs([run(0.4), run(0.6)])
    assert summary["evasion"]["10.0"] == 0.0
    assert summary["attack_score_mean"] == pytest.approx(0.5)
    assert summary["soft_evasion"] == pytest.approx(0.5)
