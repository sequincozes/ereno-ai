"""E3: linhagem, retreino adversarial, placebo, matriz cruzada e medidas — sem Java nem Groq.

Os detectores reais só aparecem no teste de ``fit_traces`` (sobre o
``data/baseline_dataset.csv``); o resto troca o treino por um registro do que
seria treinado.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from adversarial_ids.agents.attacker.context import build_round_message
from adversarial_ids.config.settings import BASELINE_DATASET_PATH
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.experiments.arms_race import (
    DetectorLineage,
    IncompleteRound,
    Lineage,
    adversarial_extras,
    importance_shift,
    placebo_closure,
    placebo_seeds,
    plan_measures,
    race_measures,
    reference_cost,
    round_complete,
    top_variants,
)
from adversarial_ids.experiments.protocol import SeedPlan
from adversarial_ids.experiments.search import (
    VALID,
    ProposalContext,
    RandomSearch,
    SearchBox,
    run_campaign,
)
from adversarial_ids.experiments.sensitivity import reference_config


def _valid(index, objective, ap=0.5, config=None):
    return {
        "index": index,
        "status": VALID,
        "objective": objective,
        "search": {"ap": ap},
        "config": config or {"v": index},
        "diff": [{"path": "fault.prob", "new_value": index / 10}],
    }


def _ledger(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Seleção das variantes de treino                                             #
# --------------------------------------------------------------------------- #
def test_top_variants_follow_the_selection_order_and_skip_invalid():
    entries = [
        _valid(0, 0.4),
        {"index": 1, "status": "oracle_failed"},
        _valid(2, 0.7, ap=0.9),
        _valid(3, 0.7, ap=0.6),
        _valid(4, 0.1),
        {"index": 5, "strategy": "B2", "status": "exhausted"},
    ]
    assert [e["index"] for e in top_variants(entries, 3)] == [3, 2, 0]
    assert [e["index"] for e in top_variants(entries[:2], 3)] == [0]


def test_a_round_is_complete_at_the_budget_or_when_exhausted():
    assert not round_complete([_valid(i, 0.1) for i in range(9)], 10)
    assert round_complete([_valid(i, 0.1) for i in range(10)], 10)
    assert round_complete([_valid(0, 0.1), {"index": 1, "status": "exhausted"}], 10)


def test_extras_accumulate_round_by_round_and_seed_by_seed():
    selections = [[_valid(0, 0.5), _valid(1, 0.4)], [_valid(7, 0.9)]]
    extras = adversarial_extras(selections, (101, 102))
    assert extras == (
        ({"v": 0}, 101), ({"v": 0}, 102),
        ({"v": 1}, 101), ({"v": 1}, 102),
        ({"v": 7}, 101), ({"v": 7}, 102),
    )


def test_placebo_seeds_stay_out_of_every_protocol_set():
    seeds = SeedPlan()
    chosen = placebo_seeds(36, seeds)
    assert chosen[0] == 111 and len(chosen) == 36
    used = {*seeds.train, *seeds.calibration, *seeds.search, *seeds.test}
    assert not used.intersection(chosen)
    with pytest.raises(ValueError):
        placebo_seeds(100, seeds)


# --------------------------------------------------------------------------- #
# Linhagem                                                                    #
# --------------------------------------------------------------------------- #
class _RecordingLineage(DetectorLineage):
    """Em vez de treinar, devolve o que o treino receberia."""

    def _frozen(self, extras, max_rows=None):
        return {"extras": tuple(extras), "max_rows": max_rows}


def _lineage(tmp_path, defended=True, rounds=3):
    lineage = Lineage("high_stnum", "B1", 1, defended=defended)
    dl = _RecordingLineage(lineage, {"ref": True}, store=None, e1_root=tmp_path / "e1", e3_root=tmp_path / "e3", budget=2)
    for r in range(rounds):
        _ledger(dl.ledger_path(r), [_valid(0, 0.1 * r, config={"r": r, "i": 0}), _valid(1, 0.5, config={"r": r, "i": 1})])
    return dl


def test_round_zero_is_the_e1_ledger_and_later_rounds_live_in_e3(tmp_path):
    dl = _lineage(tmp_path, rounds=0)
    assert dl.ledger_path(0) == tmp_path / "e1" / "ledger" / "high_stnum" / "B1" / "rep1.jsonl"
    assert dl.ledger_path(2) == tmp_path / "e3" / "ledger" / "high_stnum" / "B1" / "rep1" / "round2.jsonl"
    nodef = _lineage(tmp_path, defended=False, rounds=0)
    assert nodef.ledger_path(0) == dl.ledger_path(0)  # a rodada 0 é a mesma
    assert "B1-nodef" in nodef.ledger_path(1).parts


def test_detector_r_sees_the_top_variants_of_every_earlier_round(tmp_path):
    dl = _lineage(tmp_path)
    assert dl.detector(0)["extras"] == ()
    extras = dl.detector(2)["extras"]
    configs = [config for config, _ in extras]
    # rodada 0: a 1 (0.5) antes da 0 (0.0); rodada 1: a 1 (0.5) antes da 0 (0.1)
    assert configs[::3] == [{"r": 0, "i": 1}, {"r": 0, "i": 0}, {"r": 1, "i": 1}, {"r": 1, "i": 0}]
    assert [seed for _, seed in extras[:3]] == [101, 102, 103]


def test_without_defense_every_detector_is_d0(tmp_path):
    assert _lineage(tmp_path, defended=False).detector(2)["extras"] == ()


def test_placebo_gets_as_many_reference_traces_and_the_same_row_cap(tmp_path):
    dl = _lineage(tmp_path)
    placebo = dl.placebo_detector(2, rows=1234)
    assert placebo["max_rows"] == 1234
    assert len(placebo["extras"]) == len(dl.detector(2)["extras"]) == 12
    assert all(config == {"ref": True} for config, _ in placebo["extras"])
    assert [seed for _, seed in placebo["extras"]] == list(range(111, 123))


def test_an_unfinished_round_blocks_the_next_detector(tmp_path):
    dl = _lineage(tmp_path, rounds=1)
    _ledger(dl.ledger_path(1), [_valid(0, 0.3)])  # 1 de 2 propostas
    with pytest.raises(IncompleteRound):
        dl.detector(2)


# --------------------------------------------------------------------------- #
# Rodadas da busca                                                            #
# --------------------------------------------------------------------------- #
def _context(index, race_round=0, prior=()):
    box = SearchBox.for_family("high_stnum", reference_config("high_stnum"))
    return ProposalContext("high_stnum", 1, index, 10, box, (), race_round, tuple(prior))


def test_random_search_keeps_the_e1_draw_in_round_zero_and_redraws_later():
    search = RandomSearch()
    e1 = search.propose(ProposalContext("high_stnum", 1, 4, 10, _context(0).box, ()))
    assert search.propose(_context(4)).intent == e1.intent
    later = search.propose(_context(4, race_round=1)).intent
    assert later.restrictions.target_values != e1.intent.restrictions.target_values
    assert later.source_prompt.startswith("E3 rodada 1")


class _Evaluator:
    def __init__(self):
        self.box = _context(0).box

    def evaluate(self, intent, error=None):
        return {"status": VALID, "objective": 0.1, "search": {"ap": 0.5}, "diff": [], "config": {}}


class _Proposer:
    strategy = "stub"

    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return RandomSearch().propose(context)


def test_campaign_passes_the_round_and_the_memory_to_the_proposer(tmp_path):
    proposer, prior = _Proposer(), ({"race_round": 0},)
    entries = run_campaign(
        proposer, _Evaluator(), family="high_stnum", rep=1, ledger_path=tmp_path / "r.jsonl",
        budget=2, race_round=2, prior=prior,
    )
    assert [c.race_round for c in proposer.contexts] == [2, 2]
    assert proposer.contexts[0].prior == prior
    assert all(e["race_round"] == 2 for e in entries)


def test_round_message_shows_the_prior_rounds_only_when_there_are_some():
    assert "Rodadas anteriores" not in build_round_message(_context(0))
    prior = [{"race_round": 0, "diff": [{"path": "stNum.min", "new_value": 7}],
              "objective_then": 0.42, "objective_now": 0.05}]
    message = build_round_message(_context(0, race_round=1, prior=prior))
    assert "rodada 0: stNum.min=7" in message
    assert "0.420" in message and "0.050" in message


# --------------------------------------------------------------------------- #
# Medidas                                                                     #
# --------------------------------------------------------------------------- #
def test_race_measures_read_the_matrix_as_the_protocol_says():
    # D0..D3 × A0..A2
    matrix = np.array(
        [
            [0.6, 0.9, 0.9],
            [0.1, 0.5, 0.8],
            [0.2, 0.1, 0.4],
            [0.3, 0.2, 0.0],
        ]
    )
    out = race_measures(matrix)
    assert out["closure"] == pytest.approx([0.5, 0.4, 0.4])
    assert out["forgetting"] == pytest.approx([0.3 - 0.1, 0.2 - 0.1])
    assert out["race"] == pytest.approx([0.6, 0.5, 0.4])
    assert out["race_change"] == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        race_measures(matrix[:3])


def test_placebo_closure_uses_the_detector_after_each_round():
    matrix = np.array([[0.6, 0.9], [0.1, 0.5], [0.2, 0.1]])
    placebo = np.array([[0.55, 0.9], [0.5, 0.45]])  # D'1, D'2
    assert placebo_closure(matrix, placebo) == pytest.approx([0.05, 0.05])


def test_reference_cost_sums_counts_before_the_ratios():
    runs = [
        {"ap": 0.9, "budgets": {"10.0": {"tp": 90, "fn": 10, "fp": 0, "tn": 900, "false_alerts_per_10k_normal": 0.0}}},
        {"ap": 0.7, "budgets": {"10.0": {"tp": 10, "fn": 90, "fp": 2, "tn": 898, "false_alerts_per_10k_normal": 22.2}}},
    ]
    cost = reference_cost(runs)
    assert cost["recall"] == pytest.approx(0.5)
    assert cost["precision"] == pytest.approx(100 / 102)
    assert cost["f1"] == pytest.approx(2 * (100 / 102) * 0.5 / (100 / 102 + 0.5))
    assert cost["ap"] == pytest.approx(0.8)


def test_importance_shift_and_plan_overlap():
    before = [{"feature": "a", "importance": 0.5}, {"feature": "b", "importance": 0.3}, {"feature": "c", "importance": 0.2}]
    after = [{"feature": "a", "importance": 0.2}, {"feature": "b", "importance": 0.3}, {"feature": "d", "importance": 0.5}]
    assert importance_shift(before, after, top=2) == ["d", "a"]
    plan = {
        "detection_actions": [
            {"technique": "detector_retraining", "evidence": [{"metric_or_feature": "a"}, {"metric_or_feature": "recall"}]},
        ],
        "containment_actions": [{"technique": "operator_alerting", "evidence": [{"metric_or_feature": "c"}]}],
        "hardening_actions": [],
    }
    out = plan_measures(plan, {"a", "b", "c", "d"}, ["d", "a"])
    assert out["recommends_retraining"] is True
    assert out["cited_features"] == ["a", "c"]
    assert out["overlap"] == pytest.approx(0.5)
    assert plan_measures(None, set(), [])["accepted"] is False


# --------------------------------------------------------------------------- #
# Treino com linhas só de ataque                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not BASELINE_DATASET_PATH.exists(), reason="data/baseline_dataset.csv não versionado")
def test_fit_traces_adds_only_attack_rows_up_to_the_cap():
    path = str(BASELINE_DATASET_PATH)
    plain = IdsEvaluator(n_estimators=5, random_state=42).fit_traces([path])
    assert "attack_only_rows" not in plain  # o E1 não muda
    normal = plain["class_counts"]["normal"]
    attack = plain["rows"] - normal

    doubled = IdsEvaluator(n_estimators=5, random_state=42).fit_traces([path], attack_only_paths=[path])
    assert doubled["attack_only_rows"] == attack
    assert doubled["class_counts"]["normal"] == normal
    assert doubled["rows"] == plain["rows"] + attack

    capped = IdsEvaluator(n_estimators=5, random_state=42).fit_traces(
        [path], attack_only_paths=[path], attack_only_max_rows=100
    )
    assert capped["attack_only_rows"] == 100 and capped["rows"] == plain["rows"] + 100
