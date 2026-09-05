"""Testes de core/feedback_policy.py — política determinística do E10.

Sem LLM, sem I/O: cobre a métrica-objetivo por efeito, a escada de
escalonamento (intensity -> max_fields_changed) e a precedência dos motivos
de parada.
"""

from __future__ import annotations

import pytest

from adversarial_ids.core.feedback_policy import (
    FeedbackPolicyError,
    decide_feedback,
)
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.domain.feedback_decision import FeedbackStopReason, RoundOutcome
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentSpec,
)


def _intent(**overrides: object) -> IntentSpec:
    data: dict[str, object] = {
        "source_prompt": "Reduza o recall variando a temporização da falha.",
        "objective": IntentObjective.EVADE_DETECTION,
        "base_attack": "masquerade_fault",
        "desired_effect": DesiredEffect.LOWER_RECALL,
    }
    data.update(overrides)
    return IntentSpec.model_validate(data)


def _report(*, f1: float = 0.7, recall: float = 0.7, accuracy: float = 0.8,
            precision: float = 0.72) -> DetectionReport:
    return DetectionReport(
        model_name="random_forest",
        split="train_test_80_20_seed42",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=ConfusionMatrix(tp=12, fp=4, fn=8, tn=20),
        latency_ms=3.5,
    )


def _plan(*, priority: str = "medium") -> DefensePlan:
    return DefensePlan.model_validate(
        {
            "priority": priority,
            "detection_actions": [
                {
                    "description": "Revisar o limiar de decisão.",
                    "technique": "detector_threshold_tuning",
                    "evidence": [
                        {"metric_or_feature": "recall", "value": 0.7},
                    ],
                    "validation_test": {
                        "metric": "recall",
                        "direction": "increase",
                        "target": 0.85,
                        "procedure": "Reavaliar no mesmo split.",
                    },
                }
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Métrica-objetivo por efeito                                                 #
# --------------------------------------------------------------------------- #
def test_objective_metric_is_f1_for_lower_f1():
    decision = decide_feedback(
        intent=_intent(desired_effect=DesiredEffect.LOWER_F1),
        report=_report(f1=0.7),
        plan=_plan(),
        run_id="run-1",
    )
    assert decision.objective_metric == "f1"
    assert decision.objective_value == pytest.approx(0.7)


@pytest.mark.parametrize(
    "effect", [DesiredEffect.LOWER_RECALL, DesiredEffect.MIMIC_NORMAL_TRAFFIC]
)
def test_objective_metric_is_recall_for_lower_recall_and_mimic_normal_traffic(effect):
    decision = decide_feedback(
        intent=_intent(desired_effect=effect),
        report=_report(recall=0.65),
        plan=_plan(),
        run_id="run-1",
    )
    assert decision.objective_metric == "recall"
    assert decision.objective_value == pytest.approx(0.65)


def test_decision_stops_when_the_effect_has_no_measurable_objective():
    decision = decide_feedback(
        intent=_intent(desired_effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY),
        report=_report(),
        plan=_plan(),
        run_id="run-1",
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.NO_OBJECTIVE_METRIC
    assert decision.objective_metric is None
    assert decision.objective_value is None
    assert decision.next_intent is None


# --------------------------------------------------------------------------- #
# Rodada 1 — sem histórico                                                    #
# --------------------------------------------------------------------------- #
def test_first_round_has_no_parent_and_no_improvement_comparison():
    decision = decide_feedback(
        intent=_intent(),
        report=_report(recall=0.7),
        plan=_plan(),
        run_id="run-1",
        max_rounds=10,
    )
    assert decision.round == 1
    assert decision.parent_run_id is None
    assert decision.improvement is None
    assert decision.should_continue is True


def test_round_and_parent_are_taken_from_call_arguments():
    decision = decide_feedback(
        intent=_intent(),
        report=_report(recall=0.7),
        plan=_plan(),
        run_id="run-2",
        round_index=2,
        parent_run_id="run-1",
        max_rounds=5,
        history=(RoundOutcome(round=1, run_id="run-1", objective_value=0.9),),
    )
    assert decision.round == 2
    assert decision.parent_run_id == "run-1"


def test_rejects_a_round_index_below_one():
    with pytest.raises(FeedbackPolicyError):
        decide_feedback(
            intent=_intent(), report=_report(), plan=_plan(),
            run_id="run-1", round_index=0,
        )


def test_rejects_a_non_positive_max_rounds():
    with pytest.raises(FeedbackPolicyError):
        decide_feedback(
            intent=_intent(), report=_report(), plan=_plan(),
            run_id="run-1", max_rounds=0,
        )


# --------------------------------------------------------------------------- #
# Escada de alavancas                                                         #
# --------------------------------------------------------------------------- #
def test_escalation_raises_intensity_before_widening_fields():
    intent = _intent(intensity=IntentIntensity.MEDIUM)
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    assert decision.next_intent.intensity == IntentIntensity.HIGH
    assert (
        decision.next_intent.restrictions.max_fields_changed
        == intent.restrictions.max_fields_changed
    )


def test_escalation_widens_max_fields_changed_once_intensity_is_high():
    intent = _intent(intensity=IntentIntensity.HIGH)
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    assert decision.next_intent.intensity == IntentIntensity.HIGH
    assert (
        decision.next_intent.restrictions.max_fields_changed
        == intent.restrictions.max_fields_changed + 1
    )


def test_escalation_never_exceeds_the_number_of_candidate_fields():
    # 12 campos candidatos em masquerade_fault/lower_recall.
    intent = _intent(
        intensity=IntentIntensity.HIGH,
        restrictions={"max_fields_changed": 12},
    )
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.LEVERS_EXHAUSTED
    assert decision.next_intent is None


def test_escalation_ceiling_respects_a_narrower_allowed_fields_restriction():
    intent = _intent(
        intensity=IntentIntensity.HIGH,
        restrictions={
            "max_fields_changed": 2,
            "allowed_fields": ("fault.prob", "cbStatus"),
        },
    )
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.LEVERS_EXHAUSTED


def test_escalated_intent_preserves_effect_prompt_and_base_attack():
    intent = _intent()
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    next_intent = decision.next_intent
    assert next_intent.source_prompt == intent.source_prompt
    assert next_intent.objective == intent.objective
    assert next_intent.base_attack == intent.base_attack
    assert next_intent.desired_effect == intent.desired_effect
    assert next_intent.seed == intent.seed


def test_escalated_intent_still_compiles_into_an_attack_candidate():
    intent = _intent()
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", max_rounds=10,
    )
    candidate = compile_attack_candidate(decision.next_intent)
    assert candidate.attack_key == "masquerade_fault"


def test_decide_feedback_is_deterministic_for_the_same_input():
    kwargs = dict(
        intent=_intent(), report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", round_index=1, max_rounds=10,
    )
    first = decide_feedback(**kwargs)
    second = decide_feedback(**kwargs)
    assert first.model_dump() == second.model_dump()


# --------------------------------------------------------------------------- #
# Precedência dos motivos de parada                                           #
# --------------------------------------------------------------------------- #
def test_stops_when_the_target_metric_is_reached():
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.3), plan=_plan(),
        run_id="run-1", max_rounds=10, target_metric=0.5,
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.OBJECTIVE_REACHED


def test_target_reached_takes_precedence_over_a_plateau():
    history = (RoundOutcome(round=1, run_id="run-0", objective_value=0.30),)
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.30), plan=_plan(),
        run_id="run-1", round_index=2, parent_run_id="run-0",
        max_rounds=10, target_metric=0.5, history=history,
    )
    # Sem melhoria nenhuma (0.30 -> 0.30) mas já abaixo do alvo: alvo vence.
    assert decision.stop_reason == FeedbackStopReason.OBJECTIVE_REACHED


def test_stops_on_a_plateau_when_improvement_is_below_the_epsilon():
    history = (RoundOutcome(round=1, run_id="run-0", objective_value=0.70),)
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.695), plan=_plan(),
        run_id="run-1", round_index=2, parent_run_id="run-0",
        max_rounds=10, min_delta=0.01, history=history,
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.NO_IMPROVEMENT
    assert decision.improvement == pytest.approx(0.005)


def test_stops_on_a_plateau_when_the_metric_gets_worse():
    history = (RoundOutcome(round=1, run_id="run-0", objective_value=0.60),)
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.70), plan=_plan(),
        run_id="run-1", round_index=2, parent_run_id="run-0",
        max_rounds=10, min_delta=0.01, history=history,
    )
    assert decision.stop_reason == FeedbackStopReason.NO_IMPROVEMENT
    assert decision.improvement == pytest.approx(-0.10)


def test_continues_when_improvement_exceeds_the_epsilon():
    history = (RoundOutcome(round=1, run_id="run-0", objective_value=0.70),)
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.50), plan=_plan(),
        run_id="run-1", round_index=2, parent_run_id="run-0",
        max_rounds=10, min_delta=0.01, history=history,
    )
    assert decision.should_continue is True
    assert decision.improvement == pytest.approx(0.20)


def test_stops_when_the_round_budget_is_reached():
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.5), plan=_plan(),
        run_id="run-1", round_index=3, parent_run_id="run-2", max_rounds=3,
        history=(
            RoundOutcome(round=1, run_id="run-0", objective_value=0.9),
            RoundOutcome(round=2, run_id="run-1", objective_value=0.7),
        ),
    )
    assert decision.should_continue is False
    assert decision.stop_reason == FeedbackStopReason.MAX_ROUNDS_REACHED


def test_levers_exhausted_takes_precedence_over_the_round_budget():
    intent = _intent(intensity=IntentIntensity.HIGH, restrictions={"max_fields_changed": 12})
    decision = decide_feedback(
        intent=intent, report=_report(recall=0.7), plan=_plan(),
        run_id="run-1", round_index=1, max_rounds=10,
    )
    assert decision.stop_reason == FeedbackStopReason.LEVERS_EXHAUSTED


def test_plateau_takes_precedence_over_the_round_budget():
    history = (RoundOutcome(round=1, run_id="run-0", objective_value=0.70),)
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.70), plan=_plan(),
        run_id="run-1", round_index=2, parent_run_id="run-0",
        max_rounds=2, min_delta=0.01, history=history,
    )
    assert decision.stop_reason == FeedbackStopReason.NO_IMPROVEMENT


# --------------------------------------------------------------------------- #
# Melhor rodada                                                               #
# --------------------------------------------------------------------------- #
def test_best_value_so_far_is_the_lowest_value_measured():
    history = (
        RoundOutcome(round=1, run_id="run-0", objective_value=0.70),
        RoundOutcome(round=2, run_id="run-1", objective_value=0.55),
    )
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.60), plan=_plan(),
        run_id="run-2", round_index=3, parent_run_id="run-1",
        max_rounds=10, min_delta=0.0, history=history,
    )
    assert decision.best_value_so_far == pytest.approx(0.55)


def test_defense_priority_comes_from_the_plan_not_recomputed():
    decision = decide_feedback(
        intent=_intent(), report=_report(recall=0.7), plan=_plan(priority="critical"),
        run_id="run-1", max_rounds=10,
    )
    assert decision.defense_priority == "critical"
