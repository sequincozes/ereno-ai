"""Congelamento dos contratos restantes (ação 72h #2 do plano de 60 dias).

``IntentSpec`` já tinha cobertura própria (``test_intent_contract.py``). Este
módulo cobre os cinco contratos que faltavam: ``AttackCandidate``,
``DatasetBundle``, ``DetectionReport``, ``DefensePlan`` e ``LoopRecord``.
Cada um é testado por: versionamento/congelamento/round-trip JSON e pelas
regras de consistência interna específicas do contrato.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from adversarial_ids.domain import (
    AttackCandidate,
    ConfusionMatrix,
    DatasetBundle,
    DefenseAction,
    DefensePlan,
    DesiredEffect,
    DetectionReport,
    Evidence,
    FeedbackDecision,
    FeedbackStopReason,
    FieldChange,
    IntentObjective,
    IntentSpec,
    LoopRecord,
    LoopStage,
    LoopStageStatus,
)

_SHA256_ZEROS = "0" * 64


def _intent(**overrides: object) -> IntentSpec:
    data: dict[str, object] = {
        "source_prompt": "Reduza o recall variando a temporização da falha.",
        "objective": IntentObjective.EVADE_DETECTION,
        "base_attack": "masquerade_fault",
        "desired_effect": DesiredEffect.LOWER_RECALL,
    }
    data.update(overrides)
    return IntentSpec.model_validate(data)


def _candidate(**overrides: object) -> AttackCandidate:
    data: dict[str, object] = {
        "source_intent": _intent(),
        "capability_id": "masquerade_fault.v1",
        "attack_key": "masquerade_fault",
        "config": {"fault": {"prob": 0.5}},
        "diff": [
            FieldChange(path="fault.prob", old_value=0.2, new_value=0.5),
        ],
        "rationale": "Aumenta a probabilidade de falha para reduzir o recall.",
    }
    data.update(overrides)
    return AttackCandidate.model_validate(data)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _candidate(),
        lambda: DatasetBundle(
            trace_path="outputs/experiments/run-1/trace.csv",
            columns=("ts", "field", "label"),
            classes=("normal", "attack"),
            class_counts={"normal": 100, "attack": 20},
            content_hash=_SHA256_ZEROS,
            lineage_run_id="run-1",
        ),
        lambda: DetectionReport(
            model_name="random_forest",
            split="train_test_80_20_seed42",
            accuracy=0.9,
            precision=0.85,
            recall=0.8,
            f1=0.82,
            confusion_matrix=ConfusionMatrix(tp=16, fp=3, fn=4, tn=97),
            latency_ms=12.5,
        ),
        lambda: DefensePlan(
            priority="high",
            detection_actions=(
                DefenseAction(
                    description="Monitorar variação de stNum acima do limiar.",
                    evidence=(
                        Evidence(metric_or_feature="recall_attack", value=0.42),
                    ),
                    validation_method="Reexecutar detector com regra ativa e medir recall.",
                ),
            ),
        ),
        lambda: LoopRecord(
            run_id="run-1",
            source_prompt="Reduza o recall variando a temporização da falha.",
            seed=42,
            stages=(
                LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),
                LoopStage(name="generator", status=LoopStageStatus.PENDING),
            ),
        ),
    ],
    ids=["AttackCandidate", "DatasetBundle", "DetectionReport", "DefensePlan", "LoopRecord"],
)
def test_contract_is_versioned_frozen_and_json_stable(build):
    instance = build()

    assert instance.schema_version == 1
    dumped = instance.model_dump(mode="json")
    assert type(instance).model_validate_json(json.dumps(dumped)) == instance

    first_field = next(iter(type(instance).model_fields))
    with pytest.raises(ValidationError):
        setattr(instance, first_field, getattr(instance, first_field))


def test_attack_candidate_rejects_attack_key_mismatch():
    with pytest.raises(ValidationError, match="attack_key deve ser igual"):
        _candidate(attack_key="other_attack")


def test_attack_candidate_rejects_duplicate_diff_paths():
    with pytest.raises(ValidationError, match="mesmo campo mais de uma vez"):
        _candidate(
            diff=[
                FieldChange(path="fault.prob", old_value=0.2, new_value=0.5),
                FieldChange(path="fault.prob", old_value=0.5, new_value=0.9),
            ]
        )


def test_attack_candidate_requires_at_least_one_diff_entry():
    with pytest.raises(ValidationError):
        _candidate(diff=[])


def test_dataset_bundle_rejects_class_count_mismatch():
    with pytest.raises(ValidationError, match="mesmo conjunto de rótulos"):
        DatasetBundle(
            trace_path="trace.csv",
            columns=("a",),
            classes=("normal", "attack"),
            class_counts={"normal": 10},
            content_hash=_SHA256_ZEROS,
            lineage_run_id="run-1",
        )


def test_dataset_bundle_rejects_non_positive_class_count():
    with pytest.raises(ValidationError, match="não pode conter contagens não positivas"):
        DatasetBundle(
            trace_path="trace.csv",
            columns=("a",),
            classes=("normal", "attack"),
            class_counts={"normal": 10, "attack": 0},
            content_hash=_SHA256_ZEROS,
            lineage_run_id="run-1",
        )


def test_dataset_bundle_rejects_malformed_hash():
    with pytest.raises(ValidationError):
        DatasetBundle(
            trace_path="trace.csv",
            columns=("a",),
            classes=("normal",),
            class_counts={"normal": 10},
            content_hash="not-a-sha256",
            lineage_run_id="run-1",
        )


def test_defense_plan_rejects_a_plan_without_any_action():
    with pytest.raises(ValidationError, match="ao menos uma ação"):
        DefensePlan(priority="low")


def test_defense_action_rejects_evidence_free_claims():
    with pytest.raises(ValidationError):
        DefenseAction(
            description="Bloquear tudo.",
            evidence=(),
            validation_method="N/A",
        )


def test_loop_record_requires_at_least_one_stage():
    with pytest.raises(ValidationError):
        LoopRecord(
            run_id="run-1",
            source_prompt="prompt",
            seed=42,
            stages=(),
        )


# --------------------------------------------------------------------------- #
# Linhagem da campanha (E10) — LoopRecord.round / parent_run_id               #
# --------------------------------------------------------------------------- #
def _loop_record(**overrides: object) -> LoopRecord:
    data: dict[str, object] = {
        "run_id": "run-1",
        "source_prompt": "Reduza o recall.",
        "seed": 42,
        "stages": (LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),),
    }
    data.update(overrides)
    return LoopRecord.model_validate(data)


def test_loop_record_defaults_round_to_one_with_no_parent():
    record = _loop_record()

    assert record.round == 1
    assert record.parent_run_id is None


def test_loop_record_rejects_a_round_below_one():
    with pytest.raises(ValidationError):
        _loop_record(round=0)


def test_loop_record_accepts_a_parent_run_id_for_a_chained_round():
    record = _loop_record(round=2, parent_run_id="run-1")

    assert record.round == 2
    assert record.parent_run_id == "run-1"


def test_loop_record_rejects_round_one_with_a_parent_run_id():
    with pytest.raises(ValidationError, match="parent_run_id"):
        _loop_record(round=1, parent_run_id="run-0")


def test_loop_record_rejects_a_later_round_without_a_parent_run_id():
    with pytest.raises(ValidationError, match="parent_run_id"):
        _loop_record(round=2, parent_run_id=None)


# --------------------------------------------------------------------------- #
# FeedbackDecision (E10)                                                      #
# --------------------------------------------------------------------------- #
def _decision(**overrides: object) -> FeedbackDecision:
    data: dict[str, object] = {
        "should_continue": True,
        "stop_reason": FeedbackStopReason.CONTINUE_CAMPAIGN,
        "objective_metric": "recall",
        "objective_value": 0.8,
        "best_value_so_far": 0.8,
        "best_round": 1,
        "round": 1,
        "run_id": "run-1",
        "defense_priority": "high",
        "next_intent": _intent(),
        "rationale": "recall=0.8000 na rodada 1 de 3; escalando intensidade.",
    }
    data.update(overrides)
    return FeedbackDecision.model_validate(data)


def test_feedback_decision_is_versioned_frozen_and_json_stable():
    decision = _decision()

    assert decision.schema_version == 1
    dumped = decision.model_dump(mode="json")
    assert FeedbackDecision.model_validate_json(json.dumps(dumped)) == decision


def test_feedback_decision_rejects_continue_without_next_intent():
    with pytest.raises(ValidationError, match="next_intent"):
        _decision(should_continue=True, next_intent=None)


def test_feedback_decision_rejects_stop_with_a_next_intent():
    with pytest.raises(ValidationError, match="next_intent"):
        _decision(
            should_continue=False,
            stop_reason=FeedbackStopReason.MAX_ROUNDS_REACHED,
        )


def test_feedback_decision_rejects_should_continue_disagreeing_with_stop_reason():
    with pytest.raises(ValidationError, match="stop_reason"):
        _decision(
            should_continue=False,
            stop_reason=FeedbackStopReason.CONTINUE_CAMPAIGN,
            next_intent=None,
        )


def test_feedback_decision_rejects_round_one_with_a_parent_run_id():
    with pytest.raises(ValidationError, match="parent_run_id"):
        _decision(round=1, parent_run_id="run-0")


def test_feedback_decision_rejects_a_later_round_without_a_parent_run_id():
    with pytest.raises(ValidationError, match="parent_run_id"):
        _decision(round=2, parent_run_id=None)
