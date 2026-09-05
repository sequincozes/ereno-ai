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
    DetectorManifest,
    DroppedColumn,
    Evidence,
    FeatureManifest,
    FeedbackDecision,
    FeedbackStopReason,
    FieldChange,
    IntentObjective,
    IntentSpec,
    LoopRecord,
    LoopStage,
    LoopStageStatus,
    ScalerStat,
    SelectionManifest,
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


def _feature_manifest(**overrides: object) -> FeatureManifest:
    data: dict[str, object] = {
        "label_column": "class",
        "feature_columns": ("num", "cat"),
        "dropped_columns": (DroppedColumn(name="Time", reason="always_drop"),),
        "numeric_features": ("num",),
        "categorical_features": ("cat",),
        "numeric_fill_values": {"num": 0.0},
        "categorical_codes": {"cat": {"a": 0, "b": 1}},
        "fitted_rows": 70,
        "fitted_content_hash": _SHA256_ZEROS,
    }
    data.update(overrides)
    return FeatureManifest.model_validate(data)


def _detector_manifest(**overrides: object) -> DetectorManifest:
    data: dict[str, object] = {
        "detector": "random_forest",
        "model_name": "random_forest",
        "importance_kind": "gini",
        "supports_shap": True,
        "trained_rows": 70,
        "trained_features": ("num", "cat"),
        "train_duration_seconds": 0.5,
    }
    data.update(overrides)
    return DetectorManifest.model_validate(data)


def _selection_manifest(**overrides: object) -> SelectionManifest:
    data: dict[str, object] = {
        "candidate_features": ("num", "cat"),
        "selected_features": ("num", "cat"),
        "fitted_rows_before_undersampling": 70,
        "fitted_rows_after_undersampling": 70,
    }
    data.update(overrides)
    return SelectionManifest.model_validate(data)


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
        lambda: _feature_manifest(),
        lambda: _selection_manifest(),
        lambda: _detector_manifest(),
    ],
    ids=[
        "AttackCandidate",
        "DatasetBundle",
        "DetectionReport",
        "DefensePlan",
        "LoopRecord",
        "FeatureManifest",
        "SelectionManifest",
        "DetectorManifest",
    ],
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
# FeatureManifest (E6)                                                        #
# --------------------------------------------------------------------------- #
def test_feature_manifest_rejects_numeric_categorical_overlap():
    with pytest.raises(ValidationError, match="não podem se sobrepor"):
        _feature_manifest(
            feature_columns=("num",),
            numeric_features=("num",),
            categorical_features=("num",),
            numeric_fill_values={"num": 0.0},
            categorical_codes={},
        )


def test_feature_manifest_rejects_feature_columns_not_covered_by_numeric_or_categorical():
    with pytest.raises(ValidationError, match="deve cobrir exatamente"):
        _feature_manifest(feature_columns=("num", "cat", "orphan"))


def test_feature_manifest_rejects_dropped_column_that_is_also_a_feature():
    with pytest.raises(ValidationError, match="não podem se sobrepor"):
        _feature_manifest(
            dropped_columns=(DroppedColumn(name="num", reason="constant"),),
        )


def test_feature_manifest_rejects_fill_value_for_unknown_numeric_feature():
    with pytest.raises(ValidationError, match="numeric_fill_values"):
        _feature_manifest(numeric_fill_values={"num": 0.0, "ghost": 1.0})


def test_feature_manifest_rejects_categorical_codes_for_unknown_feature():
    with pytest.raises(ValidationError, match="categorical_codes"):
        _feature_manifest(categorical_codes={"cat": {"a": 0}, "ghost": {"x": 0}})


def test_feature_manifest_rejects_standard_scaler_missing_stats():
    with pytest.raises(ValidationError, match="scaler='standard'"):
        _feature_manifest(scaler="standard", scaler_stats={})


def test_feature_manifest_accepts_standard_scaler_with_complete_stats():
    manifest = _feature_manifest(
        scaler="standard",
        scaler_stats={"num": ScalerStat(mean=0.0, std=1.0)},
    )
    assert manifest.scaler_stats["num"].std == 1.0


def test_feature_manifest_rejects_non_positive_fitted_rows():
    with pytest.raises(ValidationError):
        _feature_manifest(fitted_rows=0)


def test_feature_manifest_rejects_malformed_hash():
    with pytest.raises(ValidationError):
        _feature_manifest(fitted_content_hash="not-a-sha256")


def test_feature_manifest_rejects_non_exhaustive_numeric_fill_values():
    with pytest.raises(ValidationError, match="numeric_fill_values deve cobrir"):
        _feature_manifest(numeric_fill_values={})


def test_feature_manifest_rejects_non_exhaustive_categorical_codes():
    with pytest.raises(ValidationError, match="categorical_codes deve cobrir"):
        _feature_manifest(categorical_codes={})


def test_feature_manifest_rejects_non_finite_fill_value():
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        _feature_manifest(numeric_fill_values={"num": float("nan")})


def test_feature_manifest_rejects_non_finite_scaler_stat():
    with pytest.raises(ValidationError, match="NaN/Infinity"):
        _feature_manifest(
            scaler="standard",
            scaler_stats={"num": ScalerStat(mean=float("inf"), std=1.0)},
        )


def test_feature_manifest_rejects_duplicate_dropped_column_names():
    with pytest.raises(ValidationError, match="não pode repetir"):
        _feature_manifest(
            dropped_columns=(
                DroppedColumn(name="Time", reason="always_drop"),
                DroppedColumn(name="Time", reason="constant"),
            ),
        )


# --------------------------------------------------------------------------- #
# SelectionManifest (E7)                                                      #
# --------------------------------------------------------------------------- #
def test_selection_manifest_rejects_selected_feature_outside_candidates():
    with pytest.raises(ValidationError, match="fora de candidate_features"):
        _selection_manifest(selected_features=("num", "ghost"))


def test_selection_manifest_rejects_feature_scores_when_strategy_is_none():
    with pytest.raises(ValidationError, match="só é preenchido quando"):
        _selection_manifest(feature_scores={"num": 0.1})


def test_selection_manifest_requires_feature_scores_for_mutual_info():
    with pytest.raises(ValidationError, match="deve cobrir exatamente candidate_features"):
        _selection_manifest(
            feature_selection="mutual_info",
            feature_selection_top_k=1,
            feature_scores={"num": 0.5},  # falta "cat"
        )


def test_selection_manifest_accepts_complete_feature_scores_for_mutual_info():
    manifest = _selection_manifest(
        feature_selection="mutual_info",
        feature_selection_top_k=1,
        selected_features=("num",),
        feature_scores={"num": 0.5, "cat": 0.1},
    )
    assert manifest.feature_scores["num"] == 0.5


def test_selection_manifest_rejects_mutual_info_without_a_stopping_criterion():
    with pytest.raises(ValidationError, match="exige feature_selection_top_k"):
        _selection_manifest(
            feature_selection="mutual_info",
            feature_scores={"num": 0.5, "cat": 0.1},
        )


def test_selection_manifest_rejects_rows_increasing_after_undersampling():
    with pytest.raises(ValidationError, match="não pode ser maior"):
        _selection_manifest(fitted_rows_before_undersampling=10, fitted_rows_after_undersampling=20)


def test_selection_manifest_rejects_row_count_change_when_undersampling_is_none():
    with pytest.raises(ValidationError, match="não pode alterar o número de linhas"):
        _selection_manifest(fitted_rows_before_undersampling=10, fitted_rows_after_undersampling=8)


def test_selection_manifest_accepts_row_count_drop_when_undersampling_is_random():
    manifest = _selection_manifest(
        undersampling="random",
        fitted_rows_before_undersampling=10,
        fitted_rows_after_undersampling=8,
        class_counts_before={"normal": 6, "attack": 4},
        class_counts_after={"normal": 4, "attack": 4},
    )
    assert manifest.fitted_rows_after_undersampling == 8


def test_selection_manifest_rejects_class_counts_before_not_matching_row_total():
    with pytest.raises(ValidationError, match="class_counts_before"):
        _selection_manifest(class_counts_before={"normal": 1, "attack": 1})


def test_selection_manifest_rejects_class_counts_after_not_matching_row_total():
    with pytest.raises(ValidationError, match="class_counts_after"):
        _selection_manifest(class_counts_after={"normal": 1, "attack": 1})


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


# --------------------------------------------------------------------------- #
# DetectorManifest (E8)                                                       #
# --------------------------------------------------------------------------- #
def test_detector_manifest_rejects_importance_kind_that_the_detector_cannot_produce():
    # Um SVM RBF não expõe importância nenhuma; anunciar "gini" faria o
    # consumidor comparar um ranking que não existe.
    with pytest.raises(ValidationError, match="produz importance_kind"):
        _detector_manifest(detector="svm_rbf", model_name="svm_rbf", importance_kind="gini")


def test_detector_manifest_rejects_shap_support_on_a_non_tree_detector():
    with pytest.raises(ValidationError, match="só vale para detectores de árvore"):
        _detector_manifest(
            detector="svm_linear",
            model_name="svm_linear",
            importance_kind="linear_coef",
            supports_shap=True,
        )


def test_detector_manifest_accepts_the_svm_linear_combination():
    manifest = _detector_manifest(
        detector="svm_linear",
        model_name="svm_linear",
        importance_kind="linear_coef",
        supports_shap=False,
        requires_scaling=True,
        resolved_scaler="standard",
    )
    assert manifest.importance_kind == "linear_coef"
    assert manifest.resolved_scaler == "standard"


def test_detector_manifest_records_a_scaler_disagreeing_with_the_recommendation():
    # Legítimo (ablação controlada), por isso não é erro — mas fica registrado.
    manifest = _detector_manifest(
        detector="svm_rbf",
        model_name="svm_rbf",
        importance_kind="none",
        supports_shap=False,
        requires_scaling=True,
        resolved_scaler="none",
    )
    assert manifest.requires_scaling is True
    assert manifest.resolved_scaler == "none"


def test_detector_manifest_rejects_an_unregistered_detector_key():
    with pytest.raises(ValidationError):
        _detector_manifest(detector="xgboost", model_name="xgboost")


def test_detector_manifest_requires_a_non_empty_trained_feature_space():
    with pytest.raises(ValidationError):
        _detector_manifest(trained_features=())


def test_detector_manifest_requires_positive_trained_rows():
    with pytest.raises(ValidationError):
        _detector_manifest(trained_rows=0)
