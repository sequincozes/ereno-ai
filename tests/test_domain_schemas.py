"""Testes de compatibilidade dos schemas de domain/ (issue #2).

Garantem que os contratos congelados aceitam exatamente o que o código atual
produz/consome hoje — o critério "compatíveis com o código atual" da issue.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adversarial_ids.domain import AttackConfig, IterationRecord, Metrics, StrategistOutput

BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_ATTACK_JSON = BASE_DIR / "inputs" / "uc03_masquerade_fault.json"


# --------------------------------------------------------------------------- #
# AttackConfig                                                                 #
# --------------------------------------------------------------------------- #
def test_attack_config_round_trips_baseline_json():
    """O JSON baseline valida e ``model_dump`` reproduz o arquivo idêntico.

    O gerador ERENO lê esse JSON de volta, então o round-trip precisa ser
    lossless (inclusive tipos int vs. float e nomes camelCase).
    """
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))

    config = AttackConfig.model_validate(raw)
    dumped = config.model_dump()

    assert dumped == raw


def test_attack_config_has_twelve_editable_fields_intact():
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
    config = AttackConfig.model_validate(raw)

    assert config.fault.prob == 0.6
    assert config.fault.durationMs.min == 50
    assert config.fault.durationMs.max == 800
    assert config.cbStatus == 1
    assert config.incrementStNumOnFault is True
    assert config.sqnumMode == "fast"
    assert config.ttlMsValues == [20, 40, 80]
    assert config.analog.deltaAbs.min == 0.2
    assert config.analog.deltaAbs.max == 0.8
    assert config.trapArea.multiplier.min == 1.2
    assert config.trapArea.multiplier.max == 2.5
    assert config.trapArea.spikeProb == 0.5


def test_attack_config_rejects_unknown_field():
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
    raw["campoInexistente"] = 123

    with pytest.raises(ValidationError):
        AttackConfig.model_validate(raw)


def test_attack_config_rejects_out_of_range_probability():
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
    raw["fault"]["prob"] = 1.5

    with pytest.raises(ValidationError):
        AttackConfig.model_validate(raw)


def test_attack_config_rejects_min_greater_than_max():
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
    raw["analog"]["deltaAbs"]["min"] = 0.9  # max é 0.8

    with pytest.raises(ValidationError):
        AttackConfig.model_validate(raw)


def test_attack_config_rejects_invalid_cb_status():
    raw = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
    raw["cbStatus"] = 2

    with pytest.raises(ValidationError):
        AttackConfig.model_validate(raw)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _evaluator_like_metrics() -> dict:
    """Reproduz a saída de ``ids_evaluator._compute_metrics`` (variante)."""
    return {
        "evaluation_type": "variant_external_test",
        "dataset_path": "outputs/dataset_iteration_1.csv",
        "accuracy": 0.97,
        "attack_label_encoded": 1,
        "attack_label_original": "masquerade_fake_fault",
        "precision_attack": 0.95,
        "recall_attack": 0.90,
        "f1_score_attack": 0.925,
        "support_attack": 300,
        "attack_count": 300,
        "normal_count": 700,
        "tp": 270,
        "fp": 15,
        "fn": 30,
        "tn": 685,
        "label_column": "class",
        "class_mapping": {0: "normal", 1: "masquerade_fake_fault"},
        "dataset_rows": 1000,
        "dataset_columns": 42,
        "removed_columns": ["Time", "SqNum"],
        "used_features": ["cbStatus", "sv"],
        "top_feature_importances": [
            {"feature": "cbStatus", "importance": 0.42},
            {"feature": "sv", "importance": 0.31},
        ],
    }


def test_metrics_accepts_evaluator_output_plus_cli_annotations():
    data = _evaluator_like_metrics()
    # anotações que cli.py adiciona depois:
    data["config_changed"] = True
    data["attack_count_ratio_vs_baseline"] = 0.98
    data["degenerate_variant"] = False

    metrics = Metrics.model_validate(data)

    assert metrics.f1_score_attack == 0.925
    assert metrics.tp == 270
    assert metrics.class_mapping == {0: "normal", 1: "masquerade_fake_fault"}
    assert metrics.top_feature_importances[0].feature == "cbStatus"
    assert metrics.degenerate_variant is False


def test_metrics_accepts_skipped_shape_with_nones():
    """``build_skipped_metrics`` de cli.py: quase tudo None + full_* counts."""
    skipped = {
        "evaluation_type": "skipped_no_config_change",
        "dataset_path": None,
        "accuracy": None,
        "precision_attack": None,
        "recall_attack": None,
        "f1_score_attack": None,
        "support_attack": None,
        "attack_count": None,
        "normal_count": None,
        "full_attack_count": None,
        "full_normal_count": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
        "attack_label_original": "masquerade_fake_fault",
        "label_column": "class",
        "dataset_rows": None,
        "dataset_columns": None,
        "config_changed": False,
        "attack_count_ratio_vs_baseline": None,
        "degenerate_variant": None,
    }

    metrics = Metrics.model_validate(skipped)

    assert metrics.evaluation_type == "skipped_no_config_change"
    assert metrics.f1_score_attack is None
    assert metrics.config_changed is False


# --------------------------------------------------------------------------- #
# IterationRecord                                                              #
# --------------------------------------------------------------------------- #
def test_iteration_record_stitches_schemas():
    raw_attack = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))

    record = IterationRecord(
        iteration=1,
        attack_config=AttackConfig.model_validate(raw_attack),
        strategist_output=StrategistOutput(
            persona="aggressive",
            changes=[],
            reasoning="teste de costura de schemas",
        ),
        metrics=Metrics.model_validate(_evaluator_like_metrics()),
        analyst_output=None,  # AnalystOutput já existe (#5)
    )

    assert record.iteration == 1
    assert record.attack_config["fault"]["prob"] == 0.6
    assert record.metrics.f1_score_attack == 0.925
    assert record.timestamp  # default_factory preencheu ISO-8601

    # round-trip serializável para persistência (json_io.save_json usa json.dump)
    dumped = record.model_dump(mode="json")
    assert json.dumps(dumped)  # não levanta


def test_iteration_record_rejects_negative_iteration():
    with pytest.raises(ValidationError):
        IterationRecord(
            iteration=-1,
            attack_config=AttackConfig.model_validate(
                json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))
            ),
            metrics=Metrics(),
        )
