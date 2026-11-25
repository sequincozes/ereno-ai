"""Testes de build_detection_report — traduz o dict do IdsEvaluator (E3/E4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.core.detection_reporter import (
    DetectionReportError,
    build_detection_report,
)
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain.detection_report import DetectionReport


def _labeled_csv(path: Path, *, attack_label: str = "attack_label") -> Path:
    rows = ["f1,f2,class"]
    for i in range(30):
        rows.append(f"{i % 5},{i % 3},normal")
        rows.append(f"{100 + i % 5},{10 + i % 3},{attack_label}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _trained_evaluator(tmp_path: Path) -> IdsEvaluator:
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = IdsEvaluator(drop_cb_status=False, target_attack_label="attack_label")
    evaluator.train_baseline(str(dataset))
    return evaluator


def test_build_detection_report_wraps_evaluator_output(tmp_path):
    evaluator = _trained_evaluator(tmp_path)
    variant = _labeled_csv(tmp_path / "variant.csv")

    report = build_detection_report(evaluator, str(variant), split="train_test_70_30_seed42")

    assert isinstance(report, DetectionReport)
    assert report.model_name == "random_forest"
    assert report.split == "train_test_70_30_seed42"
    assert 0.0 <= report.accuracy <= 1.0
    assert report.confusion_matrix.tp >= 0
    assert report.latency_ms >= 0.0


def test_build_detection_report_uses_custom_model_name(tmp_path):
    evaluator = _trained_evaluator(tmp_path)
    variant = _labeled_csv(tmp_path / "variant.csv")

    report = build_detection_report(
        evaluator, str(variant), split="s", model_name="decision_tree"
    )

    assert report.model_name == "decision_tree"


def test_build_detection_report_rejects_non_binary_confusion_matrix(tmp_path, monkeypatch):
    evaluator = _trained_evaluator(tmp_path)
    variant = _labeled_csv(tmp_path / "variant.csv")

    def _fake_evaluate_variant(dataset_path: str):
        return {
            "accuracy": 0.9,
            "precision_attack": 0.8,
            "recall_attack": 0.7,
            "f1_score_attack": 0.75,
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
            "top_feature_importances": [],
        }

    monkeypatch.setattr(evaluator, "evaluate_variant", _fake_evaluate_variant)

    with pytest.raises(DetectionReportError, match="matriz de confusão binária"):
        build_detection_report(evaluator, str(variant), split="s")
