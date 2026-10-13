"""Testes das métricas e do plano de seeds dos experimentos (sem Java)."""

from __future__ import annotations

import numpy as np
import pytest

from adversarial_ids.experiments.metrics import (
    bootstrap_mean,
    calibrate_threshold,
    counts_at,
    paired_difference,
    rates,
    run_summary,
)
from adversarial_ids.experiments.protocol import SeedPlan


def test_threshold_respects_the_alert_budget_on_normal_traffic():
    normal = np.linspace(0.0, 1.0, 10_000)
    point = calibrate_threshold(normal, budget_per_10k=10.0)
    assert (normal >= point.threshold).sum() <= 10
    assert point.calibration_alert_rate <= 10.0


def test_zero_budget_means_no_alarm_on_calibration():
    normal = np.array([0.1, 0.5, 0.9])
    point = calibrate_threshold(normal, budget_per_10k=1.0)
    assert (normal >= point.threshold).sum() == 0


def test_ties_at_the_top_are_reported_not_hidden():
    # Scores discretos (RF): o orçamento pode não ser atingível exatamente.
    normal = np.array([0.0] * 9990 + [0.3] * 10)
    point = calibrate_threshold(normal, budget_per_10k=100.0)
    assert point.calibration_alert_rate <= 100.0


def test_counts_and_rates():
    is_attack = np.array([1, 1, 0, 0], dtype=bool)
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    counts = counts_at(is_attack, scores, 0.5)
    assert counts == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    r = rates(counts)
    assert r["recall"] == 0.5 and r["evasion"] == 0.5
    assert r["false_alerts_per_10k_normal"] == 5000.0


def test_run_summary_reports_every_budget():
    is_attack = np.array([1, 0, 0, 1], dtype=bool)
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    points = [calibrate_threshold(scores[~is_attack], b) for b in (1.0, 100.0)]
    summary = run_summary(is_attack, scores, scores > 0.5, points)
    assert summary["ap"] == 1.0
    assert set(summary["budgets"]) == {"1.0", "100.0"}


def test_paired_difference_detects_a_consistent_shift():
    first = [0.5, 0.6, 0.55, 0.65, 0.58]
    second = [0.1, 0.2, 0.15, 0.25, 0.18]
    interval = paired_difference(first, second)
    assert interval.excludes_zero and interval.value == pytest.approx(0.4)


def test_paired_difference_refuses_misaligned_units():
    with pytest.raises(ValueError, match="Pareamento"):
        paired_difference([0.1, 0.2], [0.1])


def test_bootstrap_of_a_constant_has_no_width():
    interval = bootstrap_mean([0.3, 0.3, 0.3])
    assert interval.low == interval.high == 0.3


def test_seed_plan_refuses_overlapping_sets():
    with pytest.raises(ValueError, match="disjuntos"):
        SeedPlan(train=(1, 2), calibration=(2,), search=(3,), test=(4,))
