"""Evidência do Defensor no E1 (sem Java): contagens no ponto primário + SHAP das que evadiram."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from adversarial_ids.config.settings import BASELINE_DATASET_PATH
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.experiments.defense_evidence import (
    ALL_ATTACK_POPULATION,
    MISSED_POPULATION,
    EvidenceError,
    candidate_report,
)
from adversarial_ids.experiments.metrics import OperatingPoint
from adversarial_ids.experiments.protocol import FrozenIds

pytestmark = [
    pytest.mark.skipif(not BASELINE_DATASET_PATH.exists(), reason="data/baseline_dataset.csv não versionado"),
    pytest.mark.skipif(importlib.util.find_spec("shap") is None, reason="extra 'shap' não instalado"),
]


@dataclass
class _SameTraceStore:
    """Toda (config, seed) aponta para o mesmo CSV — basta para testar a contabilidade."""

    path: Path

    def trace(self, attack_key, config, seed):  # noqa: ARG002
        return self.path


@pytest.fixture(scope="module")
def evaluator() -> IdsEvaluator:
    model = IdsEvaluator(n_estimators=20, random_state=42)
    model.fit_traces([str(BASELINE_DATASET_PATH)])
    return model


def _frozen(evaluator: IdsEvaluator, threshold: float) -> FrozenIds:
    frozen = FrozenIds(
        attack_key="masquerade_fault",
        reference_config={},
        store=_SameTraceStore(BASELINE_DATASET_PATH),  # type: ignore[arg-type]
    )
    frozen.evaluator = evaluator
    frozen.points = (OperatingPoint(10.0, threshold, 0.0),)
    return frozen


def test_counts_are_summed_over_seeds_and_metrics_follow_them(evaluator):
    one = candidate_report(_frozen(evaluator, 0.5), {}, [301])
    two = candidate_report(_frozen(evaluator, 0.5), {}, [301, 302])
    for key in ("tp", "fp", "fn", "tn"):
        assert getattr(two.confusion_matrix, key) == 2 * getattr(one.confusion_matrix, key)
    cm = two.confusion_matrix
    assert two.recall == pytest.approx(cm.tp / (cm.tp + cm.fn))
    assert two.precision == pytest.approx(cm.tp / (cm.tp + cm.fp))


def test_explains_the_missed_attack_messages_when_some_evade(evaluator):
    # Limiar acima de qualquer score: nenhuma mensagem dispara, todo ataque evade.
    report = candidate_report(_frozen(evaluator, np.inf), {}, [301])
    assert report.recall == 0.0
    assert report.split.endswith(f"shap={MISSED_POPULATION}")
    assert report.top_features
    importances = [f.importance for f in report.top_features]
    assert importances == sorted(importances, reverse=True)


def test_falls_back_to_all_attack_messages_when_nothing_evades(evaluator):
    # Limiar abaixo de qualquer score: todo ataque é pego, não há falso negativo.
    report = candidate_report(_frozen(evaluator, -np.inf), {}, [301])
    assert report.confusion_matrix.fn == 0
    assert report.split.endswith(f"shap={ALL_ATTACK_POPULATION}")
    assert report.top_features


def test_missing_shap_fails_loudly_instead_of_an_empty_report(evaluator, monkeypatch):
    monkeypatch.setattr(evaluator, "get_shap_importances_for_frame", lambda *a, **k: [])
    with pytest.raises(EvidenceError):
        candidate_report(_frozen(evaluator, np.inf), {}, [301])


def test_unknown_budget_is_an_error(evaluator):
    with pytest.raises(ValueError):
        candidate_report(_frozen(evaluator, 0.5), {}, [301], budget_per_10k=1.0)

