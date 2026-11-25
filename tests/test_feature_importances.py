"""Testes da API estável de feature importances do ids_evaluator (issue #15).

Congelam a assinatura consumida pelo Analista (#12):

- ``get_feature_importances(top_n)`` — Gini, sempre disponível;
- ``get_shap_importances(dataset_path, top_n, max_samples)`` — best-effort;
- ``Metrics.top_feature_importances`` populado no formato ``FeatureImportance``.

Treina o RF uma única vez (fixture de módulo) sobre o seed real, com poucas
árvores para manter o teste rápido e sem Java.
"""

import importlib.util

import pytest

from adversarial_ids.config.settings import BASELINE_DATASET_PATH
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain import Metrics

pytestmark = pytest.mark.skipif(
    not BASELINE_DATASET_PATH.exists(),
    reason="data/baseline_dataset.csv não versionado",
)

SHAP_INSTALLED = importlib.util.find_spec("shap") is not None


@pytest.fixture(scope="module")
def trained_evaluator() -> IdsEvaluator:
    evaluator = IdsEvaluator(n_estimators=20, random_state=42)
    evaluator.train_baseline(str(BASELINE_DATASET_PATH))
    return evaluator


def _assert_importance_shape(ranking: list) -> None:
    assert isinstance(ranking, list)
    for item in ranking:
        # formato exato que domain.metrics.FeatureImportance aceita (extra=forbid)
        assert set(item.keys()) == {"feature", "importance"}
        assert isinstance(item["feature"], str)
        assert isinstance(item["importance"], float)


# --------------------------------------------------------------------------- #
# Gini importances — API estável                                              #
# --------------------------------------------------------------------------- #
def test_get_feature_importances_shape_and_order(trained_evaluator):
    ranking = trained_evaluator.get_feature_importances()

    assert len(ranking) > 0
    _assert_importance_shape(ranking)

    importances = [item["importance"] for item in ranking]
    assert importances == sorted(importances, reverse=True)


def test_get_feature_importances_respects_top_n(trained_evaluator):
    top3 = trained_evaluator.get_feature_importances(top_n=3)

    assert len(top3) <= 3
    # o top_n recorta o mesmo ranking, sem reordenar
    full = trained_evaluator.get_feature_importances(top_n=1000)
    assert top3 == full[:3]


def test_get_feature_importances_empty_before_training():
    assert IdsEvaluator().get_feature_importances() == []


def test_top_feature_importances_validates_as_metrics(trained_evaluator):
    metrics_dict = trained_evaluator.evaluate_variant(str(BASELINE_DATASET_PATH))

    assert metrics_dict["top_feature_importances"] == trained_evaluator.get_feature_importances()

    metrics = Metrics.model_validate(metrics_dict)
    assert len(metrics.top_feature_importances) > 0
    assert metrics.top_feature_importances[0].importance >= (
        metrics.top_feature_importances[-1].importance
    )


# --------------------------------------------------------------------------- #
# SHAP importances — best-effort ("quando possível")                          #
# --------------------------------------------------------------------------- #
def test_get_shap_importances_is_graceful(trained_evaluator):
    result = trained_evaluator.get_shap_importances(
        str(BASELINE_DATASET_PATH), top_n=5, max_samples=50
    )

    _assert_importance_shape(result)

    if not SHAP_INSTALLED:
        # sem o pacote opcional, a API não quebra — só devolve vazio.
        assert result == []


@pytest.mark.skipif(not SHAP_INSTALLED, reason="pacote 'shap' não instalado")
def test_get_shap_importances_when_available(trained_evaluator):
    result = trained_evaluator.get_shap_importances(
        str(BASELINE_DATASET_PATH), top_n=5, max_samples=50
    )

    assert 0 < len(result) <= 5
    importances = [item["importance"] for item in result]
    assert importances == sorted(importances, reverse=True)


def test_get_shap_importances_empty_before_training():
    assert IdsEvaluator().get_shap_importances(str(BASELINE_DATASET_PATH)) == []
