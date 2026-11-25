"""Testes das ferramentas determinísticas do Analista."""

import pytest

from adversarial_ids.agents.analyst.tools import (
    select_feature_importances,
    severity_from_f1,
    validate_output_against_metrics,
)
from adversarial_ids.domain import AnalystOutput, Metrics


def build_metrics(
    f1_score: float | None = 0.65,
) -> Metrics:
    """Cria métricas mínimas para os testes."""

    return Metrics.model_validate(
        {
            "f1_score_attack": f1_score,
            "precision_attack": 0.70,
            "recall_attack": 0.60,
            "tp": 12,
            "fp": 4,
            "fn": 8,
            "tn": 20,
            "top_feature_importances": [
                {
                    "feature": "TrapAreaSum",
                    "importance": 0.42,
                },
                {
                    "feature": "cbStatusDiff",
                    "importance": 0.30,
                },
                {
                    "feature": "analogDelta",
                    "importance": 0.18,
                },
            ],
        }
    )


def build_output(
    *,
    feature: str = "TrapAreaSum",
    importance: float = 0.42,
    severity: str = "medium",
) -> AnalystOutput:
    """Cria uma saída válida do Analista."""

    return AnalystOutput.model_validate(
        {
            "iteration": 1,
            "deceptive_features": [
                {
                    "feature": feature,
                    "importance": importance,
                    "explanation": (
                        "A feature apresentou influência relevante "
                        "na decisão do Random Forest."
                    ),
                }
            ],
            "diagnosis": (
                "O IDS apresentou redução na capacidade de detectar "
                "amostras da classe masquerade."
            ),
            "mitigations": [
                {
                    "type": "threshold",
                    "recommendation": (
                        "Revisar o limiar de decisão do classificador."
                    ),
                }
            ],
            "severity": severity,
        }
    )


@pytest.mark.parametrize(
    ("f1_score", "expected"),
    [
        (None, "low"),
        (0.95, "low"),
        (0.80, "low"),
        (0.79, "medium"),
        (0.50, "medium"),
        (0.49, "high"),
        (0.10, "high"),
    ],
)
def test_severity_from_f1(
    f1_score: float | None,
    expected: str,
):
    assert severity_from_f1(f1_score) == expected


def test_select_importances_uses_gini_as_fallback():
    selected = select_feature_importances(
        metrics=build_metrics(),
        top_n=2,
    )

    assert len(selected) == 2
    assert selected[0]["feature"] == "TrapAreaSum"
    assert selected[1]["feature"] == "cbStatusDiff"


def test_select_importances_prioritizes_shap():
    shap_importances = [
        {
            "feature": "shapFeature",
            "importance": 0.75,
        },
        {
            "feature": "secondShapFeature",
            "importance": 0.25,
        },
    ]

    selected = select_feature_importances(
        metrics=build_metrics(),
        shap_importances=shap_importances,
    )

    assert selected[0]["feature"] == "shapFeature"
    assert selected[0]["importance"] == 0.75
    assert all(
        item["feature"] != "TrapAreaSum"
        for item in selected
    )


def test_select_importances_orders_values():
    shap_importances = [
        {
            "feature": "menor",
            "importance": 0.10,
        },
        {
            "feature": "maior",
            "importance": 0.90,
        },
    ]

    selected = select_feature_importances(
        metrics=build_metrics(),
        shap_importances=shap_importances,
    )

    assert selected[0]["feature"] == "maior"
    assert selected[1]["feature"] == "menor"


def test_validate_accepts_supported_output():
    output = build_output()

    validated = validate_output_against_metrics(
        output=output,
        metrics=build_metrics(),
    )

    assert validated.severity == "medium"
    assert validated.deceptive_features[0].feature == "TrapAreaSum"


def test_validate_rejects_invented_feature():
    output = build_output(
        feature="featureInventada",
        importance=0.42,
    )

    with pytest.raises(ValueError, match="Feature inventada"):
        validate_output_against_metrics(
            output=output,
            metrics=build_metrics(),
        )


def test_validate_rejects_changed_importance():
    output = build_output(
        feature="TrapAreaSum",
        importance=0.99,
    )

    with pytest.raises(ValueError, match="Importância incorreta"):
        validate_output_against_metrics(
            output=output,
            metrics=build_metrics(),
        )


def test_validate_rejects_wrong_severity():
    output = build_output(severity="high")

    with pytest.raises(ValueError, match="Severidade incompatível"):
        validate_output_against_metrics(
            output=output,
            metrics=build_metrics(),
        )


def test_validate_uses_shap_evidence():
    shap_importances = [
        {
            "feature": "shapFeature",
            "importance": 0.73,
        }
    ]

    output = build_output(
        feature="shapFeature",
        importance=0.73,
    )

    validated = validate_output_against_metrics(
        output=output,
        metrics=build_metrics(),
        shap_importances=shap_importances,
    )

    assert validated.deceptive_features[0].feature == "shapFeature"
