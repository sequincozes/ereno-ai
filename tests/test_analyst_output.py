"""Testes do contrato AnalystOutput."""

import pytest
from pydantic import ValidationError

from adversarial_ids.domain import AnalystOutput
from tests.fakes import FakeAnalyst


def valid_output() -> dict:
    return {
        "iteration": 1,
        "deceptive_features": [
            {
                "feature": "vsbBTrapAreaSum",
                "importance": 0.42,
                "explanation": (
                    "A feature apresentou influência relevante "
                    "na decisão do Random Forest."
                ),
            }
        ],
        "diagnosis": (
            "O IDS apresentou redução no desempenho de detecção "
            "da classe masquerade."
        ),
        "mitigations": [
            {
                "type": "threshold",
                "recommendation": (
                    "Revisar o limiar de decisão utilizado "
                    "pelo classificador."
                ),
            }
        ],
        "severity": "medium",
    }


def test_analyst_output_accepts_valid_data():
    output = AnalystOutput.model_validate(valid_output())

    assert output.iteration == 1
    assert output.severity == "medium"
    assert output.deceptive_features[0].feature == "vsbBTrapAreaSum"
    assert output.mitigations[0].type == "threshold"


def test_analyst_output_rejects_invalid_severity():
    data = valid_output()
    data["severity"] = "critical"

    with pytest.raises(ValidationError):
        AnalystOutput.model_validate(data)


def test_analyst_output_rejects_invalid_mitigation_type():
    data = valid_output()
    data["mitigations"][0]["type"] = "delete_model"

    with pytest.raises(ValidationError):
        AnalystOutput.model_validate(data)


def test_analyst_output_rejects_negative_importance():
    data = valid_output()
    data["deceptive_features"][0]["importance"] = -0.2

    with pytest.raises(ValidationError):
        AnalystOutput.model_validate(data)


def test_fake_analyst_matches_contract():
    fake = FakeAnalyst()

    result = fake.analyze(
        iteration=1,
        metrics={
            "f1_score_attack": 0.65,
            "top_feature_importances": [
                {
                    "feature": "vsbBTrapAreaSum",
                    "importance": 0.42,
                }
            ],
        },
    )

    output = AnalystOutput.model_validate(result)

    assert output.iteration == 1
    assert output.severity == "medium"
    assert output.deceptive_features[0].feature == "vsbBTrapAreaSum"
