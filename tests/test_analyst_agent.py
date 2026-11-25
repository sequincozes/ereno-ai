"""Testes do agente Analista sem chamadas externas."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from adversarial_ids.agents.analyst import AnalystAgent
from adversarial_ids.domain import AnalystOutput


class FakeAgnoAgent:
    """Simula o método run() do Agno."""

    def __init__(self, content: Any) -> None:
        self.content = content
        self.last_prompt: str | None = None
        self.call_count = 0

    def run(self, prompt: str) -> SimpleNamespace:
        self.last_prompt = prompt
        self.call_count += 1

        return SimpleNamespace(content=self.content)


def metrics_data() -> dict[str, Any]:
    return {
        "accuracy": 0.75,
        "precision_attack": 0.70,
        "recall_attack": 0.60,
        "f1_score_attack": 0.65,
        "tp": 12,
        "fp": 4,
        "fn": 8,
        "tn": 20,
        "attack_count": 20,
        "normal_count": 24,
        "degenerate_variant": False,
        "top_feature_importances": [
            {
                "feature": "TrapAreaSum",
                "importance": 0.42,
            },
            {
                "feature": "cbStatusDiff",
                "importance": 0.30,
            },
        ],
    }


def valid_output(
    *,
    iteration: int = 3,
    feature: str = "TrapAreaSum",
    importance: float = 0.42,
    severity: str = "medium",
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "deceptive_features": [
            {
                "feature": feature,
                "importance": importance,
                "explanation": (
                    "A feature apresentou influência relevante "
                    "na decisão do classificador."
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
                    "Revisar e calibrar o limiar de decisão "
                    "do classificador."
                ),
            }
        ],
        "severity": severity,
    }


def test_build_prompt_contains_metrics_and_evidence():
    fake = FakeAgnoAgent(valid_output())
    analyst = AnalystAgent(agent=fake)

    prompt = analyst.build_prompt(
        iteration=3,
        metrics=metrics_data(),
    )

    assert '"iteration": 3' in prompt
    assert '"f1_score_attack": 0.65' in prompt
    assert '"required_severity": "medium"' in prompt
    assert '"feature": "TrapAreaSum"' in prompt


def test_analyze_accepts_dictionary_content():
    fake = FakeAgnoAgent(valid_output())
    analyst = AnalystAgent(agent=fake)

    result = analyst.analyze(
        iteration=3,
        metrics=metrics_data(),
    )

    assert isinstance(result, AnalystOutput)
    assert result.iteration == 3
    assert result.severity == "medium"
    assert fake.call_count == 1
    assert fake.last_prompt is not None


def test_analyze_accepts_json_string_content():
    fake = FakeAgnoAgent(
        json.dumps(valid_output(), ensure_ascii=False)
    )
    analyst = AnalystAgent(agent=fake)

    result = analyst.analyze(
        iteration=3,
        metrics=metrics_data(),
    )

    assert result.iteration == 3
    assert result.deceptive_features[0].feature == "TrapAreaSum"


def test_analyze_accepts_pydantic_content():
    output = AnalystOutput.model_validate(valid_output())

    fake = FakeAgnoAgent(output)
    analyst = AnalystAgent(agent=fake)

    result = analyst.analyze(
        iteration=3,
        metrics=metrics_data(),
    )

    assert result == output


def test_analyze_rejects_invented_feature():
    fake = FakeAgnoAgent(
        valid_output(
            feature="featureInventada",
            importance=0.42,
        )
    )
    analyst = AnalystAgent(agent=fake)

    with pytest.raises(ValueError, match="Feature inventada"):
        analyst.analyze(
            iteration=3,
            metrics=metrics_data(),
        )


def test_analyze_rejects_changed_importance():
    fake = FakeAgnoAgent(
        valid_output(
            feature="TrapAreaSum",
            importance=0.99,
        )
    )
    analyst = AnalystAgent(agent=fake)

    with pytest.raises(ValueError, match="Importância incorreta"):
        analyst.analyze(
            iteration=3,
            metrics=metrics_data(),
        )


def test_analyze_rejects_wrong_iteration():
    fake = FakeAgnoAgent(
        valid_output(iteration=99)
    )
    analyst = AnalystAgent(agent=fake)

    with pytest.raises(
        ValueError,
        match="iteração retornada",
    ):
        analyst.analyze(
            iteration=3,
            metrics=metrics_data(),
        )


def test_analyze_rejects_wrong_severity():
    fake = FakeAgnoAgent(
        valid_output(severity="high")
    )
    analyst = AnalystAgent(agent=fake)

    with pytest.raises(
        ValueError,
        match="Severidade incompatível",
    ):
        analyst.analyze(
            iteration=3,
            metrics=metrics_data(),
        )


def test_build_prompt_rejects_negative_iteration():
    fake = FakeAgnoAgent(valid_output())
    analyst = AnalystAgent(agent=fake)

    with pytest.raises(
        ValueError,
        match="iteração não pode ser negativa",
    ):
        analyst.build_prompt(
            iteration=-1,
            metrics=metrics_data(),
        )
