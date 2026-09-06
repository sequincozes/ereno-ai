"""Testes do agente Defensor sem chamadas externas (épico E5)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from adversarial_ids.agents.defender.agent import DefenderAgent
from adversarial_ids.agents.defender.tools import priority_from_report
from adversarial_ids.agents.orchestrator.intent_loop import DefenderLike
from adversarial_ids.domain import DefensePlan, DetectionReport


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


def report_data() -> dict[str, Any]:
    return {
        "model_name": "random_forest",
        "split": "train_test_80_20_seed42",
        "accuracy": 0.75,
        "precision": 0.70,
        "recall": 0.60,
        "f1": 0.65,
        "confusion_matrix": {"tp": 12, "fp": 4, "fn": 8, "tn": 20},
        "top_features": [
            {"feature": "TrapAreaSum", "importance": 0.42},
            {"feature": "cbStatusDiff", "importance": 0.30},
        ],
        "latency_ms": 3.5,
    }


def valid_plan(
    *,
    priority: str | None = None,
    metric_or_feature: str = "recall",
    value: float | int | str = 0.60,
) -> dict[str, Any]:
    report = DetectionReport.model_validate(report_data())
    return {
        "priority": priority or priority_from_report(report),
        "detection_actions": [
            {
                "description": "Revisar o limiar de decisão do classificador.",
                "technique": "detector_threshold_tuning",
                "evidence": [
                    {
                        "metric_or_feature": metric_or_feature,
                        "value": value,
                        "detection_report_ref": None,
                    }
                ],
                "validation_test": {
                    "metric": "recall",
                    "direction": "increase",
                    "target": 0.80,
                    "procedure": "Reavaliar o recall no mesmo split de teste.",
                },
            }
        ],
    }


def test_build_prompt_contains_metrics_evidence_and_ref():
    fake = FakeAgnoAgent(valid_plan())
    defender = DefenderAgent(agent=fake)

    prompt = defender.build_prompt(
        report_data(), report_ref="outputs/intent_loop/run-1/detection_report.json"
    )

    assert '"recall": 0.6' in prompt
    assert '"confusion_matrix.fn": 8' in prompt
    assert '"TrapAreaSum": 0.42' in prompt
    assert '"required_priority": "medium"' in prompt
    assert '"outputs/intent_loop/run-1/detection_report.json"' in prompt


def test_defend_accepts_dictionary_content():
    fake = FakeAgnoAgent(valid_plan())
    defender = DefenderAgent(agent=fake)

    result = defender.defend(report_data())

    assert isinstance(result, DefensePlan)
    assert fake.call_count == 1
    assert fake.last_prompt is not None


def test_defend_accepts_json_string_content():
    fake = FakeAgnoAgent(json.dumps(valid_plan()))
    defender = DefenderAgent(agent=fake)

    result = defender.defend(report_data())

    assert result.detection_actions[0].evidence[0].metric_or_feature == "recall"


def test_defend_accepts_fenced_json_string_content():
    fake = FakeAgnoAgent(f"```json\n{json.dumps(valid_plan())}\n```")
    defender = DefenderAgent(agent=fake)

    result = defender.defend(report_data())

    assert result.priority == priority_from_report(
        DetectionReport.model_validate(report_data())
    )


def test_defend_accepts_pydantic_content():
    plan = DefensePlan.model_validate(valid_plan())
    fake = FakeAgnoAgent(plan)
    defender = DefenderAgent(agent=fake)

    result = defender.defend(report_data())

    assert result == plan


def test_defend_rejects_invented_evidence():
    fake = FakeAgnoAgent(valid_plan(metric_or_feature="feature_inventada", value=0.5))
    defender = DefenderAgent(agent=fake)

    with pytest.raises(ValueError, match="Evidência inventada"):
        defender.defend(report_data())


def test_defend_rejects_wrong_priority():
    fake = FakeAgnoAgent(valid_plan(priority="critical"))
    defender = DefenderAgent(agent=fake)

    with pytest.raises(ValueError, match="Prioridade incompatível"):
        defender.defend(report_data())


def test_defend_round_trips_a_matching_report_ref():
    ref = "outputs/intent_loop/run-1/detection_report.json"
    plan = valid_plan()
    plan["detection_actions"][0]["evidence"][0]["detection_report_ref"] = ref
    fake = FakeAgnoAgent(plan)
    defender = DefenderAgent(agent=fake)

    result = defender.defend(report_data(), report_ref=ref)

    assert result.detection_actions[0].evidence[0].detection_report_ref == ref


def test_defend_rejects_mismatched_report_ref():
    plan = valid_plan()
    plan["detection_actions"][0]["evidence"][0]["detection_report_ref"] = "outro/relatorio.json"
    fake = FakeAgnoAgent(plan)
    defender = DefenderAgent(agent=fake)

    with pytest.raises(ValueError, match="Referência de DetectionReport inválida"):
        defender.defend(
            report_data(), report_ref="outputs/intent_loop/run-1/detection_report.json"
        )


def test_defend_raises_when_response_is_none():
    class NoneAgent:
        def run(self, prompt: str) -> None:
            return None

    defender = DefenderAgent(agent=NoneAgent())

    with pytest.raises(RuntimeError, match="não retornou uma resposta"):
        defender.defend(report_data())


def test_defend_raises_when_content_is_none():
    fake = FakeAgnoAgent(None)
    defender = DefenderAgent(agent=fake)

    with pytest.raises(RuntimeError, match="não possui conteúdo"):
        defender.defend(report_data())


def test_constructor_raises_for_missing_prompt_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prompt do Defensor"):
        DefenderAgent(agent=FakeAgnoAgent(valid_plan()), prompt_path=tmp_path / "missing.md")


def test_constructor_raises_for_empty_prompt_file(tmp_path):
    empty_prompt = tmp_path / "empty.md"
    empty_prompt.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError, match="O prompt do Defensor está vazio"):
        DefenderAgent(agent=FakeAgnoAgent(valid_plan()), prompt_path=empty_prompt)


def test_defender_agent_satisfies_the_defenderlike_protocol():
    defender = DefenderAgent(agent=FakeAgnoAgent(valid_plan()))
    assert isinstance(defender, DefenderLike)


def test_build_prompt_lists_the_legal_techniques_per_bucket():
    """A LLM não pode adivinhar o vocabulário: ele chega junto com os dados."""

    import json

    from adversarial_ids.domain import techniques_for_bucket

    defender = DefenderAgent(agent=FakeAgnoAgent(valid_plan()))

    prompt = defender.build_prompt(report_data())
    payload = json.loads(prompt.split("## Dados da execução", 1)[1].strip())

    assert payload["legal_techniques"] == {
        bucket: list(techniques_for_bucket(bucket))
        for bucket in ("detection_actions", "containment_actions", "hardening_actions")
    }


def test_build_prompt_lists_the_metrics_a_validation_test_may_use():
    import json

    from adversarial_ids.domain import VALIDATION_METRICS

    defender = DefenderAgent(agent=FakeAgnoAgent(valid_plan()))

    prompt = defender.build_prompt(report_data())
    payload = json.loads(prompt.split("## Dados da execução", 1)[1].strip())

    assert payload["validation_metrics"] == list(VALIDATION_METRICS)
    # Importância de feature não é resultado defensivo, então fica de fora
    # mesmo estando em `citable_evidence`.
    assert "TrapAreaSum" in payload["citable_evidence"]
    assert "TrapAreaSum" not in payload["validation_metrics"]
