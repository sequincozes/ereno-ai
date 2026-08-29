"""Testes do prompt do agente Analista."""

from pathlib import Path


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "adversarial_ids"
    / "prompts"
    / "analyst.md"
)


def test_analyst_prompt_exists():
    assert PROMPT_PATH.exists()


def test_analyst_prompt_is_not_empty():
    content = PROMPT_PATH.read_text(encoding="utf-8")

    assert content.strip()
    assert len(content) > 500


def test_analyst_prompt_contains_required_rules():
    content = PROMPT_PATH.read_text(encoding="utf-8").lower()

    required_terms = [
        "analystoutput",
        "não invente nomes de features",
        "não altere valores numéricos",
        "falsos negativos",
        "threshold",
        "feature",
        "retrain",
        "required_severity",
    ]

    for term in required_terms:
        assert term in content


def test_analyst_prompt_contains_severity_thresholds():
    content = PROMPT_PATH.read_text(encoding="utf-8")

    assert "0.50" in content
    assert "0.80" in content
    assert "`high`" in content
    assert "`medium`" in content
    assert "`low`" in content
