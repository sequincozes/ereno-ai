"""Testes do prompt do agente Defensor (épico E5)."""

from pathlib import Path


PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "adversarial_ids"
    / "prompts"
    / "defender.md"
)


def test_defender_prompt_exists():
    assert PROMPT_PATH.exists()


def test_defender_prompt_is_not_empty():
    content = PROMPT_PATH.read_text(encoding="utf-8")

    assert content.strip()
    assert len(content) > 500


def test_defender_prompt_contains_required_terms():
    content = PROMPT_PATH.read_text(encoding="utf-8").lower()

    required_terms = [
        "defenseplan",
        "citable_evidence",
        "required_priority",
        "validation_method",
        "detection_report_ref",
        "não invente",
        "detection_actions",
        "containment_actions",
        "hardening_actions",
        "falsos negativos",
        "nunca é executado automaticamente",
    ]

    for term in required_terms:
        assert term in content, term


def test_defender_prompt_contains_priority_thresholds():
    content = PROMPT_PATH.read_text(encoding="utf-8")

    assert "0.50" in content
    assert "0.80" in content
    assert "`low`" in content
    assert "`medium`" in content
    assert "`high`" in content
    assert "`critical`" in content
