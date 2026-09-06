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
        "validation_test",
        "legal_techniques",
        "validation_metrics",
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


def test_defender_prompt_example_is_a_valid_defense_plan():
    """O exemplo do prompt é a especificação de formato que a LLM copia.

    Um exemplo que não valida contra o contrato ensina o modelo a produzir
    exatamente a resposta que o portão vai recusar — foi o que aconteceria se
    o prompt tivesse ficado na v1 (``validation_method``) depois da v2.
    """

    import json
    import re

    from adversarial_ids.domain import DefensePlan

    content = PROMPT_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"^\{$.*?^\}$", content, flags=re.DOTALL | re.MULTILINE)

    assert blocks, "O prompt precisa conter um exemplo JSON do DefensePlan."

    for block in blocks:
        DefensePlan.model_validate(json.loads(block))


def test_defender_prompt_does_not_hardcode_the_technique_vocabulary():
    """As técnicas chegam em ``legal_techniques``, montado do mapa do contrato.

    Se o prompt listasse os nomes em texto, a lista envelheceria em silêncio no
    dia em que o catálogo crescesse — e o Defensor continuaria propondo só o
    subconjunto antigo, sem nenhum teste ficando vermelho.
    """

    from adversarial_ids.domain import DEFENSE_TECHNIQUES

    content = PROMPT_PATH.read_text(encoding="utf-8")
    # O exemplo de formato cita uma técnica, e só ela, para mostrar o campo.
    quoted = {t for t in DEFENSE_TECHNIQUES if f'"{t}"' in content}

    assert quoted == {"detector_threshold_tuning"}
