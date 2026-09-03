"""Golden prompts do IntentAgent (ação 72h #4 do plano de 60 dias).

10 prompts válidos + 5 ambíguos (devem resolver para uma intenção válida e
conservadora, sem exigir esclarecimento) + 10 inválidos/adversariais (devem
ser bloqueados antes de alcançar o compilador/JAR do ERENO).

Os ``payload`` fixam o que um compilador real produziria a partir de cada
prompt — isso testa a cadeia completa de validação (tool -> IntentSpec ->
validate_intent_capability) sem depender da API da Groq, mantendo o gate
determinístico e reproduzível em CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_ids.agents.intent.agent import IntentCompilationError
from tests.fakes.fake_intent_agent import FakeIntentAgent

FIXTURE_PATH = Path(__file__).parent / "golden_prompts" / "masquerade_fault_intents.json"
ENTRIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

VALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] in ("valid", "ambiguous")]
INVALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] == "invalid"]


def test_golden_fixture_has_the_prescribed_composition():
    by_category = {
        category: len([e for e in ENTRIES if e["category"] == category])
        for category in ("valid", "ambiguous", "invalid")
    }
    assert by_category == {"valid": 10, "ambiguous": 5, "invalid": 10}
    assert len({entry["id"] for entry in ENTRIES}) == len(ENTRIES)


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=[e["id"] for e in VALID_ENTRIES])
def test_valid_and_ambiguous_prompts_compile_to_an_authorized_intent(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    assert intent.source_prompt == entry["prompt"]
    assert intent.base_attack == entry["payload"]["base_attack"]
    assert intent.objective.value == entry["payload"]["objective"]
    assert intent.desired_effect.value == entry["payload"]["desired_effect"]


@pytest.mark.parametrize("entry", INVALID_ENTRIES, ids=[e["id"] for e in INVALID_ENTRIES])
def test_invalid_prompts_never_reach_a_compiled_intent(entry):
    with pytest.raises(IntentCompilationError, match=entry["expected_error"]):
        FakeIntentAgent.compile(entry["payload"], entry["prompt"])
