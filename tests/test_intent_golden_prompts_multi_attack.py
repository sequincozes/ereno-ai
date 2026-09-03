"""Golden prompts multi-ataque: trava cada decisão de exclusão do catálogo.

Espelha ``test_intent_golden_prompts.py`` (mesmo ``FakeIntentAgent``, mesmos
dois portões determinísticos) sobre ``multi_attack_intents.json`` — 9 prompts
válidos (um por ataque novo) e um conjunto de adversariais que fixam em código
as exclusões documentadas em ``config/capabilities/*.py``: ``orderBy`` sem
enum, ``randomSeed`` como controle de reprodutibilidade, as faixas mortas do
``injection`` em modo ``"random"``, o gate ``protectStatusChanges`` do
``grayhole``, ``increase_resource_pressure`` em qualquer ataque, o teto
estreito do catálogo de 2 campos do ``injection``, e vazamento de caminho
entre ataques.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_ids.agents.intent.agent import IntentCompilationError
from tests.fakes.fake_intent_agent import FakeIntentAgent

FIXTURE_PATH = Path(__file__).parent / "golden_prompts" / "multi_attack_intents.json"
ENTRIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

VALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] == "valid"]
INVALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] == "invalid"]


def test_fixture_has_no_duplicate_ids():
    assert len({entry["id"] for entry in ENTRIES}) == len(ENTRIES)


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=[e["id"] for e in VALID_ENTRIES])
def test_valid_multi_attack_prompts_compile_to_an_authorized_intent(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    assert intent.source_prompt == entry["prompt"]
    assert intent.base_attack == entry["payload"]["base_attack"]
    assert intent.objective.value == entry["payload"]["objective"]
    assert intent.desired_effect.value == entry["payload"]["desired_effect"]


@pytest.mark.parametrize("entry", INVALID_ENTRIES, ids=[e["id"] for e in INVALID_ENTRIES])
def test_invalid_multi_attack_prompts_never_reach_a_compiled_intent(entry):
    with pytest.raises(IntentCompilationError, match=entry["expected_error"]):
        FakeIntentAgent.compile(entry["payload"], entry["prompt"])
