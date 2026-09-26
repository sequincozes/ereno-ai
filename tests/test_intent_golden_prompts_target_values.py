"""Golden prompts com valores ditados pelo pedido (``target_values``).

Espelha ``test_intent_golden_prompts_multi_attack.py`` (mesmo ``FakeIntentAgent``,
mesmos dois portões determinísticos) sobre ``target_value_intents.json`` — cinco
pedidos válidos que fixam valores (intervalo, escalar, enum + booleano, lista,
um limite de par só) e cinco recusados, um por decisão de validação: fora da
faixa, par invertido, cota estourada, enum inventado, caminho fora da allowlist.

Fixa em código que um número ditado atravessa a cadeia sem depender da Groq — e
que os pedidos impossíveis nunca chegam a um candidato compilado.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_ids.agents.intent.agent import IntentCompilationError
from tests.fakes.fake_intent_agent import FakeIntentAgent

FIXTURE_PATH = Path(__file__).parent / "golden_prompts" / "target_value_intents.json"
ENTRIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

VALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] == "valid"]
INVALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] == "invalid"]


def test_fixture_has_no_duplicate_ids():
    assert len({entry["id"] for entry in ENTRIES}) == len(ENTRIES)


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=[e["id"] for e in VALID_ENTRIES])
def test_valid_pinned_prompts_compile_to_an_authorized_intent(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    assert intent.source_prompt == entry["prompt"]
    pinned = {t.path: t.value for t in intent.restrictions.target_values}
    for path, value in entry["payload"]["target_values"].items():
        expected = tuple(value) if isinstance(value, list) else value
        assert pinned[path] == expected


@pytest.mark.parametrize("entry", INVALID_ENTRIES, ids=[e["id"] for e in INVALID_ENTRIES])
def test_impossible_pinned_prompts_never_reach_a_compiled_intent(entry):
    with pytest.raises((IntentCompilationError, ValueError), match=entry["expected_error"]):
        FakeIntentAgent.compile(entry["payload"], entry["prompt"])
