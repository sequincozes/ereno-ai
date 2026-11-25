"""Golden prompts do ataque programável (uc11) — a gramática das regras.

Espelha ``test_intent_golden_prompts_target_values.py`` (mesmo ``FakeIntentAgent``,
mesmos portões determinísticos) sobre ``programmable_intents.json``, com uma
diferença que é do próprio ataque: aqui existem **três** categorias, não duas.

Nos outros ataques um pedido impossível sempre morre nos portões da intenção,
porque a impossibilidade é sempre de um campo só (fora da faixa, fora do enum,
caminho inexistente). No programável um pedido pode passar os dois portões e
ainda assim ser inexecutável, porque a impossibilidade está na **combinação** de
dois campos legais da mesma regra — ``op=scale`` existe, ``field=timestamp``
existe, e escalar um relógio absoluto não é nada que se possa gerar. Quem recusa
isso é o schema, dentro do compilador. A terceira categoria (``incoherent``) é o
que fixa essa divisão de trabalho em teste, e não só em prosa:

- ``valid``      — atravessa os portões **e** compila para um candidato;
- ``invalid``    — recusado nos portões da intenção (campo a campo);
- ``incoherent`` — aprovado campo a campo, recusado pelo schema no compilador.

As cinco entradas ``valid`` com ``source: "llm"`` não foram inventadas: são os
payloads que ``openai/gpt-oss-120b`` de fato produziu em 01/10 nos quatro eixos
da gramática (ver ``docs/programmable_attack.md``). Fixá-los aqui é o que torna
a validação com LLM real repetível sem gastar a Groq a cada ``pytest``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_ids.agents.intent.agent import IntentCompilationError
from adversarial_ids.core.intent_compiler import IntentCompilerError, compile_attack_candidate
from adversarial_ids.domain.attack_configs import ProgrammableConfig
from tests.fakes.fake_intent_agent import FakeIntentAgent

FIXTURE_PATH = Path(__file__).parent / "golden_prompts" / "programmable_intents.json"
ENTRIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

VALID_ENTRIES = [e for e in ENTRIES if e["category"] == "valid"]
INVALID_ENTRIES = [e for e in ENTRIES if e["category"] == "invalid"]
INCOHERENT_ENTRIES = [e for e in ENTRIES if e["category"] == "incoherent"]


def _ids(entries):
    return [entry["id"] for entry in entries]


def test_fixture_has_no_duplicate_ids():
    assert len({entry["id"] for entry in ENTRIES}) == len(ENTRIES)


def test_every_category_is_covered():
    assert VALID_ENTRIES and INVALID_ENTRIES and INCOHERENT_ENTRIES
    assert {e["category"] for e in ENTRIES} == {"valid", "invalid", "incoherent"}


def test_every_entry_declares_where_its_payload_came_from():
    """``llm`` é payload que o modelo real produziu; ``hand`` foi escrito aqui.
    Sem a marca, um fixture inventado passaria por evidência de validação."""

    assert all(entry["source"] in {"llm", "hand"} for entry in ENTRIES)


def test_the_four_grammar_axes_came_from_the_real_model():
    from_llm = [e["id"] for e in VALID_ENTRIES if e["source"] == "llm"]
    assert len(from_llm) == 4, from_llm


# --------------------------------------------------------------------------- #
# valid — atravessa os portões e compila                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=_ids(VALID_ENTRIES))
def test_valid_prompts_compile_to_an_authorized_intent(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    assert intent.source_prompt == entry["prompt"]
    assert intent.base_attack == "programmable"
    pinned = {t.path: t.value for t in intent.restrictions.target_values}
    for path, value in entry["payload"]["target_values"].items():
        expected = tuple(value) if isinstance(value, list) else value
        assert pinned[path] == expected


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=_ids(VALID_ENTRIES))
def test_valid_prompts_reach_a_generatable_config(entry):
    """O ditado chega à config verbatim e a config é gerável.

    Vale mais que o teste anterior: entre a intenção e o ERENO ainda há o
    schema, e é ele que recusa a combinação impossível — uma intenção
    autorizada não é por si só um ataque que exista.
    """

    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])
    config = compile_attack_candidate(intent).config

    ProgrammableConfig.model_validate(config)
    for path, value in entry["payload"]["target_values"].items():
        node = config
        for key in path.split("."):
            node = node[key]
        assert node == value, path


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=_ids(VALID_ENTRIES))
def test_the_untouched_slot_keeps_its_baseline_selection(entry):
    """Autorar só ``r0`` deixa ``r1`` no default, e o default mira tudo —
    é o que faz uma config sem seleção se comportar como antes da Fase 2.2."""

    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])
    config = compile_attack_candidate(intent).config

    assert config["rules"]["r1"]["when"]["cmp"] == "always"


# --------------------------------------------------------------------------- #
# invalid — recusado nos portões da intenção                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", INVALID_ENTRIES, ids=_ids(INVALID_ENTRIES))
def test_impossible_prompts_never_reach_a_compiled_intent(entry):
    with pytest.raises((IntentCompilationError, ValueError), match=entry["expected_error"]):
        FakeIntentAgent.compile(entry["payload"], entry["prompt"])


# --------------------------------------------------------------------------- #
# incoherent — aprovado campo a campo, recusado pelo schema no compilador     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", INCOHERENT_ENTRIES, ids=_ids(INCOHERENT_ENTRIES))
def test_incoherent_rules_pass_the_gate_and_fail_the_compile(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    with pytest.raises(IntentCompilerError, match=entry["expected_error"]):
        compile_attack_candidate(intent)
