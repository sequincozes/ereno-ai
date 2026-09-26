"""Regressão do schema da tool ``submit_intent_spec``.

Espelha ``test_strategist_tool_schema.py``: o mesmo bug de tool calling do
Estrategista (listas viravam ``list[object]`` no schema do Groq) poderia se
repetir aqui em ``allowed_fields``/``forbidden_fields``. Este teste garante
que o schema gerado continua expondo arrays de string.
"""

from __future__ import annotations

from adversarial_ids.agents.intent.tools import submit_intent_spec


def _properties() -> dict:
    return submit_intent_spec.parameters["properties"]


def test_field_path_lists_are_string_arrays_not_objects():
    props = _properties()
    for name in ("allowed_fields", "forbidden_fields"):
        variants = props[name]["anyOf"]
        array_variant = next(v for v in variants if v.get("type") == "array")
        assert array_variant["items"]["type"] == "string"
        assert {"type": "null"} in variants


def test_core_fields_are_required_strings():
    props = _properties()
    required = set(submit_intent_spec.parameters["required"])
    for name in ("objective", "base_attack", "desired_effect"):
        assert name in required
        assert props[name]["type"] == "string"


def test_source_prompt_is_not_part_of_the_tool_schema():
    # A LLM nunca deve reconstituir source_prompt — ele é injetado pelo
    # IntentAgent depois da tool call.
    assert "source_prompt" not in _properties()


def test_target_values_is_an_object_whose_values_are_scalars_not_empty_objects():
    """Mesmo bug de ``list[object]``, agora em ``dict[str, Any]``.

    Anotado como ``Any``, o Pydantic rende cada valor como um objeto vazio que
    **proíbe** qualquer propriedade — o schema diria à LLM que
    ``{"fault.prob": 0.42}`` é inválido, e o número certo nunca chegaria à
    tool. A união explícita é o que mantém o schema dizendo a verdade sobre o
    que a ferramenta aceita.
    """

    variants = _properties()["target_values"]["anyOf"]
    assert {"type": "null"} in variants

    obj = next(v for v in variants if v.get("type") == "object")
    assert obj["propertyNames"]["type"] == "string"
    accepted = {v["type"] for v in obj["additionalProperties"]["anyOf"]}
    assert accepted == {"boolean", "integer", "number", "string", "array"}


def test_target_values_is_optional_so_a_prompt_without_numbers_stays_simple():
    assert "target_values" not in set(submit_intent_spec.parameters["required"])
