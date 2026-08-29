"""Regressão do schema da tool ``submit_strategist_output``.

Bug (encontrado ao rodar o loop live): ``changes: list[dict[str, Any]]`` fazia o
agno gerar um schema em que ``field``/``value`` eram **objetos**, então o Groq
rejeitava toda tool call (``tool_use_failed``) e o Estrategista caía num fallback
silencioso com ``changes=[]`` — o ataque nunca evoluía. A correção usa o modelo
tipado ``Change``. Este teste garante que o schema continua correto.
"""

from __future__ import annotations

from adversarial_ids.agents.strategist.tools import submit_strategist_output


def _changes_item_schema() -> dict:
    return submit_strategist_output.parameters["properties"]["changes"]["items"]


def test_changes_field_is_string_not_object():
    item = _changes_item_schema()
    assert item["type"] == "object"
    assert item["properties"]["field"]["type"] == "string"


def test_changes_value_is_scalar_union_not_object():
    value_schema = _changes_item_schema()["properties"]["value"]
    any_of_types = {branch.get("type") for branch in value_schema.get("anyOf", [])}
    # deve aceitar escalares (número/int/bool/string), nunca só "object"
    assert any_of_types == {"number", "integer", "boolean", "string"}
    assert "object" not in any_of_types
