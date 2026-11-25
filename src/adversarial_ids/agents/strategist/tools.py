"""Ferramenta de tool calling estruturado do Estrategista (Agno).

A LLM chama ``submit_strategist_output`` para produzir um
``StrategistOutput`` validado.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agno.tools import tool

from adversarial_ids.domain.strategist_output import Change, StrategistOutput

_VALID_PERSONAS = {"conservative", "aggressive"}

# Caminho ``dot`` válido (data-driven, agnóstico ao tipo de ataque). Aceita
# nomes de campo camelCase e índices aninhados, ex.: ``fault.durationMs.min``,
# ``trapArea.spikeProb``, ``burst.gapMs.max``, ``cbStatus.values``.
_FIELD_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _validate_field(field: Any) -> str | None:
    """Valida a forma do caminho do campo (não o tipo do ataque específico)."""
    if not isinstance(field, str) or not field.strip():
        return "changes.field deve ser uma string não vazia."
    if not _FIELD_PATH_RE.match(field):
        return f"changes.field='{field}' não é um caminho de campo válido (use dot-path)."
    return None


def _validate_value(field: str, value: Any) -> str | None:
    """Aceita apenas valores parametrizáveis: escalar (str/int/float/bool) ou lista.

    Objetos aninhados nunca são propostos diretamente — o Estrategista edita as
    folhas (ex.: ``burst.gapMs.max``), não subárvores inteiras.
    """
    if value is None:
        return f"Valor inválido para {field}: None não é editável."
    if isinstance(value, dict):
        return (
            f"Valor inválido para {field}: edite as folhas do objeto "
            "(ex.: '<campo>.min'), não o objeto inteiro."
        )
    if isinstance(value, (bool, int, float, str, list)):
        return None
    return f"Tipo inválido para {field}: {type(value).__name__} não é suportado."


@tool(stop_after_tool_call=True)
def submit_strategist_output(
    reasoning: str,
    persona: str,
    changes: list[Change],
) -> str:
    """Submete a saída do Estrategista com reasoning, persona e alterações.

    Use esta ferramenta para registrar a saída final da iteração.
    A ferramenta valida os parâmetros e retorna o resultado validado
    ou uma mensagem de erro.

    Args:
        reasoning: Explicação textual da estratégia adotada.
        persona: Perfil de ataque ("conservative" ou "aggressive").
        changes: Lista de alterações, cada uma com "field" (nome do campo)
                 e "value" (novo valor).
    """
    if not reasoning or not reasoning.strip():
        return json.dumps({"error": "reasoning não pode estar vazio."})

    if persona not in _VALID_PERSONAS:
        return json.dumps(
            {
                "error": f"Persona inválida: '{persona}'. "
                f"Use 'conservative' ou 'aggressive'."
            }
        )

    if not isinstance(changes, list):
        return json.dumps({"error": "changes deve ser uma lista."})

    if not changes:
        return json.dumps({"error": "changes não pode estar vazia."})

    parsed_changes: list[Change] = []
    errors: list[str] = []

    for i, entry in enumerate(changes):
        # ``changes: list[Change]`` gera o schema correto no Groq, mas o agno pode
        # entregar aqui tanto instâncias ``Change`` quanto dicts crus — normalizamos.
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()

        if not isinstance(entry, dict):
            errors.append(f"changes[{i}] não é um dicionário.")
            continue

        field = entry.get("field")
        value = entry.get("value")

        if field is None or "value" not in entry:
            errors.append(
                f"changes[{i}] precisa conter 'field' e 'value'."
            )
            continue

        field_error = _validate_field(field)
        if field_error:
            errors.append(f"changes[{i}]: {field_error}")
            continue

        value_error = _validate_value(field, value)
        if value_error:
            errors.append(f"changes[{i}]: {value_error}")
            continue

        parsed_changes.append(Change(field=field, value=value))

    if errors:
        return json.dumps({"errors": errors})

    output = StrategistOutput(
        reasoning=reasoning.strip(),
        persona=persona,  # type: ignore[arg-type]
        changes=parsed_changes,
    )

    return output.model_dump_json()