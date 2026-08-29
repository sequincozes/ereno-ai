"""Derivação data-driven dos campos editáveis de um ataque.

Em vez de listar à mão os campos de cada tipo de ataque, achatamos o próprio
JSON de configuração em caminhos ``dot`` até as folhas. Assim qualquer ataque
modelado no ERENO — masquerade, replay, injection, flooding, grayhole, delayed —
expõe automaticamente seus parâmetros ao Estrategista, sem schema por ataque.

Regras:
- Chaves **estruturais** (``attackType``, ``enabled``, ``description``,
  ``comment``) nunca são editáveis.
- Listas são tratadas como **folhas** (o Estrategista propõe a lista inteira,
  ex.: ``ttlMsValues``, ``sqnumStrideValues``, ``cbStatus.values``).
- Cada folha vira ``{"field": <path>, "current_value": <v>, "type": <nome>}``.
"""

from __future__ import annotations

from typing import Any

# Chaves que descrevem/estruturam o ataque, não parâmetros a otimizar.
STRUCTURAL_KEYS: frozenset[str] = frozenset(
    {"attackType", "enabled", "description", "comment"}
)


def derive_editable_fields(
    config: dict[str, Any],
    *,
    max_fields: int | None = None,
) -> list[dict[str, Any]]:
    """Achata ``config`` nas folhas editáveis (caminhos ``dot``)."""

    fields: list[dict[str, Any]] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not prefix and key in STRUCTURAL_KEYS:
                    continue
                child_prefix = f"{prefix}.{key}" if prefix else key
                walk(value, child_prefix)
            return

        # Listas e escalares são folhas editáveis.
        fields.append(
            {
                "field": prefix,
                "current_value": node,
                "type": type(node).__name__,
            }
        )

    walk(config, "")

    if max_fields is not None:
        return fields[:max_fields]
    return fields


def editable_field_paths(config: dict[str, Any]) -> set[str]:
    """Conjunto de caminhos editáveis — usado para validar propostas do agente."""
    return {item["field"] for item in derive_editable_fields(config)}
