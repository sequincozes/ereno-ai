"""StrategistOutput — contrato de saída do Estrategista (Red Team, M1 · issue #4).

Define o formato que o agente Estrategista deve produzir a cada iteração:
reasoning (texto livre), persona (conservative/aggressive) e lista de alterações
(``Change``) a aplicar na configuração do ataque (schema variável — o loop
legado aceita qualquer um dos ataques registrados, não só o masquerade_fault).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Change(BaseModel):
    """Uma alteração atómica em um campo editável da config do ataque."""

    model_config = ConfigDict(extra="forbid")

    field: str
    value: float | int | bool | str


class StrategistOutput(BaseModel):
    """Saída tipada do Estrategista (Red Team) para uma iteração."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    persona: Literal["conservative", "aggressive"]
    changes: list[Change]
