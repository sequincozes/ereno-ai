"""FakeStrategist — stub determinístico do agente Red Team (issue #3).

Permite **rodar o loop / testar o Analista (M2)** sem depender do
``StrategistAgent`` real, que exige a API da Groq. Emite um ``StrategistOutput``
no formato do contrato (#4, M1): ``{reasoning, persona, changes}``.

Determinístico: dada a mesma iteração, devolve sempre a mesma sugestão — é o que
torna as fixtures/golden reprodutíveis.
"""

from __future__ import annotations

from typing import Any

from adversarial_ids.domain.strategist_output import Change, StrategistOutput

# Sequência fixa de jogadas do "Red Team", sobre campos editáveis do AttackConfig.
_SCRIPTED_MOVES: list[dict[str, Any]] = [
    {"persona": "aggressive", "field": "fault.prob", "value": 0.75},
    {"persona": "conservative", "field": "trapArea.spikeProb", "value": 0.6},
    {"persona": "aggressive", "field": "analog.deltaAbs.max", "value": 1.2},
    {"persona": "conservative", "field": "fault.durationMs.max", "value": 1000},
]


class FakeStrategist:
    """Stand-in do Estrategista. Interface nova (contrato StrategistOutput)."""

    def __init__(self, moves: list[dict[str, Any]] | None = None) -> None:
        self.moves = moves or _SCRIPTED_MOVES

    def propose(self, iteration: int, **_: Any) -> StrategistOutput:
        """Sugere uma alteração de ataque para a iteração (1-based)."""
        move = self.moves[(iteration - 1) % len(self.moves)]
        return StrategistOutput(
            reasoning=(
                f"[fake] iteração {iteration}: ajustar {move['field']} "
                f"para {move['value']}."
            ),
            persona=move["persona"],
            changes=[Change(field=move["field"], value=move["value"])],
        )

    @staticmethod
    def to_patch(strategist_output: dict[str, Any]) -> list[dict[str, Any]]:
        """Converte ``StrategistOutput.changes`` → patch do ``shared/json_patch``."""
        return [
            {
                "operation": "replace",
                "field": change["field"],
                "old_value": None,
                "new_value": change["value"],
                "reason": "FakeStrategist",
            }
            for change in strategist_output.get("changes", [])
        ]
