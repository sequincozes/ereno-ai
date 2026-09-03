"""Schema estrito do ataque ``masquerade_fault`` (uc03).

Congelado desde a Fase 0 (issue #2) — era ``domain/attack_config.py::AttackConfig``
até este módulo passar a ser um entre dez (``domain/attack_configs/``), um por
ataque registrado em ``config/attacks_registry.py``. O nome genérico
``AttackConfig`` deixou de fazer sentido quando outros ataques passaram a ter
seu próprio schema — daí o rename para ``MasqueradeFaultConfig``.

É o único schema desta família que também funciona como serializador: o golden
byte-estável ``data/iteration_history.json`` é gerado gravando
``model_dump(mode="json")`` deste modelo via ``IterationRecord`` (ver
``scripts/generate_golden_history.py``). Mudar a ordem dos campos ou trocar
``int`` por ``float`` em qualquer folha muda o golden.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from adversarial_ids.domain.attack_configs.ranges import (
    FloatRange,
    IntRange,
    Probability,
    FrozenConfig,
)


class FaultConfig(FrozenConfig):
    prob: Probability
    durationMs: IntRange


class AnalogConfig(FrozenConfig):
    deltaAbs: FloatRange


class TrapAreaConfig(FrozenConfig):
    multiplier: FloatRange
    spikeProb: Probability


class MasqueradeFaultConfig(FrozenConfig):
    """Configuração completa do ataque ``masquerade_fault`` (uc03)."""

    # --- estruturais (não editáveis pela persona) ---
    attackType: Literal["masquerade_fault"]
    enabled: bool = True

    # --- 12 campos editáveis ---
    fault: FaultConfig
    cbStatus: Literal[0, 1]
    incrementStNumOnFault: bool
    sqnumMode: str
    ttlMsValues: list[int] = Field(min_length=1)
    analog: AnalogConfig
    trapArea: TrapAreaConfig
