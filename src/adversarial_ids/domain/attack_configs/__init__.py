"""domain/attack_configs — um schema Pydantic por ataque ERENO registrado.

Cada módulo irmão define o schema fechado (``extra="forbid"``) de um ataque
específico de ``config/attacks_registry.py``. Nenhum caminho de config pode
ser tipado globalmente: o mesmo nome (``cbStatus``, ``ttlMs``, ``burst``)
assume formas incompatíveis entre ataques (ex.: ``cbStatus`` é um inteiro no
``masquerade_fault`` e um objeto ``{"values": [...]}`` na ``injection``) — daí
um modelo por ataque em vez de um schema genérico.

Este pacote não importa nada de ``config/`` (regra de camada: ``domain/`` é
vocabulário puro). A resolução ataque→modelo mora em ``CONFIG_MODEL_BY_ATTACK``
como um dict chaveado por string literal, para o ``core/intent_compiler.py``
(que já importa das duas camadas) amarrar o ``AttackSpec.key`` ao modelo certo.
"""

from __future__ import annotations

from pydantic import BaseModel

from adversarial_ids.domain.attack_configs.delayed_replay import (
    DelayedReplayBackoffConfig,
    DelayedReplayBatchDumpConfig,
    DelayedReplayConfig,
    DelayedReplayDoubleDropConfig,
)
from adversarial_ids.domain.attack_configs.flooding import FloodingConfig
from adversarial_ids.domain.attack_configs.grayhole import GrayholeConfig
from adversarial_ids.domain.attack_configs.high_stnum import HighStNumInjectionConfig
from adversarial_ids.domain.attack_configs.injection import InjectionConfig
from adversarial_ids.domain.attack_configs.inverse_replay import InverseReplayConfig
from adversarial_ids.domain.attack_configs.masquerade_fault import (
    AnalogConfig,
    FaultConfig,
    MasqueradeFaultConfig,
    TrapAreaConfig,
)
from adversarial_ids.domain.attack_configs.random_replay import RandomReplayConfig
from adversarial_ids.domain.attack_configs.ranges import (
    FloatRange,
    FrozenConfig,
    IntRange,
    Probability,
    ProbabilityRange,
    ProbabilityValue,
)

CONFIG_MODEL_BY_ATTACK: dict[str, type[BaseModel]] = {
    "masquerade_fault": MasqueradeFaultConfig,
    "random_replay": RandomReplayConfig,
    "inverse_replay": InverseReplayConfig,
    "injection": InjectionConfig,
    "high_stnum": HighStNumInjectionConfig,
    "flooding": FloodingConfig,
    "grayhole": GrayholeConfig,
    "delayed_replay": DelayedReplayConfig,
    "delayed_replay_backoff": DelayedReplayBackoffConfig,
    "delayed_replay_batch_dump": DelayedReplayBatchDumpConfig,
    "delayed_replay_double_drop": DelayedReplayDoubleDropConfig,
}


def config_model_for(attack_key: str) -> type[BaseModel] | None:
    """Modelo Pydantic que valida a config compilada de ``attack_key``, se houver."""

    return CONFIG_MODEL_BY_ATTACK.get(attack_key)


__all__ = [
    "CONFIG_MODEL_BY_ATTACK",
    "config_model_for",
    "FrozenConfig",
    "IntRange",
    "FloatRange",
    "Probability",
    "ProbabilityRange",
    "ProbabilityValue",
    "MasqueradeFaultConfig",
    "FaultConfig",
    "AnalogConfig",
    "TrapAreaConfig",
    "RandomReplayConfig",
    "InverseReplayConfig",
    "InjectionConfig",
    "HighStNumInjectionConfig",
    "FloodingConfig",
    "GrayholeConfig",
    "DelayedReplayConfig",
    "DelayedReplayBackoffConfig",
    "DelayedReplayBatchDumpConfig",
    "DelayedReplayDoubleDropConfig",
]
