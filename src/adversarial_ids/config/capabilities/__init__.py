"""config/capabilities — um módulo por ataque, cada um exportando um ``AttackCapability``.

Reunidos aqui em ``ALL_CAPABILITIES`` para a fachada
(``config/attack_capabilities.py``) montar ``ATTACK_CAPABILITY_CATALOG``.
"""

from __future__ import annotations

from adversarial_ids.config.capabilities.delayed_replay import (
    DELAYED_REPLAY_BACKOFF_CAPABILITY,
    DELAYED_REPLAY_BATCH_DUMP_CAPABILITY,
    DELAYED_REPLAY_CAPABILITY,
    DELAYED_REPLAY_DOUBLE_DROP_CAPABILITY,
)
from adversarial_ids.config.capabilities.flooding import FLOODING_CAPABILITY
from adversarial_ids.config.capabilities.grayhole import GRAYHOLE_CAPABILITY
from adversarial_ids.config.capabilities.high_stnum import HIGH_STNUM_CAPABILITY
from adversarial_ids.config.capabilities.injection import INJECTION_CAPABILITY
from adversarial_ids.config.capabilities.inverse_replay import INVERSE_REPLAY_CAPABILITY
from adversarial_ids.config.capabilities.masquerade_fault import MASQUERADE_FAULT_CAPABILITY
from adversarial_ids.config.capabilities.random_replay import RANDOM_REPLAY_CAPABILITY

ALL_CAPABILITIES = (
    MASQUERADE_FAULT_CAPABILITY,
    RANDOM_REPLAY_CAPABILITY,
    INVERSE_REPLAY_CAPABILITY,
    INJECTION_CAPABILITY,
    HIGH_STNUM_CAPABILITY,
    FLOODING_CAPABILITY,
    GRAYHOLE_CAPABILITY,
    DELAYED_REPLAY_CAPABILITY,
    DELAYED_REPLAY_BACKOFF_CAPABILITY,
    DELAYED_REPLAY_BATCH_DUMP_CAPABILITY,
    DELAYED_REPLAY_DOUBLE_DROP_CAPABILITY,
)

__all__ = ["ALL_CAPABILITIES"]
