"""Injector: as estratégias de injeção (intent/model/data-driven) atrás de um contrato só.

Este ``__init__`` não importa ``agno``: a CLI lista as chaves sem carregar o LLM.
O agente real fica em ``agent.py`` e só é importado por ``build_injector``.
"""

from adversarial_ids.agents.injector.gate import (
    InjectorOutputError,
    enforce_injector_output,
)
from adversarial_ids.agents.injector.registry import (
    DEFAULT_INJECTOR_KEY,
    InjectorError,
    PreparedInjector,
    build_injector,
    injector_spec,
    list_injector_keys,
    prepare_injector,
)

__all__ = [
    "DEFAULT_INJECTOR_KEY",
    "InjectorError",
    "InjectorOutputError",
    "PreparedInjector",
    "build_injector",
    "enforce_injector_output",
    "injector_spec",
    "list_injector_keys",
    "prepare_injector",
]
