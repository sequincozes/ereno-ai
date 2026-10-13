"""Instruções do LLM para os injetores model-driven e data-driven.

Ordem do que o modelo lê: o papel da estratégia (``prompts/injector_<key>.md``),
o bloco comum de como registrar o ataque (``prompts/injector_common.md``), o
catálogo **só do programmable** e, por último, o contexto próprio da estratégia
(o creator Java ou o perfil do trace).

Só o programmable entra no catálogo, e não os 12 ataques: a estratégia só pode
devolver ``programmable`` (``gate.py``), e o catálogo inteiro custaria ~7k dos
8k tokens por minuto da conta — sem sobrar espaço para o código que o modelo
precisa ler. Ver ``docs/injectors.md``.
"""

from __future__ import annotations

from adversarial_ids.agents.intent.catalog_context import build_capability_context
from adversarial_ids.config.attack_capabilities import ATTACK_CAPABILITY_CATALOG
from adversarial_ids.config.settings import PROMPTS_DIR
from adversarial_ids.domain.injector_manifest import base_attack_constraint_for

_COMMON_PROMPT = "injector_common.md"


def _constrained_catalog(injector: str) -> str:
    attack_key = base_attack_constraint_for(injector)
    subset = {
        key: capability
        for key, capability in ATTACK_CAPABILITY_CATALOG.items()
        if capability.attack_key == attack_key
    }
    return build_capability_context(subset)


def build_injector_instructions(injector: str, context_block: str) -> str:
    """Instruções completas de um injetor contextual (``model`` ou ``data``)."""

    if base_attack_constraint_for(injector) is None:
        raise ValueError(
            f"O injetor {injector!r} não é contextual; ele usa as instruções do "
            "IntentAgent."
        )
    role = (PROMPTS_DIR / f"injector_{injector}.md").read_text(encoding="utf-8")
    common = (PROMPTS_DIR / _COMMON_PROMPT).read_text(encoding="utf-8")
    return "\n\n".join(
        part.strip()
        for part in (role, common, _constrained_catalog(injector), context_block)
    ) + "\n"


def render_java_context(attack_key: str, ref: str, source: str) -> str:
    """O creator Java como bloco que fecha as instruções do model-driven."""

    return (
        f"## Creator Java de `{attack_key}`\n\n"
        f"Arquivo `{ref}` do ERENO (pacote e imports omitidos).\n\n"
        f"```java\n{source.rstrip()}\n```\n"
    )
