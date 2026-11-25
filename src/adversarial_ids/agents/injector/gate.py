"""Portão determinístico do Injector: o que model/data-driven podem devolver.

Roda **depois** dos dois portões que toda ``IntentSpec`` já atravessa
(``compile_intent``: contrato Pydantic + allowlist de capacidades). Os dois
primeiros dizem se a intenção é válida; este diz se ela é o que a estratégia
promete ser:

- o ``base_attack`` é o que a estratégia exige (``programmable`` para model e
  data — ver ``domain/injector_manifest.py``);
- há ao menos uma regra **autorada** (um caminho ``rules.`` em
  ``target_values``). Sem isso, o ``programmable`` roda as regras do baseline,
  e o "ataque novo" que o manifest anuncia seria o ataque padrão.

Sem ``agno`` de propósito: é Python puro, testável sem LLM, e é o mesmo código
que a execução real chama.
"""

from __future__ import annotations

from adversarial_ids.domain.injector_manifest import base_attack_constraint_for
from adversarial_ids.domain.intent_spec import IntentSpec

_RULE_PREFIX = "rules."


class InjectorOutputError(ValueError):
    """A intenção é válida, mas não é o que esta estratégia de injeção promete."""


def enforce_injector_output(intent: IntentSpec, injector: str) -> IntentSpec:
    """Devolve ``intent`` intacta, ou levanta ``InjectorOutputError``."""

    constraint = base_attack_constraint_for(injector)
    if constraint is None:
        return intent

    if intent.base_attack != constraint:
        raise InjectorOutputError(
            f"O injetor {injector!r} só propõe ataques novos na DSL do "
            f"{constraint!r}; o LLM devolveu base_attack={intent.base_attack!r}."
        )

    authored = [
        pinned.path
        for pinned in intent.restrictions.target_values
        if pinned.path.startswith(_RULE_PREFIX)
    ]
    if not authored:
        raise InjectorOutputError(
            f"O injetor {injector!r} não autorou nenhuma regra (nenhum caminho "
            f"{_RULE_PREFIX!r} em target_values): o que rodaria é o "
            f"{constraint!r} padrão, não um ataque novo."
        )
    return intent
