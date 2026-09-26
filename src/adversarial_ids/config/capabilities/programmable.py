"""Capacidade intent-driven do ataque ``programmable`` (uc11).

Este é o catálogo do ataque que carrega comportamentos **fora** dos dez fixos
(Fase 2.1). O comportamento é uma lista de regras de mutação em slots
(``rules.r0``, ``rules.r1``), e a capacidade descreve o que cada campo de cada
slot aceita — a **gramática** das regras, não uma lista de campos do ERENO.

Dois tipos de alavanca, e a distinção é o coração do desenho:

- **``fraction`` (com efeito):** a intensidade da regra — quantas mensagens ela
  muta. É a única alavanca que a heurística do compilador varre por intensidade,
  porque é a única que mapeia limpo para evasão (menos mutação → mais parecido
  com o benigno) e atividade (mais mutação). Baseline 0.5 para ter folga nas
  duas direções (o piso 0.05 evita degenerar em "não muta nada").

- **``op``/``field``/``value`` (efeito vazio):** *o que* a regra faz — o
  comportamento em si. Não são varridos pela heurística (efeito vazio =
  ``fields_for_effect`` nunca os escolhe); são **autorados pelo LLM via
  ``target_values``** (Fase 1), que aplica um valor ditado sem depender de
  efeito. É assim que "force o disjuntor em 1 em 30% das mensagens" vira
  ``rules.r0 = {op:set, field:cbStatus, value:1, fraction:0.3}`` sem um
  compilador novo: a gramática está aqui, o LLM preenche os slots.

Os campos mutáveis (``field.choices``) são os inteiros que o creator Java sabe
tocar (``ProgrammableCreatorC.FIELD_ALLOWLIST``) — as duas listas precisam andar
juntas, ou o Java rejeita em tempo de geração o que o portão Python aceitou.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import DETECTION_AND_ACTIVITY_EFFECTS
from adversarial_ids.domain.intent_spec import DesiredEffect, IntentObjective

# Espelha ProgrammableCreatorC.FIELD_ALLOWLIST no lado Java.
_MUTABLE_FIELDS = ("cbStatus", "stNum", "sqNum", "gooseTimeAllowedtoLive", "confRev")
_OPS = ("set", "add", "scale")
_NO_EFFECT: frozenset[DesiredEffect] = frozenset()


def _slot_fields(slot: str) -> tuple[FieldCapability, ...]:
    """Os quatro campos de um slot de regra: fraction (alavanca) + op/field/value."""

    return (
        FieldCapability(
            path=f"rules.{slot}.fraction",
            value_type="number",
            description=(
                f"Fração das mensagens que a regra {slot} muta (intensidade). "
                "Menor = mais evasivo; maior = mais atividade de ataque."
            ),
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.05,
            maximum=1.0,
        ),
        FieldCapability(
            path=f"rules.{slot}.op",
            value_type="string",
            description=f"Operação da regra {slot}: set/add/scale sobre o campo.",
            effects=_NO_EFFECT,
            choices=_OPS,
        ),
        FieldCapability(
            path=f"rules.{slot}.field",
            value_type="string",
            description=f"Campo GOOSE que a regra {slot} muta.",
            effects=_NO_EFFECT,
            choices=_MUTABLE_FIELDS,
        ),
        FieldCapability(
            path=f"rules.{slot}.value",
            value_type="number",
            description=f"Operando da regra {slot} (o valor de set/add/scale).",
            effects=_NO_EFFECT,
        ),
    )


PROGRAMMABLE_CAPABILITY = AttackCapability(
    capability_id="programmable.v1",
    attack_key="programmable",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=DETECTION_AND_ACTIVITY_EFFECTS,
    # fraction primeiro (fields[0] precisa ter efeito para os testes que usam
    # capability.fields[0]); depois os campos author-only de cada slot.
    fields=_slot_fields("r0") + _slot_fields("r1"),
)
