"""Capacidade intent-driven do ataque ``programmable`` (uc11).

Este é o catálogo do ataque que carrega comportamentos **fora** dos dez fixos.
O comportamento é uma lista de regras em slots (``rules.r0``, ``rules.r1``), e a
capacidade descreve o que cada campo de cada slot aceita — a **gramática** das
regras, não uma lista de campos do ERENO.

Dois tipos de alavanca, e a distinção é o coração do desenho:

- **``fraction`` (com efeito):** a intensidade da regra — em quantas das
  mensagens que ela mira a regra dispara. É a única alavanca que a heurística do
  compilador varre por intensidade, porque é a única que mapeia limpo para
  evasão (menos mutação → mais parecido com o benigno) e atividade (mais
  mutação). Baseline 0.5 para ter folga nas duas direções (o piso 0.05 evita
  degenerar em "não muta nada").

- **``op``/``field``/``value``/``when.*`` (efeito vazio):** *o que* a regra faz e
  *em quem* — o comportamento em si. Não são varridos pela heurística (efeito
  vazio = ``fields_for_effect`` nunca os escolhe); são **autorados pelo LLM via
  ``target_values``** (Fase 1), que aplica um valor ditado sem depender de
  efeito. É assim que "force o disjuntor em 1 em 30% das mensagens" vira
  ``rules.r0 = {op:set, field:cbStatus, value:1, fraction:0.3}`` sem um
  compilador novo: a gramática está aqui, o LLM preenche os slots.

Três eixos de comportamento, uma gramática só (Fase 2.2):

- **mutação** — ``set``/``add``/``scale`` sobre um campo do quadro;
- **seleção** — ``when``: a regra só mira as mensagens em que a condição vale;
- **temporização** — ``add`` nos campos de tempo (``t``, ``timestamp``), em
  segundos: operando positivo atrasa, negativo reordena. Não há op ``delay``
  nem ``reorder`` porque não poderia haver: o fluxo escrito é ordenado por
  ``timestamp``, então o deslocamento do relógio *é* a reordenação. Descarte e
  duplicação, que mexem na cardinalidade e não no relógio, são ops próprias
  (``drop``/``duplicate``).

Os nomes de campo e o teto de cópias vêm de ``domain/attack_configs/programmable``
— a mesma fonte que o schema usa para as regras de coerência entre campos, para
não existirem duas listas. Elas espelham ``ProgrammableCreatorC`` no lado Java,
que é a última linha do guardrail: se o Python aceitar um campo que o Java não
conhece, a geração falha alto em vez de passar batido.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, FieldCapability
from adversarial_ids.config.capabilities._effects import DETECTION_AND_ACTIVITY_EFFECTS
from adversarial_ids.domain.attack_configs.programmable import (
    MAX_EXTRA_COPIES,
    MUTABLE_FIELDS,
)
from adversarial_ids.domain.intent_spec import DesiredEffect, IntentObjective

_OPS = ("set", "add", "scale", "drop", "duplicate")
_COMPARATORS = ("always", "eq", "ne", "gt", "lt", "gte", "lte")
_NO_EFFECT: frozenset[DesiredEffect] = frozenset()
_SLOTS = ("r0", "r1")
_FIRST_SLOT = _SLOTS[0]


#: A gramática por folha, escrita uma vez. Os slots seguintes referenciam o
#: primeiro em vez de repetir o texto: a descrição é o que a LLM lê, e o prompt
#: do IntentAgent paga por token contra o TPM — repetir a gramática por slot
#: dobraria o custo do ataque sem dizer nada novo. O portão não lê descrição
#: nenhuma, então os slots continuam idênticos onde importa (choices/faixas).
_GRAMMAR: dict[str, str] = {
    "fraction": (
        "Fração das mensagens miradas em que a regra dispara (intensidade). "
        "Menor = mais evasivo; maior = mais atividade."
    ),
    "op": (
        "O que a regra faz: set/add/scale mutam o campo; drop descarta a "
        "mensagem; duplicate a repete. Em t/timestamp (segundos) só add vale — "
        "operando positivo atrasa, negativo reordena."
    ),
    "field": "Campo que a regra muta (drop/duplicate o ignoram).",
    "value": (
        "Operando: o valor de set/add/scale, ou o número de cópias extras de "
        f"duplicate (1 a {MAX_EXTRA_COPIES})."
    ),
    "when.cmp": (
        "Seleção: 'always' mira toda mensagem (default); eq/ne/gt/lt/gte/lte "
        "comparam when.field com when.value, lidos antes de qualquer mutação."
    ),
    "when.field": "Campo lido pela seleção.",
    "when.value": "Valor comparado pela seleção.",
}


def _describe(slot: str, leaf: str) -> str:
    if slot == _FIRST_SLOT:
        return _GRAMMAR[leaf]
    return f"Igual a `rules.{_FIRST_SLOT}.{leaf}`, na regra {slot}."


def _slot_fields(slot: str) -> tuple[FieldCapability, ...]:
    """Os campos de um slot de regra: fraction (alavanca) + o resto, autorado."""

    return (
        FieldCapability(
            path=f"rules.{slot}.fraction",
            value_type="number",
            description=_describe(slot, "fraction"),
            effects=DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.05,
            maximum=1.0,
        ),
        FieldCapability(
            path=f"rules.{slot}.op",
            value_type="string",
            description=_describe(slot, "op"),
            effects=_NO_EFFECT,
            choices=_OPS,
        ),
        FieldCapability(
            path=f"rules.{slot}.field",
            value_type="string",
            description=_describe(slot, "field"),
            effects=_NO_EFFECT,
            choices=MUTABLE_FIELDS,
        ),
        FieldCapability(
            path=f"rules.{slot}.value",
            value_type="number",
            description=_describe(slot, "value"),
            effects=_NO_EFFECT,
        ),
        FieldCapability(
            path=f"rules.{slot}.when.cmp",
            value_type="string",
            description=_describe(slot, "when.cmp"),
            effects=_NO_EFFECT,
            choices=_COMPARATORS,
        ),
        FieldCapability(
            path=f"rules.{slot}.when.field",
            value_type="string",
            description=_describe(slot, "when.field"),
            effects=_NO_EFFECT,
            choices=MUTABLE_FIELDS,
        ),
        FieldCapability(
            path=f"rules.{slot}.when.value",
            value_type="number",
            description=_describe(slot, "when.value"),
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
    fields=tuple(field for slot in _SLOTS for field in _slot_fields(slot)),
)
