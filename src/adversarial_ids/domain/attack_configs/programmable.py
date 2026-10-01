"""Schema estrito do ataque ``programmable`` (uc11).

Diferente dos outros dez: o comportamento não é um conjunto fixo de campos, e
sim uma **lista de regras** aplicadas às mensagens GOOSE. As regras vivem sob
``rules`` como um **objeto** chaveado por slot (``{"r0": {...}, "r1": {...}}``),
não um array — o modelo de caminhos do repo é dict pontilhado
(``rules.r0.fraction``) e não tem suporte a índice de array; um dict de slots
mantém as duas pontas (Python e o creator Java) sobre o mesmo modelo de caminho.

Cada regra é ``{op, field, value, fraction, when}``:

- ``op``: ``set``/``add``/``scale`` (mutação de campo) ou ``drop``/``duplicate``
  (cardinalidade do fluxo — a mensagem some, ou vira várias);
- ``field``: o campo GOOSE que a mutação toca (as ops de cardinalidade o ignoram);
- ``value``: o operando — da mutação, ou o número de cópias **extras** do
  ``duplicate``;
- ``fraction``: probabilidade em [0, 1] de a regra disparar por mensagem;
- ``when``: a **seleção** — ``{field, cmp, value}``; a regra só dispara nas
  mensagens em que a condição vale (``cmp="always"`` é o neutro, toda mensagem).

O ``field`` fica como ``str`` (e não um enum) de propósito: o domínio de campos
válidos é a allowlist da **capacidade** (``config/capabilities/programmable.py``),
que é o que o portão determinístico aplica — o schema garante o *shape*, a
capacidade garante o *domínio*. O que mora aqui são as regras de **coerência
entre campos da mesma regra**, que nenhuma allowlist campo-a-campo consegue
expressar, e os poucos nomes de campo que elas precisam citar (``_TIME_FIELDS``),
importados de volta pela capacidade para não haver duas listas.

Temporização (atraso, reordenação) **não tem op própria**: o fluxo escrito é
ordenado por ``timestamp`` (o ERENO drena as mensagens de um ``PriorityQueue``),
então atrasar é ``add`` em ``timestamp`` com operando positivo e reordenar é o
mesmo ``add`` com operando negativo. Ver ``docs/programmable_attack.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from adversarial_ids.domain.attack_configs.ranges import FrozenConfig, Probability

#: Campos de tempo do quadro GOOSE, em **segundos** (10 ms = ``0.01``). São
#: valores de relógio absolutos: ``set``/``scale`` neles não atrasam a mensagem,
#: realocam ela para um ponto arbitrário da captura. Só ``add`` (um deslocamento
#: com sinal) tem significado, e é o que o creator Java aceita.
TIME_FIELDS: tuple[str, ...] = ("t", "timestamp")

#: Campos inteiros que o creator Java sabe mutar.
INTEGER_FIELDS: tuple[str, ...] = (
    "cbStatus",
    "stNum",
    "sqNum",
    "gooseTimeAllowedtoLive",
    "confRev",
)

#: Espelha ``ProgrammableCreatorC.FIELD_ALLOWLIST`` no lado Java.
MUTABLE_FIELDS: tuple[str, ...] = INTEGER_FIELDS + TIME_FIELDS

#: Teto de cópias extras de uma regra ``duplicate``. Espelha
#: ``ProgrammableCreatorC.MAX_EXTRA_COPIES`` — sem teto, uma fração alta
#: multiplicaria o trace inteiro.
MAX_EXTRA_COPIES = 10


class RuleCondition(FrozenConfig):
    """Seleção de uma regra: em quais mensagens ela pode disparar.

    Avaliada contra a mensagem **original**, antes de qualquer mutação — qual
    mensagem uma regra mira não pode depender do que o slot anterior já mudou,
    ou a ordem dos slots viraria parte do comportamento.

    ``cmp="always"`` é o neutro e ignora ``field``/``value``: é o default do
    baseline, e é o que faz uma config sem seleção se comportar exatamente como
    antes da Fase 2.2.
    """

    field: str = Field(min_length=1)
    cmp: Literal["always", "eq", "ne", "gt", "lt", "gte", "lte"]
    value: float


class MutationRule(FrozenConfig):
    """Uma regra de um slot do ataque programável."""

    op: Literal["set", "add", "scale", "drop", "duplicate"]
    field: str = Field(min_length=1)
    value: float
    fraction: Probability
    when: RuleCondition

    @model_validator(mode="after")
    def _op_matches_field_and_value(self) -> "MutationRule":
        # Campo de tempo só aceita deslocamento. Checado por pertencimento a
        # TIME_FIELDS e não por exclusão: um ``field`` desconhecido continua
        # passando aqui para ser recusado pelo portão de capacidades (e, em
        # último caso, pelo creator Java) com a mensagem certa.
        if self.field in TIME_FIELDS and self.op in ("set", "scale"):
            raise ValueError(
                f"op {self.op!r} não é permitida no campo de tempo {self.field!r} — "
                "ele é um relógio absoluto em segundos, e só 'add' (deslocamento "
                "com sinal) o atrasa ou o reordena em vez de realocá-lo."
            )
        if self.op == "duplicate":
            copies = round(self.value)
            if not 1 <= copies <= MAX_EXTRA_COPIES:
                raise ValueError(
                    f"op 'duplicate' usa 'value' como número de cópias extras, "
                    f"que precisa estar em [1, {MAX_EXTRA_COPIES}]; recebido: "
                    f"{self.value}."
                )
        return self


class ProgrammableConfig(FrozenConfig):
    """Configuração completa do ataque ``programmable`` (uc11)."""

    attackType: Literal["programmable"]
    enabled: bool
    # Objeto de slots, não lista. min_length=1: um ataque sem regra nenhuma não
    # muta nada e seria indistinguível do tráfego benigno (o gate E4 o reprovaria
    # de qualquer forma, mas falhar já no schema é mais cedo e mais claro).
    rules: dict[str, MutationRule] = Field(min_length=1)
