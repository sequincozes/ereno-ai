"""Catálogo de capacidades intent-driven dos ataques suportados.

O catálogo é a allowlist entre uma intenção semântica e os campos que um
compilador poderá alterar — um ``AttackCapability`` por ataque registrado em
``config/attacks_registry.py`` (ver ``config/capabilities/``). Seis invariantes
valem para todo catálogo, cada um coberto por
``tests/test_attack_capabilities_catalog.py``:

1. **Nenhum caminho inventado**: ``editable_paths`` é sempre um subconjunto dos
   campos editáveis do baseline (``shared/editable_fields.py``); toda omissão
   é deliberada e documentada (campos mortos na baseline, controles de
   reprodutibilidade, enums sem opções conhecidas).
2. **Nenhuma alavanca morta**: todo par ``(campo, efeito)`` catalogado precisa
   de fato mudar o valor em intensidade alta — um campo já no piso/teto na
   direção do efeito não pode carregar aquele efeito.
3. **``minimum`` é o piso anti-degeneração**: o pipeline intent-driven não roda
   ``shared/validator.py`` (isso é só do loop legado) — o ``minimum`` do
   catálogo é a única defesa contra compilar o ataque para fora de existência.
4. **As três alavancas de evasão** (``lower_f1``, ``lower_recall``,
   ``mimic_normal_traffic``) sempre carregam o mesmo conjunto de campos numa
   capacidade — é o que justifica a política E10 não escalonar por
   ``desired_effect``.
5. **``supported_effects`` é a união dos efeitos dos campos** — anunciar um
   efeito que nenhum campo produz faz o portão falhar tarde, com a mensagem
   "removem todos os campos" em vez da real.
6. **Probabilidades sem "prob" no nome** (``dropRate``, ``selectionProb.value``)
   declaram ``minimum=0.0, maximum=1.0`` explicitamente — o clamp genérico
   legado (``shared/validator.py``) casa só pela substring ``"prob"`` na chave.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentObjective,
    IntentSpec,
)


class FieldCapability(BaseModel):
    """Campo de configuração autorizado e sua semântica operacional."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    value_type: Literal["boolean", "integer", "number", "string", "integer_list"]
    description: str
    effects: frozenset[DesiredEffect]
    # "direct"  — valor maior = ataque mais intenso (a maioria dos campos).
    # "inverse" — valor maior = ataque MENOS intenso (gaps, intervalos entre
    #   rajadas). O compilador inverte a direção efetiva destes campos, para
    #   que "evadir detecção" nunca signifique apertar um gap — o que tornaria
    #   o ataque mais detectável, não menos.
    polarity: Literal["direct", "inverse"] = "direct"
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[Any, ...] | None = None

    @model_validator(mode="after")
    def _string_fields_need_choices(self) -> "FieldCapability":
        """Um campo ``string`` sem ``choices`` não tem alternante determinístico.

        ``_compute_choice_value`` (``core/intent_compiler.py``) levantaria em
        tempo de compilação, e só quando a seed sorteasse esse campo — falha
        intermitente por construção (ex.: ``orderBy`` do uc10, que não tem
        enum documentado em nenhum lugar do repo). O catálogo é uma allowlist:
        um campo cujo espaço de valores é desconhecido não pode entrar nela.
        Falhar aqui é falhar no import, não numa rodada aleatória.
        """
        if self.value_type == "string" and not self.choices:
            raise ValueError(
                f"Campo {self.path!r} é 'string' sem 'choices' — declare as "
                "opções aceitas pelo gerador ou remova o campo do catálogo."
            )
        return self


class AttackCapability(BaseModel):
    """Capacidade intent-driven versionada de um ``AttackSpec``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_id: str
    schema_version: Literal[1] = 1
    attack_key: str
    supported_objectives: frozenset[IntentObjective]
    supported_effects: frozenset[DesiredEffect]
    fields: tuple[FieldCapability, ...] = Field(min_length=1)

    @property
    def editable_paths(self) -> frozenset[str]:
        return frozenset(field.path for field in self.fields)

    def fields_for_effect(self, effect: DesiredEffect) -> tuple[FieldCapability, ...]:
        return tuple(field for field in self.fields if effect in field.effects)


# Import tardio (depois da definição de FieldCapability/AttackCapability, que
# os módulos de capacidade por ataque importam deste mesmo arquivo) — evita um
# ciclo de import entre a fachada e o pacote ``capabilities/``.
from adversarial_ids.config.capabilities import ALL_CAPABILITIES  # noqa: E402

ATTACK_CAPABILITY_CATALOG: dict[str, AttackCapability] = {
    capability.capability_id: capability for capability in ALL_CAPABILITIES
}

# Reexportado por conveniência/compat — a maioria do código só precisa de
# ``get_attack_capability("masquerade_fault")``, mas alguns testes e o
# golden-prompt fixture antigo referenciam a constante diretamente.
MASQUERADE_FAULT_CAPABILITY = ATTACK_CAPABILITY_CATALOG["masquerade_fault.v1"]


def get_attack_capability(attack_key: str) -> AttackCapability:
    """Retorna a capacidade ligada ao ``AttackSpec`` ou um erro acionável."""

    spec = get_attack_spec(attack_key)
    if spec.intent_capability_id is None:
        enabled = ", ".join(sorted(c.attack_key for c in ATTACK_CAPABILITY_CATALOG.values()))
        raise ValueError(
            f"Ataque {attack_key!r} ainda não possui capacidade intent-driven. "
            f"Ataques habilitados: {enabled}."
        )
    try:
        return ATTACK_CAPABILITY_CATALOG[spec.intent_capability_id]
    except KeyError:
        raise RuntimeError(
            f"AttackSpec {attack_key!r} referencia capacidade inexistente: "
            f"{spec.intent_capability_id!r}."
        ) from None


def resolve_candidate_paths(intent: IntentSpec) -> tuple[AttackCapability, frozenset[str]]:
    """Valida a allowlist e devolve os campos editáveis capazes do efeito.

    Compartilhado entre ``validate_intent_capability`` (portão do IntentAgent)
    e o compilador spec→AttackCandidate (épico E2) — ambos precisam do mesmo
    conjunto de campos candidatos, calculado uma única vez.
    """

    capability = get_attack_capability(intent.base_attack)
    if intent.objective not in capability.supported_objectives:
        raise ValueError(
            f"Objetivo {intent.objective.value!r} não é suportado por "
            f"{intent.base_attack!r}."
        )
    if intent.desired_effect not in capability.supported_effects:
        raise ValueError(
            f"Efeito {intent.desired_effect.value!r} não é suportado por "
            f"{intent.base_attack!r}."
        )

    known_paths = capability.editable_paths
    requested_paths = set(intent.restrictions.allowed_fields or ())
    forbidden_paths = set(intent.restrictions.forbidden_fields)
    unknown_paths = (requested_paths | forbidden_paths) - known_paths
    if unknown_paths:
        fields = ", ".join(sorted(unknown_paths))
        raise ValueError(
            f"Campos fora da allowlist de {intent.base_attack!r}: {fields}."
        )

    candidates = {
        field.path for field in capability.fields_for_effect(intent.desired_effect)
    }
    if requested_paths:
        candidates &= requested_paths
    candidates -= forbidden_paths
    if not candidates:
        raise ValueError(
            "As restrições removem todos os campos capazes de produzir o efeito "
            f"{intent.desired_effect.value!r}."
        )
    return capability, frozenset(candidates)


def validate_intent_capability(intent: IntentSpec) -> AttackCapability:
    """Valida se a intenção pode ser compilada usando somente a allowlist."""

    capability, _ = resolve_candidate_paths(intent)
    return capability
