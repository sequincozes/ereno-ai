"""Catálogo de capacidades intent-driven dos ataques suportados.

O catálogo é a allowlist entre uma intenção semântica e os campos que um
compilador poderá alterar. Nesta primeira versão, somente ``masquerade_fault``
está habilitado, preservando o vertical slice definido para o Loop V0.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[Any, ...] | None = None


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


_DETECTION_EFFECTS = frozenset(
    {
        DesiredEffect.LOWER_F1,
        DesiredEffect.LOWER_RECALL,
        DesiredEffect.MIMIC_NORMAL_TRAFFIC,
    }
)
_DETECTION_AND_ACTIVITY_EFFECTS = _DETECTION_EFFECTS | {
    DesiredEffect.INCREASE_ATTACK_ACTIVITY
}


MASQUERADE_FAULT_CAPABILITY = AttackCapability(
    capability_id="masquerade_fault.v1",
    attack_key="masquerade_fault",
    supported_objectives=frozenset(
        {IntentObjective.ASSESS_IDS_ROBUSTNESS, IntentObjective.EVADE_DETECTION}
    ),
    supported_effects=_DETECTION_AND_ACTIVITY_EFFECTS,
    fields=(
        FieldCapability(
            path="fault.prob",
            value_type="number",
            description="Probabilidade de injetar a falha forjada.",
            effects=_DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
        FieldCapability(
            path="fault.durationMs.min",
            value_type="integer",
            description="Limite inferior da duração da falha, em milissegundos.",
            effects=_DETECTION_EFFECTS,
            minimum=0,
        ),
        FieldCapability(
            path="fault.durationMs.max",
            value_type="integer",
            description="Limite superior da duração da falha, em milissegundos.",
            effects=_DETECTION_EFFECTS,
            minimum=0,
        ),
        FieldCapability(
            path="cbStatus",
            value_type="integer",
            description="Estado forjado do disjuntor.",
            effects=_DETECTION_EFFECTS,
            choices=(0, 1),
        ),
        FieldCapability(
            path="incrementStNumOnFault",
            value_type="boolean",
            description="Indica se stNum avança durante a falha forjada.",
            effects=_DETECTION_EFFECTS,
            choices=(False, True),
        ),
        FieldCapability(
            path="sqnumMode",
            value_type="string",
            description="Estratégia de evolução do sqNum durante o ataque.",
            effects=_DETECTION_EFFECTS,
        ),
        FieldCapability(
            path="ttlMsValues",
            value_type="integer_list",
            description="Valores de TTL usados nas mensagens GOOSE forjadas.",
            effects=_DETECTION_EFFECTS,
        ),
        FieldCapability(
            path="analog.deltaAbs.min",
            value_type="number",
            description="Perturbação analógica absoluta mínima.",
            effects=_DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="analog.deltaAbs.max",
            value_type="number",
            description="Perturbação analógica absoluta máxima.",
            effects=_DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.multiplier.min",
            value_type="number",
            description="Multiplicador mínimo da área do trapézio.",
            effects=_DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.multiplier.max",
            value_type="number",
            description="Multiplicador máximo da área do trapézio.",
            effects=_DETECTION_EFFECTS,
            minimum=0.0,
        ),
        FieldCapability(
            path="trapArea.spikeProb",
            value_type="number",
            description="Probabilidade de produzir spike na trap area.",
            effects=_DETECTION_AND_ACTIVITY_EFFECTS,
            minimum=0.0,
            maximum=1.0,
        ),
    ),
)


ATTACK_CAPABILITY_CATALOG: dict[str, AttackCapability] = {
    MASQUERADE_FAULT_CAPABILITY.capability_id: MASQUERADE_FAULT_CAPABILITY
}


def get_attack_capability(attack_key: str) -> AttackCapability:
    """Retorna a capacidade ligada ao ``AttackSpec`` ou um erro acionável."""

    spec = get_attack_spec(attack_key)
    if spec.intent_capability_id is None:
        raise ValueError(
            f"Ataque {attack_key!r} ainda não possui capacidade intent-driven. "
            "Use 'masquerade_fault' nesta versão."
        )
    try:
        return ATTACK_CAPABILITY_CATALOG[spec.intent_capability_id]
    except KeyError:
        raise RuntimeError(
            f"AttackSpec {attack_key!r} referencia capacidade inexistente: "
            f"{spec.intent_capability_id!r}."
        ) from None


def validate_intent_capability(intent: IntentSpec) -> AttackCapability:
    """Valida se a intenção pode ser compilada usando somente a allowlist."""

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
    return capability
