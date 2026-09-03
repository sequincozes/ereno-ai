"""AttackCandidate — proposta de configuração compilada a partir de uma intenção.

Contrato congelado (ação 72h #2). Materializa o resultado do compilador
spec→AttackConfig (épico E2): a intenção de origem, a configuração completa
resultante, o diff explícito em relação à baseline e a justificativa. A
allowlist de campos e as regras do registry são validadas por
``validate_intent_capability`` antes de um ``AttackCandidate`` existir — este
contrato só garante a consistência interna do artefato já compilado.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.intent_spec import IntentSpec


class FieldChange(BaseModel):
    """Uma alteração atômica em relação à configuração baseline do ataque."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    old_value: Any
    new_value: Any


class AttackCandidate(BaseModel):
    """Configuração de ataque compilada, pronta para o ``GeneratorRunner``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_intent: IntentSpec
    capability_id: str = Field(min_length=1)
    attack_key: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    config: dict[str, Any]
    diff: tuple[FieldChange, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _attack_key_matches_source_intent(self) -> "AttackCandidate":
        if self.attack_key != self.source_intent.base_attack:
            raise ValueError(
                "attack_key deve ser igual a source_intent.base_attack "
                f"({self.attack_key!r} != {self.source_intent.base_attack!r})."
            )
        return self

    @model_validator(mode="after")
    def _diff_paths_are_unique(self) -> "AttackCandidate":
        paths = [change.path for change in self.diff]
        if len(set(paths)) != len(paths):
            raise ValueError("diff não pode alterar o mesmo campo mais de uma vez.")
        return self
