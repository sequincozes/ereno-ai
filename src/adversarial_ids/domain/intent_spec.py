"""Contrato versionado da intenção operacional fornecida pelo usuário."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentObjective(str, Enum):
    """Objetivo de alto nível do experimento adversarial."""

    ASSESS_IDS_ROBUSTNESS = "assess_ids_robustness"
    EVADE_DETECTION = "evade_detection"


class DesiredEffect(str, Enum):
    """Efeito mensurável que o compilador deve tentar produzir."""

    LOWER_F1 = "lower_f1"
    LOWER_RECALL = "lower_recall"
    MIMIC_NORMAL_TRAFFIC = "mimic_normal_traffic"
    INCREASE_ATTACK_ACTIVITY = "increase_attack_activity"
    INCREASE_RESOURCE_PRESSURE = "increase_resource_pressure"


class IntentIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Limite de sanidade de ``max_fields_changed`` — NÃO é a contagem de campos de
# nenhum ataque específico. Era 12 (a contagem do masquerade_fault, único
# ataque habilitado até o catálogo crescer); o random_replay tem 15 folhas
# editáveis. O teto real por ataque/efeito é calculado em
# ``core/feedback_policy._fields_ceiling``, que já faz
# ``min(MAX_FIELDS_CHANGED_CEILING, len(candidatos do efeito))`` — esta
# constante só evita um valor absurdo antes desse cálculo rodar.
MAX_FIELDS_CHANGED_CEILING = 16


class TargetValue(BaseModel):
    """Valor exato que o pedido fixou para um campo ("duração máxima de 80 ms").

    Um campo fixado sai da heurística do compilador: ``intensity`` não o move e
    nenhum clamp o ajusta — ou o valor pedido vale como está, ou a compilação
    falha dizendo por quê (ver ``core/intent_compiler.py``). Corrigir em
    silêncio o número que o usuário ditou produziria um artefato que afirma um
    valor que ninguém pediu.

    Um intervalo do pedido ("entre 50 e 80 ms") são **dois** valores fixados,
    um por limite (``fault.durationMs.min`` e ``.max``) — o contrato não tem
    tipo de intervalo próprio porque a configuração do ataque também não tem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    # União fechada, e não ``Any``: o valor atravessa JSON e é comparado com os
    # limites do catálogo de capacidades. ``tuple`` (não ``list``) pelo mesmo
    # motivo dos demais campos deste módulo — o modelo é congelado.
    value: bool | int | float | str | tuple[int, ...]

    @field_validator("path")
    @classmethod
    def _strip_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("caminho do campo não pode ser vazio")
        return normalized

    @field_validator("value", mode="before")
    @classmethod
    def _no_quoted_numbers_inside_a_list(cls, value: Any) -> Any:
        """Uma lista só é aceita já com inteiros de verdade.

        Sem isto, a resolução de união do Pydantic trata os dois casos de
        forma diferente: ``"0.42"`` num campo escalar casa com ``str`` e é
        recusado mais adiante, mas ``["20", "40"]`` casa com
        ``tuple[int, ...]`` e é **convertido** em silêncio. A conversão
        preservaria o número, mas a assimetria não se explica para quem lê o
        contrato — e "um valor ditado não é ajustado em silêncio" tem de valer
        igual para os dois. Quem ditou o valor recebe o erro e corrige.
        """

        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(
                        "lista de valores só aceita inteiros, veio "
                        f"{item!r} — escreva o número sem aspas"
                    )
        return value


class IntentRestrictions(BaseModel):
    """Restrições determinísticas aplicadas ao futuro compilador."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preserve_attack_semantics: bool = True
    max_fields_changed: int = Field(default=3, ge=1, le=MAX_FIELDS_CHANGED_CEILING)
    allowed_fields: tuple[str, ...] | None = None
    forbidden_fields: tuple[str, ...] = ()
    # Aditivo com default, como `round`/`parent_run_id` no LoopRecord: uma
    # IntentSpec gravada antes desta versão carrega normalmente, sem valor
    # fixado nenhum, e `schema_version` continua 1.
    target_values: tuple[TargetValue, ...] = ()

    @field_validator("allowed_fields", "forbidden_fields")
    @classmethod
    def _normalize_field_paths(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(field.strip() for field in value)
        if any(not field for field in normalized):
            raise ValueError("caminhos de campos não podem ser vazios")
        if len(set(normalized)) != len(normalized):
            raise ValueError("caminhos de campos não podem ser repetidos")
        return normalized

    @model_validator(mode="after")
    def _allowed_and_forbidden_must_not_overlap(self) -> "IntentRestrictions":
        overlap = set(self.allowed_fields or ()) & set(self.forbidden_fields)
        if overlap:
            fields = ", ".join(sorted(overlap))
            raise ValueError(
                f"campos não podem ser permitidos e proibidos ao mesmo tempo: {fields}"
            )
        return self

    @model_validator(mode="after")
    def _target_values_do_not_contradict_the_field_lists(self) -> "IntentRestrictions":
        """Fixar um valor é pedir que o campo mude — as três listas têm de concordar.

        Nada aqui olha o catálogo do ataque (isso é
        ``config/attack_capabilities.py``): estas são as contradições internas
        do próprio pedido, visíveis sem saber de que ataque ele fala.
        """

        pinned = [target.path for target in self.target_values]
        if len(set(pinned)) != len(pinned):
            repeated = sorted({path for path in pinned if pinned.count(path) > 1})
            raise ValueError(
                "um campo não pode ter dois valores fixados: " + ", ".join(repeated)
            )

        forbidden = set(pinned) & set(self.forbidden_fields)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise ValueError(
                f"campos não podem ter valor fixado e ser proibidos: {fields}"
            )

        if self.allowed_fields is not None:
            outside = set(pinned) - set(self.allowed_fields)
            if outside:
                fields = ", ".join(sorted(outside))
                raise ValueError(
                    "campos com valor fixado precisam estar em allowed_fields: "
                    f"{fields}"
                )

        return self


class IntentSpec(BaseModel):
    """Representa uma intenção já interpretada, antes de gerar configuração.

    O modelo é congelado e rejeita campos desconhecidos para que o artefato JSON
    possa ser persistido e reproduzido sem mutações silenciosas.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    source_prompt: str = Field(min_length=1, max_length=4000)
    objective: IntentObjective
    base_attack: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    desired_effect: DesiredEffect
    restrictions: IntentRestrictions = Field(default_factory=IntentRestrictions)
    intensity: IntentIntensity = IntentIntensity.MEDIUM
    seed: int = Field(default=42, ge=0, le=4_294_967_295)

    @field_validator("source_prompt", "base_attack")
    @classmethod
    def _strip_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("texto não pode ser vazio")
        return normalized
