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


def _describe_value(value: Any) -> str:
    """Nome do tipo como o pedido o expressa, para a mensagem de erro."""

    if isinstance(value, bool):
        return "booleano"
    if isinstance(value, int):
        return "inteiro"
    if isinstance(value, float):
        return "número"
    if isinstance(value, str):
        return "texto"
    if isinstance(value, tuple):
        return "lista"
    return type(value).__name__


def _type_mismatch(field: FieldCapability, value: Any) -> str | None:
    """``None`` quando o valor tem o tipo que o campo declara.

    ``bool`` é subclasse de ``int`` em Python, então ``isinstance(True, int)``
    é verdadeiro — sem a checagem explícita, ``True`` passaria como valor de um
    campo inteiro e chegaria ao JSON do ERENO como ``true``.
    """

    is_bool = isinstance(value, bool)
    if field.value_type == "boolean":
        return None if is_bool else f"esperava booleano, veio {_describe_value(value)}"
    if field.value_type == "integer":
        if is_bool or not isinstance(value, int):
            return f"esperava inteiro, veio {_describe_value(value)}"
        return None
    if field.value_type == "number":
        # Um inteiro é um número válido: o pedido diz "probabilidade 1", não
        # "1.0", e recusar isso seria exigir sintaxe de ponto flutuante do texto.
        if is_bool or not isinstance(value, (int, float)):
            return f"esperava número, veio {_describe_value(value)}"
        return None
    if field.value_type == "string":
        return None if isinstance(value, str) else f"esperava texto, veio {_describe_value(value)}"
    if field.value_type == "integer_list":
        if not isinstance(value, tuple) or not value:
            return f"esperava lista de inteiros não vazia, veio {_describe_value(value)}"
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            return "esperava lista só de inteiros"
        return None
    return f"tipo de campo não suportado: {field.value_type!r}"


def _out_of_bounds(field: FieldCapability, value: Any) -> str | None:
    """``None`` quando o valor respeita ``choices``/``minimum``/``maximum``.

    É o mesmo piso que protege a heurística de compilar o ataque para fora de
    existência (invariante 3 do catálogo). Um valor ditado pelo usuário não
    ganha licença para atravessá-lo: o gerador não ficaria mais permissivo só
    porque o número veio do prompt em vez da escada de intensidade.
    """

    if field.choices is not None and value not in field.choices:
        options = ", ".join(repr(choice) for choice in field.choices)
        return f"valor fora das opções aceitas ({options})"

    numbers = value if isinstance(value, tuple) else (value,)
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in numbers):
        return None
    for item in numbers:
        if field.minimum is not None and item < field.minimum:
            return f"{item} está abaixo do mínimo {field.minimum} do campo"
        if field.maximum is not None and item > field.maximum:
            return f"{item} está acima do máximo {field.maximum} do campo"
    return None


def _pinned_pair_is_inverted(pinned: dict[str, Any]) -> str | None:
    """``min`` e ``max`` fixados no mesmo par precisam continuar sendo um intervalo.

    Só o caso em que os **dois** limites vêm do pedido é decidível aqui — é uma
    contradição do próprio texto ("entre 80 e 50 ms"), visível sem abrir a
    baseline do ataque. O caso de um limite fixado contra o irmão vindo da
    baseline precisa da config e fica no compilador
    (``_assert_pinned_ranges``), que é quem a tem em mãos.
    """

    for path, value in pinned.items():
        parent, _, leaf = path.rpartition(".")
        if leaf != "min":
            continue
        sibling = f"{parent}.max" if parent else "max"
        upper = pinned.get(sibling)
        if upper is None:
            continue
        if isinstance(value, (int, float)) and isinstance(upper, (int, float)):
            if value >= upper:
                return (
                    f"o intervalo pedido em {parent or leaf!r} está invertido: "
                    f"min={value} precisa ser estritamente menor que max={upper}"
                )
    return None


def validate_target_values(
    capability: AttackCapability, intent: IntentSpec
) -> dict[str, Any]:
    """Valida os valores fixados contra o catálogo; devolve ``{caminho: valor}``.

    Roda no mesmo portão determinístico que a allowlist de campos — o LLM
    propõe o número, o catálogo decide se ele existe. O que **não** é decidido
    aqui é tudo que depende da baseline do ataque (um limite fixado contra o
    irmão que ninguém fixou); isso é do compilador.
    """

    pinned = {target.path: target.value for target in intent.restrictions.target_values}
    if not pinned:
        return {}

    fields_by_path = {field.path: field for field in capability.fields}
    problems: list[str] = []
    for path in sorted(pinned):
        field = fields_by_path[path]
        complaint = _type_mismatch(field, pinned[path]) or _out_of_bounds(field, pinned[path])
        if complaint:
            problems.append(f"{path}: {complaint}")
    if problems:
        raise ValueError(
            f"Valores fixados inválidos para {intent.base_attack!r}: "
            + "; ".join(problems)
        )

    inverted = _pinned_pair_is_inverted(pinned)
    if inverted:
        raise ValueError(inverted)

    budget = intent.restrictions.max_fields_changed
    if len(pinned) > budget:
        raise ValueError(
            f"O pedido fixa {len(pinned)} campos, mas max_fields_changed é "
            f"{budget} — um campo com valor fixado é um campo alterado. "
            f"Use max_fields_changed >= {len(pinned)}."
        )

    return pinned


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
    pinned_paths = {target.path for target in intent.restrictions.target_values}
    unknown_paths = (requested_paths | forbidden_paths | pinned_paths) - known_paths
    if unknown_paths:
        fields = ", ".join(sorted(unknown_paths))
        raise ValueError(
            f"Campos fora da allowlist de {intent.base_attack!r}: {fields}."
        )

    validate_target_values(capability, intent)

    candidates = {
        field.path for field in capability.fields_for_effect(intent.desired_effect)
    }
    if requested_paths:
        candidates &= requested_paths
    candidates -= forbidden_paths
    # Um campo com valor fixado é sempre aplicado, carregue ele o efeito ou
    # não: o pedido disse qual valor quer, o que é mais forte que a heurística
    # que escolhe campos *capazes* do efeito. Por isso ele também salva o caso
    # abaixo — um pedido inteiramente ditado não "removeu todos os campos",
    # ele nomeou os campos um por um.
    if not candidates and not pinned_paths:
        raise ValueError(
            "As restrições removem todos os campos capazes de produzir o efeito "
            f"{intent.desired_effect.value!r}."
        )
    return capability, frozenset(candidates)


def validate_intent_capability(intent: IntentSpec) -> AttackCapability:
    """Valida se a intenção pode ser compilada usando somente a allowlist."""

    capability, _ = resolve_candidate_paths(intent)
    return capability
