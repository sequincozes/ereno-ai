"""Compilador determinístico ``IntentSpec`` → ``AttackCandidate`` (épico E2).

Traduz uma intenção já validada (``validate_intent_capability`` já passou) em
uma configuração completa de ataque, pronta para o ``GeneratorRunner``. Não há
LLM neste caminho — a seleção de campos e o cálculo dos novos valores são
determinísticos e reproduzíveis pela ``seed`` da intenção, conforme o
guardrail central do plano de 60 dias: o LLM produz a intenção tipada, este
módulo materializa a proposta, e apenas validadores decidem o resultado.

Heurística de direção
----------------------
Cada ``DesiredEffect`` mapeia para uma direção (``increase``/``decrease``)
aplicada aos campos numéricos escolhidos: efeitos de evasão (``lower_f1``,
``lower_recall``, ``mimic_normal_traffic``) reduzem a magnitude do campo,
aproximando-o da borda inferior; efeitos de atividade/pressão
(``increase_attack_activity``, ``increase_resource_pressure``) aumentam a
magnitude, aproximando-o da borda superior. Campos booleanos e de escolha
(``cbStatus``, ``incrementStNumOnFault``, ``sqnumMode``) alternam para o valor
diferente da baseline, já que a direção contínua não se aplica a eles. Esta é
uma heurística de MVP — o vínculo entre o valor escolhido e o efeito
observado é o que a etapa DETECTOR mede e a etapa DEFENDER referencia; o
compilador não promete o efeito, apenas produz uma variante plausível e
rastreável.

Valores fixados pelo pedido
---------------------------
``IntentSpec.restrictions.target_values`` fura essa heurística campo a campo:
um campo com valor fixado ("duração máxima de 80 ms") recebe exatamente o valor
pedido — não entra no sorteio da seed, não recebe o passo da intensidade e não
é clampado contra o irmão do par. Ele **conta** na cota de
``max_fields_changed``, e o que sobra da cota é o que a heurística pode usar.

A ausência de clamp é deliberada: um valor ditado que não cabe (fora dos
limites do catálogo, ou que inverte um par ``{min, max}``) faz a compilação
**falhar**, em vez de ser ajustado em silêncio. Corrigir o número produziria
uma config que afirma um valor que o usuário não pediu, e o ``diff`` e a
justificativa — que o Defensor lê — passariam a mentir sobre a origem dele.
O que é decidível sem a baseline é recusado antes, no portão de capacidades
(``config/attack_capabilities.py::validate_target_values``); o resto é
``_assert_pinned_ranges`` aqui.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from adversarial_ids.config.attack_capabilities import (
    AttackCapability,
    FieldCapability,
    resolve_candidate_paths,
    validate_target_values,
)
from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.domain.attack_candidate import AttackCandidate, FieldChange
from adversarial_ids.domain.attack_configs import config_model_for
from adversarial_ids.domain.intent_spec import DesiredEffect, IntentIntensity, IntentSpec
from adversarial_ids.shared.json_io import load_json

_DIRECTION_BY_EFFECT: dict[DesiredEffect, str] = {
    DesiredEffect.LOWER_F1: "decrease",
    DesiredEffect.LOWER_RECALL: "decrease",
    DesiredEffect.MIMIC_NORMAL_TRAFFIC: "decrease",
    DesiredEffect.INCREASE_ATTACK_ACTIVITY: "increase",
    DesiredEffect.INCREASE_RESOURCE_PRESSURE: "increase",
}

_STEP_BY_INTENSITY: dict[IntentIntensity, float] = {
    IntentIntensity.LOW: 0.2,
    IntentIntensity.MEDIUM: 0.5,
    IntentIntensity.HIGH: 0.85,
}


def _flip_direction(direction: str) -> str:
    return "decrease" if direction == "increase" else "increase"

# Campos numéricos pareados cuja ordenação (min <= max) o schema por ataque
# valida (ver ``domain/attack_configs/``). Ao aplicar mudanças em ordem
# alfabética de path, "max" é resolvido antes de "min" — o valor recém-
# calculado do campo em ``diff`` é clampado contra o irmão já presente na
# config de trabalho (baseline, se não estiver no diff).
#
# Detecção **estrutural**, não uma lista de caminhos: os 11 baselines têm mais
# de 20 objetos ``{min, max}`` (``burst``, ``dropRate``, ``networkDelayMs``,
# ``padBytes``, ...) e enumerá-los repetiria o problema que
# ``shared/validator._enforce_min_le_max`` já resolve estruturalmente para o
# loop legado — a diferença é o escopo: lá a varredura é global sobre a config
# final, aqui é local ao objeto pai do campo que está mudando.
_RANGE_SIBLING_LEAF = {"min": "max", "max": "min"}


def _paired_sibling_path(config: dict[str, Any], path: str) -> tuple[str, str] | None:
    """Irmão de um par ``{min, max}``: ``(caminho_do_irmão, folha)`` ou ``None``."""

    parent_path, _, leaf = path.rpartition(".")
    sibling_leaf = _RANGE_SIBLING_LEAF.get(leaf)
    if sibling_leaf is None:
        return None

    parent = _get_by_path(config, parent_path) if parent_path else config
    if not isinstance(parent, dict):
        return None

    sibling_value = parent.get(sibling_leaf)
    if not isinstance(sibling_value, (int, float)) or isinstance(sibling_value, bool):
        return None

    sibling_path = f"{parent_path}.{sibling_leaf}" if parent_path else sibling_leaf
    return sibling_path, leaf


class IntentCompilerError(ValueError):
    """Erro acionável: a intenção não pôde ser compilada num AttackCandidate."""


def _get_by_path(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for key in path.split("."):
        node = node[key]
    return node


def _set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _select_field_paths(
    capability: AttackCapability,
    candidates: frozenset[str],
    intent: IntentSpec,
    *,
    limit: int,
) -> tuple[str, ...]:
    """Escolhe até ``limit`` campos candidatos, reprodutível pela seed.

    ``limit`` vem de fora (e não mais de ``max_fields_changed`` direto) porque
    os campos com valor fixado já consumiram parte da cota antes de a
    heurística escolher qualquer coisa.
    """

    if limit <= 0:
        return ()
    ordered = sorted(candidates)
    if len(ordered) <= limit:
        return tuple(ordered)

    rng = random.Random(intent.seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    return tuple(sorted(shuffled[:limit]))


def _assert_pinned_ranges(config: dict[str, Any], pinned_paths: set[str]) -> None:
    """Todo par ``{min, max}`` tocado por um valor fixado ainda é um intervalo.

    ``_clamp_paired`` resolve o caso da heurística aproximando um limite do
    irmão. Um valor fixado não pode ser aproximado de nada — ou ele vale como
    pedido, ou o pedido é impossível. Os dois casos que chegam aqui:

    - o pedido fixou um limite e o irmão ficou na baseline (``min=2000`` contra
      ``max=1000``);
    - o pedido fixou um limite e a heurística mexeu no irmão sem conseguir
      preservar a ordem.

    Falhar aqui é a razão de o compilador não "consertar" o número: o pedido
    volta para o usuário como contradição declarada, em vez de virar um ataque
    que ninguém pediu (mesma disciplina do piloto E0 — um intervalo degenerado
    faz o ERENO recusar a execução inteira, ver ``docs/pilot_e0.md``).
    """

    for path in sorted(pinned_paths):
        resolved = _paired_sibling_path(config, path)
        if resolved is None:
            continue
        sibling_path, leaf = resolved
        value = _get_by_path(config, path)
        sibling = _get_by_path(config, sibling_path)
        if _keeps_order(leaf, value, sibling):
            continue
        raise IntentCompilerError(
            f"O valor fixado em {path!r} ({value}) não forma um intervalo com "
            f"{sibling_path!r} ({sibling}): 'min' precisa ser estritamente "
            "menor que 'max'. Fixe também o outro limite ou escolha um valor "
            "dentro do intervalo da baseline."
        )


def _round_like(reference: Any, value: float) -> Any:
    if isinstance(reference, bool):
        return reference
    if isinstance(reference, int):
        return int(round(value))
    return round(value, 6)


def _compute_numeric_value(
    field: FieldCapability, baseline: Any, direction: str, step: float
) -> Any:
    if field.minimum is not None and field.maximum is not None:
        bound = field.maximum if direction == "increase" else field.minimum
        new_value = baseline + step * (bound - baseline)
    elif direction == "decrease" and field.minimum is not None:
        new_value = baseline - step * (baseline - field.minimum)
        new_value = max(new_value, field.minimum)
    else:
        factor = (1 + step) if direction == "increase" else (1 - step)
        new_value = baseline * factor
        if field.minimum is not None:
            new_value = max(new_value, field.minimum)

    return _round_like(baseline, new_value)


def _compute_list_value(baseline: list[int], direction: str, step: float) -> list[int]:
    factor = (1 + step) if direction == "increase" else (1 - step)
    return [max(1, int(round(item * factor))) for item in baseline]


def _compute_choice_value(field: FieldCapability, baseline: Any) -> Any:
    for choice in field.choices or ():
        if choice != baseline:
            return choice
    raise IntentCompilerError(
        f"Campo {field.path!r} não tem alternante definido em 'choices'."
    )


def _compute_new_value(
    field: FieldCapability, baseline: Any, direction: str, step: float
) -> Any:
    if field.value_type == "boolean":
        return not baseline
    if field.value_type in ("integer", "number"):
        if field.choices:
            return _compute_choice_value(field, baseline)
        return _compute_numeric_value(field, baseline, direction, step)
    if field.value_type == "integer_list":
        return _compute_list_value(baseline, direction, step)
    if field.value_type == "string":
        return _compute_choice_value(field, baseline)
    raise IntentCompilerError(f"Tipo de campo não suportado: {field.value_type!r}")


def _keeps_order(leaf: str, value: Any, sibling: Any) -> bool:
    """O par continua sendo um intervalo: ``min`` estritamente abaixo de ``max``."""

    return value < sibling if leaf == "min" else value > sibling


def _clamp_paired(
    config: dict[str, Any], path: str, value: Any, *, baseline: Any, step: float
) -> Any:
    """Aproxima um limite do par do irmão vigente, preservando ``min < max``.

    ``_select_field_paths`` devolve os caminhos em ordem alfabética, então
    "...max" é sempre resolvido antes de "...min". Cada um é resolvido contra
    o valor *corrente* do irmão no ``config`` de trabalho — a baseline se o
    irmão não foi selecionado, o valor recém-escrito se foi (``_set_by_path``
    já mutou o dict). A invariante vale ao fim do laço para qualquer
    subconjunto do par.

    A invariante é ``<`` estrito, e não ``<=``. A versão anterior travava o
    limite **em cima** do irmão, e ``min == max`` passa por tudo que este repo
    tem: o clamp aceita, o schema do ataque em ``domain/attack_configs/``
    valida, o portão de capacidade não olha pares — e o ERENO recusa o
    processo inteiro com *"The lower limit (2000) must be less than the upper
    limit (2000)"*. Era o caso do ``random_replay`` em intensidade alta, que
    fechava ``windowS`` em ``{2.0, 2.0}`` (medido no piloto E0, 18/09/2026;
    ver ``docs/pilot_e0.md``). Um intervalo degenerado não é um intervalo.

    Quando o valor calculado cruzaria o irmão, **o irmão vira o limite efetivo**
    e o mesmo passo da intensidade é reaplicado contra ele. Isso evita inventar
    um épsilon — a distância vem da política de escalada que já está em uso — e
    mantém a escada monotônica: intensidade maior chega mais perto do irmão sem
    nunca encostar. Só o arredondamento de inteiro pode colar no irmão; aí o
    valor recua uma unidade, e se nem isso couber o campo não se move (o
    chamador o descarta do ``diff``, que passa a dizer a verdade sobre o que
    mudou).
    """

    resolved = _paired_sibling_path(config, path)
    if resolved is None:
        return value

    sibling_path, leaf = resolved
    sibling_value = _get_by_path(config, sibling_path)
    if _keeps_order(leaf, value, sibling_value):
        return value

    approached = _round_like(baseline, baseline + step * (sibling_value - baseline))
    if _keeps_order(leaf, approached, sibling_value):
        return approached

    retreat = sibling_value - 1 if leaf == "min" else sibling_value + 1
    if isinstance(baseline, int) and not isinstance(baseline, bool):
        within_direction = (
            baseline <= retreat if leaf == "min" else retreat <= baseline
        )
        if within_direction and _keeps_order(leaf, retreat, sibling_value):
            return retreat

    return baseline


def compile_attack_candidate(intent: IntentSpec) -> AttackCandidate:
    """Compila uma ``IntentSpec`` autorizada numa ``AttackCandidate`` completa.

    Assume que ``intent`` já passou por ``validate_intent_capability`` (o
    IntentAgent garante isso antes de expor a intenção) — este compilador
    revalida a allowlist internamente, então uma intenção rejeitada nunca
    chega a produzir um candidato.
    """

    capability, candidates = resolve_candidate_paths(intent)
    spec = get_attack_spec(intent.base_attack)
    baseline: dict[str, Any] = load_json(spec.baseline_path)

    # Os campos com valor fixado saem da heurística inteira: não entram no
    # sorteio, não recebem o passo da intensidade e não são clampados. O que
    # sobra da cota de ``max_fields_changed`` é o que a heurística pode usar.
    pinned = validate_target_values(capability, intent)
    budget = intent.restrictions.max_fields_changed - len(pinned)
    selected_paths = _select_field_paths(
        capability, frozenset(candidates - set(pinned)), intent, limit=budget
    )
    direction = _DIRECTION_BY_EFFECT[intent.desired_effect]
    step = _STEP_BY_INTENSITY[intent.intensity]

    fields_by_path = {field.path: field for field in capability.fields}
    config = copy.deepcopy(baseline)
    diff: list[FieldChange] = []

    # Os fixados primeiro, para que o clamp da heurística enxergue o valor
    # pedido ao resolver o irmão de um par — e ceda a ele, nunca o contrário.
    for path in sorted(pinned):
        old_value = _get_by_path(config, path)
        # O contrato guarda ``integer_list`` como tupla (o modelo é congelado);
        # a config do ERENO é JSON e fala em lista. Sem a conversão, o campo
        # entraria no ``diff`` como mudança mesmo quando o valor é o mesmo.
        new_value = list(pinned[path]) if isinstance(pinned[path], tuple) else pinned[path]
        if new_value == old_value:
            continue
        _set_by_path(config, path, new_value)
        diff.append(FieldChange(path=path, old_value=old_value, new_value=new_value))

    for path in selected_paths:
        field = fields_by_path[path]
        old_value = _get_by_path(config, path)
        # Campos "inverse" (gaps, intervalos) ficam mais agressivos quando o
        # valor DIMINUI — a direção efetiva é o oposto da direção do efeito.
        field_direction = _flip_direction(direction) if field.polarity == "inverse" else direction
        new_value = _compute_new_value(field, old_value, field_direction, step)
        new_value = _clamp_paired(config, path, new_value, baseline=old_value, step=step)
        if new_value == old_value:
            continue
        _set_by_path(config, path, new_value)
        diff.append(FieldChange(path=path, old_value=old_value, new_value=new_value))

    _assert_pinned_ranges(config, set(pinned))

    if not diff:
        detail = " Os valores fixados já são os da baseline." if pinned else ""
        raise IntentCompilerError(
            "Nenhum campo mudou de valor — a intenção não produziu uma "
            "configuração distinta da baseline." + detail
        )

    config_model = config_model_for(spec.key)
    if config_model is None:
        raise IntentCompilerError(
            f"Ataque {spec.key!r} não tem schema de config registrado em "
            "domain/attack_configs/ — não é possível validar o candidato."
        )
    try:
        config_model.model_validate(config)
    except Exception as exc:  # noqa: BLE001 - relançado como erro acionável
        raise IntentCompilerError(
            f"Configuração compilada é inválida: {exc}"
        ) from exc

    changed_fields = ", ".join(change.path for change in diff)
    # A justificativa separa as duas origens de propósito: ela é o que o
    # Defensor e o relatório leem para saber *por que* a config é essa, e
    # "o usuário pediu 80" não é a mesma afirmação que "a intensidade alta
    # calculou 80".
    pinned_in_diff = [change.path for change in diff if change.path in pinned]
    origin = (
        f" {len(pinned_in_diff)} deles com valor fixado no pedido "
        f"({', '.join(pinned_in_diff)}), fora da escada de intensidade."
        if pinned_in_diff
        else ""
    )
    rationale = (
        f"Compilado deterministicamente da intenção {intent.objective.value}/"
        f"{intent.desired_effect.value} (intensidade {intent.intensity.value}, "
        f"seed {intent.seed}, direção {direction}): "
        f"{len(diff)} campo(s) ajustado(s) — {changed_fields}.{origin}"
    )

    return AttackCandidate(
        source_intent=intent,
        capability_id=capability.capability_id,
        attack_key=intent.base_attack,
        config=config,
        diff=tuple(diff),
        rationale=rationale,
    )
