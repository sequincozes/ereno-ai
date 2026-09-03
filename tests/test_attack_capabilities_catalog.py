"""Invariantes do catálogo de capacidades intent-driven (``config/attack_capabilities.py``).

Os seis invariantes documentados no módulo — nenhum caminho inventado, nenhuma
alavanca morta, ``minimum`` como piso anti-degeneração, as três alavancas de
evasão sempre com o mesmo conjunto de campos, ``supported_effects`` como a
união dos efeitos dos campos, e probabilidades sem "prob" no nome com bounds
explícitos — valem para todo ataque registrado, não só o ``masquerade_fault``.
"""

from __future__ import annotations

import copy

import pytest

from adversarial_ids.config.attack_capabilities import (
    ATTACK_CAPABILITY_CATALOG,
    get_attack_capability,
)
from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)
from adversarial_ids.shared.editable_fields import editable_field_paths
from adversarial_ids.shared.json_io import load_json

# Documenta cada campo editável do baseline que NÃO entra no catálogo, e por
# quê (ver docstrings de config/capabilities/*.py). Uma omissão não listada
# aqui falha o teste — toda exclusão é deliberada.
_DOCUMENTED_EXCLUSIONS: dict[str, frozenset[str]] = {
    "masquerade_fault": frozenset(),
    "random_replay": frozenset(),
    "inverse_replay": frozenset(),
    "injection": frozenset(
        {
            "randomSeed",
            "stNum.min",
            "stNum.max",
            "sqNum.min",
            "sqNum.max",
            "cbStatus.values",
            "ttlMs.min",
            "ttlMs.max",
            "confRev.min",
            "confRev.max",
        }
    ),
    "high_stnum": frozenset({"sqNumDelta.min", "padBytes.min"}),
    "flooding": frozenset(),
    "grayhole": frozenset({"protectStatusChanges"}),
    "delayed_replay": frozenset({"orderBy"}),
    "delayed_replay_backoff": frozenset({"orderBy"}),
    "delayed_replay_batch_dump": frozenset({"orderBy"}),
    "delayed_replay_double_drop": frozenset({"orderBy"}),
}

_EVASION_EFFECTS = (
    DesiredEffect.LOWER_F1,
    DesiredEffect.LOWER_RECALL,
    DesiredEffect.MIMIC_NORMAL_TRAFFIC,
)


def _intent_for(
    attack_key: str,
    path: str,
    effect: DesiredEffect,
    *,
    intensity: IntentIntensity = IntentIntensity.HIGH,
) -> IntentSpec:
    return IntentSpec.model_validate(
        {
            "source_prompt": f"teste de catálogo: {attack_key}/{path}/{effect.value}",
            "objective": IntentObjective.EVADE_DETECTION.value
            if effect != DesiredEffect.INCREASE_ATTACK_ACTIVITY
            else IntentObjective.ASSESS_IDS_ROBUSTNESS.value,
            "base_attack": attack_key,
            "desired_effect": effect.value,
            "intensity": intensity.value,
            "restrictions": {
                "allowed_fields": [path],
                "max_fields_changed": 1,
            },
        }
    )


def _every_field_and_effect():
    cases = []
    for capability in ATTACK_CAPABILITY_CATALOG.values():
        for field in capability.fields:
            for effect in field.effects:
                cases.append((capability.attack_key, field.path, effect))
    return cases


# --------------------------------------------------------------------------- #
# Registry <-> catálogo                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_every_registered_attack_resolves_a_capability(attack_key):
    capability = get_attack_capability(attack_key)
    spec = get_attack_spec(attack_key)

    assert capability.attack_key == attack_key
    assert capability.capability_id == spec.intent_capability_id
    assert capability.capability_id == f"{attack_key}.v1"


def test_capability_ids_are_unique():
    ids = [c.capability_id for c in ATTACK_CAPABILITY_CATALOG.values()]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Invariante 1 — nenhum caminho inventado; toda omissão é documentada         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_capability_paths_are_a_documented_subset_of_the_baseline(attack_key):
    baseline = load_json(get_attack_spec(attack_key).baseline_path)
    baseline_paths = editable_field_paths(baseline)
    capability = get_attack_capability(attack_key)

    assert capability.editable_paths <= baseline_paths
    assert baseline_paths - capability.editable_paths == _DOCUMENTED_EXCLUSIONS[attack_key]


# --------------------------------------------------------------------------- #
# Invariante 5 — supported_effects é a união dos efeitos dos campos           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_supported_effects_is_the_union_of_field_effects(attack_key):
    capability = get_attack_capability(attack_key)
    union = frozenset().union(*(field.effects for field in capability.fields))
    assert capability.supported_effects == union


# --------------------------------------------------------------------------- #
# Invariante 4 — as três alavancas de evasão compartilham os mesmos campos    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_the_three_evasion_effects_share_the_same_candidate_fields(attack_key):
    capability = get_attack_capability(attack_key)
    path_sets = [
        {field.path for field in capability.fields_for_effect(effect)}
        for effect in _EVASION_EFFECTS
        if effect in capability.supported_effects
    ]
    if len(path_sets) > 1:
        assert all(paths == path_sets[0] for paths in path_sets)


# --------------------------------------------------------------------------- #
# increase_resource_pressure não é anunciado por nenhum catálogo (decisão     #
# registrada em core/feedback_policy.py — nenhuma métrica no DetectionReport) #
# --------------------------------------------------------------------------- #
def test_no_capability_advertises_increase_resource_pressure():
    for capability in ATTACK_CAPABILITY_CATALOG.values():
        assert DesiredEffect.INCREASE_RESOURCE_PRESSURE not in capability.supported_effects
        for field in capability.fields:
            assert DesiredEffect.INCREASE_RESOURCE_PRESSURE not in field.effects


# --------------------------------------------------------------------------- #
# Bounds sanos                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_bounds_are_sane_and_choices_include_the_baseline_value(attack_key):
    baseline = load_json(get_attack_spec(attack_key).baseline_path)
    capability = get_attack_capability(attack_key)

    for field in capability.fields:
        if field.minimum is not None and field.maximum is not None:
            assert field.minimum <= field.maximum

        node: object = baseline
        for part in field.path.split("."):
            node = node[part]  # type: ignore[index]

        if field.value_type == "string":
            assert field.choices is not None
            assert node in field.choices
            assert len(field.choices) >= 2, (
                f"{attack_key}.{field.path}: choices precisa ter uma alternativa "
                "real além do valor do baseline."
            )
        elif field.choices is not None:
            assert node in field.choices


# --------------------------------------------------------------------------- #
# Invariante 2 — nenhuma alavanca morta: todo (campo, efeito) catalogado      #
# realmente muda o valor em intensidade alta.                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("attack_key", "path", "effect"),
    _every_field_and_effect(),
    ids=[f"{k}:{p}:{e.value}" for k, p, e in _every_field_and_effect()],
)
def test_every_catalogued_field_is_a_live_lever_for_each_effect_it_claims(
    attack_key, path, effect
):
    intent = _intent_for(attack_key, path, effect)
    candidate = compile_attack_candidate(intent)

    assert [change.path for change in candidate.diff] == [path]
    change = candidate.diff[0]
    assert change.old_value != change.new_value


# --------------------------------------------------------------------------- #
# Determinismo: mesma seed -> mesmo candidato, para todo ataque catalogado    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_compilation_is_deterministic_for_every_attack(attack_key):
    capability = get_attack_capability(attack_key)
    field = capability.fields[0]
    effect = next(iter(field.effects))
    intent = _intent_for(attack_key, field.path, effect)

    first = compile_attack_candidate(intent)
    second = compile_attack_candidate(copy.deepcopy(intent))

    assert first.config == second.config
    assert first.diff == second.diff
