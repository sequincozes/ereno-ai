"""Contrato E1: IntentSpec e capability catalog do masquerade_fault."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from adversarial_ids.config.attack_capabilities import (
    ATTACK_CAPABILITY_CATALOG,
    MASQUERADE_FAULT_CAPABILITY,
    get_attack_capability,
    validate_intent_capability,
)
from adversarial_ids.config.attacks_registry import ATTACK_REGISTRY, get_attack_spec
from adversarial_ids.domain import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)
from adversarial_ids.shared.editable_fields import editable_field_paths


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "inputs" / "attacks" / "uc03_masquerade_fault.json"


def _valid_intent(**overrides: object) -> IntentSpec:
    data: dict[str, object] = {
        "source_prompt": "Reduza o recall variando a temporização da falha.",
        "objective": IntentObjective.EVADE_DETECTION,
        "base_attack": "masquerade_fault",
        "desired_effect": DesiredEffect.LOWER_RECALL,
        "intensity": IntentIntensity.MEDIUM,
        "seed": 42,
    }
    data.update(overrides)
    return IntentSpec.model_validate(data)


def test_intent_spec_is_versioned_frozen_and_json_stable():
    intent = _valid_intent()

    assert intent.schema_version == 1
    dumped = intent.model_dump(mode="json")
    assert IntentSpec.model_validate_json(json.dumps(dumped)) == intent
    with pytest.raises(ValidationError):
        intent.seed = 7


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_prompt", "   "),
        ("base_attack", "Masquerade Fault"),
        ("seed", -1),
        ("seed", 4_294_967_296),
    ],
)
def test_intent_spec_rejects_invalid_core_fields(field: str, value: object):
    with pytest.raises(ValidationError):
        _valid_intent(**{field: value})


def test_intent_spec_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _valid_intent(unexpected=True)


def test_restrictions_reject_ambiguous_or_duplicate_paths():
    with pytest.raises(ValidationError, match="permitidos e proibidos"):
        IntentRestrictions(
            allowed_fields=("fault.prob",),
            forbidden_fields=("fault.prob",),
        )

    with pytest.raises(ValidationError, match="repetidos"):
        IntentRestrictions(forbidden_fields=("fault.prob", "fault.prob"))


def test_masquerade_catalog_matches_every_editable_baseline_field():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert MASQUERADE_FAULT_CAPABILITY.editable_paths == editable_field_paths(baseline)
    assert len(MASQUERADE_FAULT_CAPABILITY.fields) == 12


def test_valid_intent_passes_deterministic_capability_gate():
    intent = _valid_intent(
        restrictions=IntentRestrictions(
            allowed_fields=("fault.durationMs.min", "fault.durationMs.max"),
            max_fields_changed=2,
        )
    )

    capability = validate_intent_capability(intent)

    assert capability.capability_id == "masquerade_fault.v1"


def test_attack_without_intent_capability_fails_with_actionable_error(monkeypatch):
    """Mantém o branch de "capacidade ausente" coberto agora que todo ataque
    registrado tem uma — injeta uma spec sem ``intent_capability_id`` em vez
    de depender de um ataque real ficar sem capacidade."""
    spec = dataclasses.replace(get_attack_spec("flooding"), intent_capability_id=None)
    monkeypatch.setitem(ATTACK_REGISTRY, "flooding", spec)

    with pytest.raises(ValueError, match="ainda não possui capacidade intent-driven"):
        get_attack_capability("flooding")


def test_every_registered_attack_has_an_intent_capability():
    for spec in ATTACK_REGISTRY.values():
        capability = get_attack_capability(spec.key)
        assert capability.attack_key == spec.key
        assert capability.capability_id == spec.intent_capability_id


def test_unknown_capability_id_on_a_registered_spec_is_a_runtime_error(monkeypatch):
    spec = dataclasses.replace(get_attack_spec("flooding"), intent_capability_id="ghost.v1")
    monkeypatch.setitem(ATTACK_REGISTRY, "flooding", spec)

    with pytest.raises(RuntimeError, match="referencia capacidade inexistente"):
        get_attack_capability("flooding")


def test_unknown_attack_fails_with_registry_options():
    intent = _valid_intent(base_attack="unknown_attack")

    with pytest.raises(ValueError, match="Ataque desconhecido"):
        validate_intent_capability(intent)


def test_unsupported_effect_is_blocked_before_compilation():
    intent = _valid_intent(desired_effect=DesiredEffect.INCREASE_RESOURCE_PRESSURE)

    with pytest.raises(ValueError, match="não é suportado"):
        validate_intent_capability(intent)


def test_unknown_allowed_field_is_blocked_by_allowlist():
    intent = _valid_intent(
        restrictions=IntentRestrictions(allowed_fields=("runtime.command",))
    )

    with pytest.raises(ValueError, match="fora da allowlist"):
        validate_intent_capability(intent)


def test_restrictions_must_leave_field_for_desired_effect():
    intent = _valid_intent(
        desired_effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        restrictions=IntentRestrictions(
            forbidden_fields=("fault.prob", "trapArea.spikeProb")
        ),
    )

    with pytest.raises(ValueError, match="removem todos os campos"):
        validate_intent_capability(intent)
