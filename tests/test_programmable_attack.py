"""Ataque programável (uc11) — comportamentos fora do catálogo (Fase 2.1).

O comportamento é uma lista de regras de mutação (op/campo/valor/fração por
slot), não uma classe por comportamento. Estes testes fixam, sem Java nem Groq:

- o schema congelado das regras (dict de slots, não array);
- a capacidade: só `fraction` é alavanca de intensidade, `op`/`field`/`value`
  são author-only (efeito vazio) — o LLM os preenche via `target_values`;
- o compilador materializa as regras a partir da intenção;
- o ataque está registrado e resolve capacidade/schema/playbook como os demais.

A geração real com o JAR (a classe `programmable` no trace) é verificada à parte,
fora da suíte — ver `docs/programmable_attack.md`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adversarial_ids.config.attack_capabilities import (
    get_attack_capability,
    resolve_candidate_paths,
)
from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.config.defense_techniques import playbook_for_attack
from adversarial_ids.core.intent_compiler import compile_attack_candidate
from adversarial_ids.domain.attack_configs import config_model_for
from adversarial_ids.domain.attack_configs.programmable import (
    MutationRule,
    ProgrammableConfig,
)
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentSpec,
)
from adversarial_ids.shared.json_io import load_json

BEHAVIOUR = "força o disjuntor em 1 em 30% das mensagens"


def _intent(target_values, *, effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
            objective=IntentObjective.ASSESS_IDS_ROBUSTNESS, max_fields=8,
            intensity=IntentIntensity.MEDIUM):
    return IntentSpec.model_validate(
        {
            "source_prompt": BEHAVIOUR,
            "objective": objective,
            "base_attack": "programmable",
            "desired_effect": effect,
            "intensity": intensity,
            "restrictions": {
                "max_fields_changed": max_fields,
                "target_values": [
                    {"path": p, "value": v} for p, v in target_values.items()
                ],
            },
        }
    )


# --------------------------------------------------------------------------- #
# Registro e resolução                                                        #
# --------------------------------------------------------------------------- #
def test_programmable_is_a_registered_attack():
    assert "programmable" in list_attack_keys()
    spec = get_attack_spec("programmable")
    assert spec.label == "programmable"
    assert spec.segment_name == "uc11_programmable"


def test_programmable_resolves_capability_schema_and_playbook():
    assert get_attack_capability("programmable").attack_key == "programmable"
    assert config_model_for("programmable") is ProgrammableConfig
    assert "programmable" in playbook_for_attack("programmable").attack_keys


def test_the_baseline_validates_against_the_schema():
    baseline = load_json(get_attack_spec("programmable").baseline_path)
    model = ProgrammableConfig.model_validate(baseline)
    assert model.attackType == "programmable"
    assert set(model.rules) == {"r0", "r1"}


# --------------------------------------------------------------------------- #
# Schema das regras                                                           #
# --------------------------------------------------------------------------- #
def test_rules_are_a_dict_of_slots_not_an_array():
    cfg = ProgrammableConfig.model_validate(
        {
            "attackType": "programmable",
            "enabled": True,
            "rules": {"r0": {"op": "set", "field": "cbStatus", "value": 1, "fraction": 0.5}},
        }
    )
    assert isinstance(cfg.rules, dict)
    assert cfg.rules["r0"].op == "set"


def test_an_empty_rules_object_is_rejected():
    with pytest.raises(ValidationError):
        ProgrammableConfig.model_validate(
            {"attackType": "programmable", "enabled": True, "rules": {}}
        )


def test_a_rule_with_an_unknown_op_is_rejected():
    with pytest.raises(ValidationError):
        MutationRule.model_validate(
            {"op": "toggle", "field": "cbStatus", "value": 1, "fraction": 0.5}
        )


def test_a_rule_fraction_outside_zero_one_is_rejected():
    with pytest.raises(ValidationError):
        MutationRule.model_validate(
            {"op": "set", "field": "cbStatus", "value": 1, "fraction": 1.5}
        )


# --------------------------------------------------------------------------- #
# Capacidade: fraction é alavanca, op/field/value são author-only             #
# --------------------------------------------------------------------------- #
def test_only_fraction_fields_carry_an_effect():
    capability = get_attack_capability("programmable")
    for field in capability.fields:
        if field.path.endswith(".fraction"):
            assert field.effects  # alavanca de intensidade
        else:
            assert field.effects == frozenset()  # author-only via target_values


def test_the_heuristic_sweeps_only_the_fraction_fields():
    """Sem valores ditados, a intenção só move os `fraction` — o comportamento
    (op/field/value) fica como no baseline."""

    intent = _intent({}, max_fields=8)
    candidate = compile_attack_candidate(intent)

    assert candidate.diff  # mudou algo
    assert all(change.path.endswith(".fraction") for change in candidate.diff)


# --------------------------------------------------------------------------- #
# Compilador materializa o comportamento autorado                            #
# --------------------------------------------------------------------------- #
def test_the_llm_authors_a_behaviour_through_target_values():
    intent = _intent(
        {
            "rules.r0.op": "scale",
            "rules.r0.field": "sqNum",
            "rules.r0.value": 3,
            "rules.r0.fraction": 0.4,
        },
        max_fields=4,
    )

    config = compile_attack_candidate(intent).config

    assert config["rules"]["r0"] == {
        "op": "scale", "field": "sqNum", "value": 3, "fraction": 0.4
    }


def test_an_authored_value_is_never_clamped():
    """Um campo mutável fora da allowlist do creator é recusado pelo portão."""

    intent = _intent({"rules.r0.field": "frameLen"}, max_fields=1)
    with pytest.raises(ValueError, match="fora das opções"):
        resolve_candidate_paths(intent)


def test_the_compiled_config_validates_against_the_schema():
    intent = _intent(
        {"rules.r0.op": "add", "rules.r0.field": "confRev", "rules.r0.value": 2,
         "rules.r0.fraction": 0.6},
        max_fields=4,
    )
    candidate = compile_attack_candidate(intent)

    # compile_attack_candidate já valida internamente; reconfirmar é barato.
    ProgrammableConfig.model_validate(candidate.config)
    assert candidate.attack_key == "programmable"


def test_the_same_authored_intent_compiles_deterministically():
    tv = {"rules.r0.op": "scale", "rules.r0.field": "stNum",
          "rules.r0.value": 2, "rules.r0.fraction": 0.5}
    first = compile_attack_candidate(_intent(tv, max_fields=4))
    second = compile_attack_candidate(_intent(tv, max_fields=4))

    assert first.config == second.config
    assert first.diff == second.diff
