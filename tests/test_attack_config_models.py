"""Cobertura dos schemas por ataque em ``domain/attack_configs/``.

Cada ataque registrado tem seu próprio modelo Pydantic fechado
(``extra="forbid"``); este arquivo garante que o registry e os modelos nunca
divergem e que as primitivas compartilhadas (``IntRange``/``FloatRange``/
``ProbabilityRange``) preservam as invariantes que ``AttackConfig`` (agora
``MasqueradeFaultConfig``) já garantia.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.domain.attack_configs import (
    CONFIG_MODEL_BY_ATTACK,
    FloatRange,
    IntRange,
    MasqueradeFaultConfig,
    ProbabilityRange,
    ProbabilityValue,
    config_model_for,
)
from adversarial_ids.shared.json_io import load_json


# --------------------------------------------------------------------------- #
# Registry <-> modelo: nenhum ataque fica sem schema, nenhum baseline rejeita  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_every_registered_attack_has_a_config_model(attack_key):
    assert config_model_for(attack_key) is not None


@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_every_baseline_validates_against_its_own_model(attack_key):
    spec = get_attack_spec(attack_key)
    raw = load_json(spec.baseline_path)
    model = config_model_for(attack_key)

    instance = model.model_validate(raw)

    dumped = instance.model_dump(by_alias=True)
    assert dumped == raw


def test_config_model_for_unknown_attack_returns_none():
    assert config_model_for("nao_existe") is None


# --------------------------------------------------------------------------- #
# MasqueradeFaultConfig — golden byte-stability (a config também é o          #
# serializador do golden data/iteration_history.json)                        #
# --------------------------------------------------------------------------- #
def test_masquerade_fault_config_dump_matches_the_baseline_bytes():
    """``IterationRecord`` grava ``model_dump(mode="json")`` deste modelo no
    golden — ordem de campos e int-vs-float no dump precisam bater com o
    arquivo baseline, ou ``data/iteration_history.json`` muda de conteúdo."""
    spec = get_attack_spec("masquerade_fault")
    raw = load_json(spec.baseline_path)

    dumped = MasqueradeFaultConfig.model_validate(raw).model_dump(mode="json")

    assert json.dumps(dumped) == json.dumps(raw)


def test_masquerade_fault_config_rejects_unknown_field():
    spec = get_attack_spec("masquerade_fault")
    raw = load_json(spec.baseline_path)
    raw["campoInexistente"] = 123

    with pytest.raises(ValidationError):
        MasqueradeFaultConfig.model_validate(raw)


# --------------------------------------------------------------------------- #
# Cada modelo rejeita chave extra e par min/max invertido                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_every_model_rejects_an_unknown_top_level_field(attack_key):
    spec = get_attack_spec(attack_key)
    raw = load_json(spec.baseline_path)
    raw["campoBemMasBemInexistente"] = "x"

    with pytest.raises(ValidationError):
        config_model_for(attack_key).model_validate(raw)


# --------------------------------------------------------------------------- #
# Primitivas de range (domain/attack_configs/ranges.py)                      #
# --------------------------------------------------------------------------- #
def test_int_range_rejects_min_greater_than_max():
    with pytest.raises(ValidationError):
        IntRange(min=10, max=5)


def test_int_range_rejects_negative_values():
    with pytest.raises(ValidationError):
        IntRange(min=-1, max=5)


def test_int_range_preserves_int_literals_through_json_dump():
    """Fixa o risco documentado em ranges.py: IntRange não pode virar float
    no dump, ou o golden do masquerade_fault muda de conteúdo."""
    r = IntRange(min=50, max=800)
    dumped = r.model_dump(mode="json")
    assert dumped == {"min": 50, "max": 800}
    assert isinstance(dumped["min"], int)
    assert isinstance(dumped["max"], int)


def test_float_range_rejects_negative_values():
    with pytest.raises(ValidationError):
        FloatRange(min=-0.1, max=1.0)


def test_probability_range_rejects_value_above_one():
    with pytest.raises(ValidationError):
        ProbabilityRange(min=0.1, max=1.5)


def test_probability_value_rejects_value_outside_unit_interval():
    with pytest.raises(ValidationError):
        ProbabilityValue(value=1.5)


# --------------------------------------------------------------------------- #
# Armadilhas de forma específicas — mesmo caminho, tipos incompatíveis        #
# --------------------------------------------------------------------------- #
def test_cb_status_is_a_scalar_in_masquerade_fault_but_an_object_in_injection():
    masquerade_raw = load_json(get_attack_spec("masquerade_fault").baseline_path)
    injection_raw = load_json(get_attack_spec("injection").baseline_path)

    assert isinstance(masquerade_raw["cbStatus"], int)
    assert isinstance(injection_raw["cbStatus"], dict)

    with pytest.raises(ValidationError):
        config_model_for("injection").model_validate(
            {**injection_raw, "cbStatus": masquerade_raw["cbStatus"]}
        )
    with pytest.raises(ValidationError):
        config_model_for("masquerade_fault").model_validate(
            {**masquerade_raw, "cbStatus": injection_raw["cbStatus"]}
        )


def test_injection_and_grayhole_baselines_have_no_enabled_key():
    for attack_key in ("injection", "grayhole"):
        raw = load_json(get_attack_spec(attack_key).baseline_path)
        assert "enabled" not in raw
        # confirma que o modelo não exige/aceita o campo por engano
        assert "enabled" not in config_model_for(attack_key).model_fields


def test_every_registered_config_model_is_a_distinct_class():
    assert len(set(CONFIG_MODEL_BY_ATTACK.values())) == len(CONFIG_MODEL_BY_ATTACK)
    assert len(CONFIG_MODEL_BY_ATTACK) == len(list_attack_keys())
