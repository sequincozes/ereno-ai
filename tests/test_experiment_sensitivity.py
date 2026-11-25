"""Testes do E0 campo a campo (sem Java): variantes, deriva e classificação."""

from __future__ import annotations

import math

import pandas as pd

from adversarial_ids.config.attack_capabilities import get_attack_capability
from adversarial_ids.experiments.sensitivity import (
    DIRECTIONS,
    classify,
    field_variants,
    moved_features,
    separation,
)


def test_every_variant_changes_only_its_own_field():
    for variant in field_variants("masquerade_fault"):
        assert variant.config is not None, variant.error
        changed = {change["path"] for change in variant.diff}
        assert changed == {variant.path}


def test_variants_cover_each_field_in_each_admitted_direction():
    capability = get_attack_capability("high_stnum")
    expected = {
        (field.path, direction)
        for field in capability.fields
        for direction, effect in DIRECTIONS.items()
        if effect in field.effects
    }
    got = {(v.path, v.direction) for v in field_variants("high_stnum")}
    assert got == expected


def test_separation_is_zero_for_identical_and_infinite_for_constant_shift():
    same = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    shifted = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    base = pd.DataFrame({"a": [4.0, 4.0, 4.0]})
    assert separation(same, same)["a"] == 0.0
    assert math.isinf(separation(shifted, base)["a"])


def test_a_feature_moves_only_above_its_own_noise_and_the_floor():
    drift = {"timing": 0.3, "quiet": 0.04, "noisy": 0.5}
    noise = {"timing": 0.1, "quiet": 0.0, "noisy": 0.4}
    assert moved_features(drift, noise) == ["timing"]


def test_classification():
    assert classify([True, True], [[], []]) == "inert"
    assert classify([False, False], [["x"], ["y"]]) == "effective"
    assert classify([False, False], [["x"], []]) == "inconclusive"
    assert classify([False, True], [[], []]) == "inconclusive"
