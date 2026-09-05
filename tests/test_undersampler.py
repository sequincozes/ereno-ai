"""Testes do RandomUndersampler (épico E7).

Cobre "estratégias configuráveis com ablação": ``strategy="none"`` não altera
nada; ``strategy="random"`` balanceia as classes de forma determinística e
nunca aumenta o total de linhas.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversarial_ids.core.undersampler import RandomUndersampler, UndersamplerError


def test_none_strategy_returns_x_and_y_unchanged():
    X = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 0, 0, 1])

    resampled_X, resampled_y = RandomUndersampler(strategy="none").fit_resample(X, y)

    assert resampled_X is X
    assert resampled_y is y


def test_fit_resample_rejects_mismatched_lengths():
    with pytest.raises(UndersamplerError, match="tamanhos diferentes"):
        RandomUndersampler(strategy="random").fit_resample(
            pd.DataFrame({"f": [1, 2, 3]}), pd.Series([0, 1])
        )


def test_fit_resample_rejects_empty_training_set():
    with pytest.raises(UndersamplerError, match="vazio"):
        RandomUndersampler(strategy="random").fit_resample(
            pd.DataFrame({"f": []}), pd.Series([], dtype=int)
        )


def test_random_strategy_balances_classes_down_to_the_minority_size():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)  # 15 normal, 5 ataque

    resampled_X, resampled_y = RandomUndersampler(strategy="random", random_state=42).fit_resample(
        X, y
    )

    counts = resampled_y.value_counts()
    assert counts[0] == counts[1] == 5
    assert len(resampled_X) == len(resampled_y) == 10


def test_random_strategy_never_increases_total_rows():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    resampled_X, resampled_y = RandomUndersampler(strategy="random", random_state=42).fit_resample(
        X, y
    )

    assert len(resampled_X) <= len(X)


def test_random_strategy_keeps_x_and_y_rows_aligned():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    resampled_X, resampled_y = RandomUndersampler(strategy="random", random_state=42).fit_resample(
        X, y
    )

    # A linha reamostrada de X ainda corresponde ao rótulo original em y —
    # nenhum embaralhamento desalinhou par (feature, rótulo).
    for feature_value, label in zip(resampled_X["f"], resampled_y):
        assert (feature_value < 15) == (label == 0)


def test_random_strategy_is_deterministic_for_the_same_seed():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    first_X, first_y = RandomUndersampler(strategy="random", random_state=7).fit_resample(X, y)
    second_X, second_y = RandomUndersampler(strategy="random", random_state=7).fit_resample(X, y)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)


def test_random_strategy_different_seeds_can_pick_different_rows():
    X = pd.DataFrame({"f": list(range(100))})
    y = pd.Series([0] * 90 + [1] * 10)

    a_X, _ = RandomUndersampler(strategy="random", random_state=1).fit_resample(X, y)
    b_X, _ = RandomUndersampler(strategy="random", random_state=2).fit_resample(X, y)

    assert list(a_X["f"]) != list(b_X["f"])


def test_random_strategy_is_a_noop_when_classes_are_already_balanced():
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    resampled_X, resampled_y = RandomUndersampler(strategy="random", random_state=42).fit_resample(
        X, y
    )

    assert len(resampled_X) == 4
    assert resampled_y.value_counts()[0] == resampled_y.value_counts()[1] == 2


def test_class_counts_uses_string_keys():
    y = pd.Series([0, 0, 1, 1, 1])

    assert RandomUndersampler.class_counts(y) == {"0": 2, "1": 3}
