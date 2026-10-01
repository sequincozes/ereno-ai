"""Testes do registro de undersamplers (épico E7).

Cobre "estratégias configuráveis com ablação": ``none`` não altera nada;
``random`` balanceia as classes de forma determinística e nunca aumenta o
total de linhas; ``ratio`` limita o desequilíbrio sem eliminá-lo. A escolha
entre elas é testada em ``tests/test_undersampling_policy.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversarial_ids.core.undersamplers import (
    NoUndersampler,
    RandomUndersampler,
    UndersamplerError,
    build_undersampler,
    class_counts,
    list_undersampler_keys,
    undersampler_spec,
)


def test_none_strategy_returns_x_and_y_unchanged():
    X = pd.DataFrame({"f": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 0, 0, 0, 1])

    resampled_X, resampled_y = NoUndersampler().fit_resample(X, y)

    assert resampled_X is X
    assert resampled_y is y


def test_fit_resample_rejects_mismatched_lengths():
    with pytest.raises(UndersamplerError, match="tamanhos diferentes"):
        RandomUndersampler().fit_resample(
            pd.DataFrame({"f": [1, 2, 3]}), pd.Series([0, 1])
        )


def test_fit_resample_rejects_empty_training_set():
    with pytest.raises(UndersamplerError, match="vazio"):
        RandomUndersampler().fit_resample(
            pd.DataFrame({"f": []}), pd.Series([], dtype=int)
        )


def test_random_strategy_balances_classes_down_to_the_minority_size():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)  # 15 normal, 5 ataque

    resampled_X, resampled_y = RandomUndersampler(random_state=42).fit_resample(
        X, y
    )

    counts = resampled_y.value_counts()
    assert counts[0] == counts[1] == 5
    assert len(resampled_X) == len(resampled_y) == 10


def test_random_strategy_never_increases_total_rows():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    resampled_X, resampled_y = RandomUndersampler(random_state=42).fit_resample(
        X, y
    )

    assert len(resampled_X) <= len(X)


def test_random_strategy_keeps_x_and_y_rows_aligned():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    resampled_X, resampled_y = RandomUndersampler(random_state=42).fit_resample(
        X, y
    )

    # A linha reamostrada de X ainda corresponde ao rótulo original em y —
    # nenhum embaralhamento desalinhou par (feature, rótulo).
    for feature_value, label in zip(resampled_X["f"], resampled_y):
        assert (feature_value < 15) == (label == 0)


def test_random_strategy_is_deterministic_for_the_same_seed():
    X = pd.DataFrame({"f": list(range(20))})
    y = pd.Series([0] * 15 + [1] * 5)

    first_X, first_y = RandomUndersampler(random_state=7).fit_resample(X, y)
    second_X, second_y = RandomUndersampler(random_state=7).fit_resample(X, y)

    pd.testing.assert_frame_equal(first_X, second_X)
    pd.testing.assert_series_equal(first_y, second_y)


def test_random_strategy_different_seeds_can_pick_different_rows():
    X = pd.DataFrame({"f": list(range(100))})
    y = pd.Series([0] * 90 + [1] * 10)

    a_X, _ = RandomUndersampler(random_state=1).fit_resample(X, y)
    b_X, _ = RandomUndersampler(random_state=2).fit_resample(X, y)

    assert list(a_X["f"]) != list(b_X["f"])


def test_random_strategy_is_a_noop_when_classes_are_already_balanced():
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    resampled_X, resampled_y = RandomUndersampler(random_state=42).fit_resample(
        X, y
    )

    assert len(resampled_X) == 4
    assert resampled_y.value_counts()[0] == resampled_y.value_counts()[1] == 2


def test_class_counts_uses_string_keys():
    y = pd.Series([0, 0, 1, 1, 1])

    assert class_counts(y) == {"0": 2, "1": 3}


# --------------------------------------------------------------------------- #
# Estratégia "ratio": limita o desequilíbrio em vez de eliminá-lo              #
# --------------------------------------------------------------------------- #
def test_ratio_strategy_caps_the_majority_at_the_declared_multiple():
    X = pd.DataFrame({"f": list(range(110))})
    y = pd.Series([0] * 100 + [1] * 10)

    resampled_X, resampled_y = RandomUndersampler(
        random_state=42, target_ratio=3.0
    ).fit_resample(X, y)

    counts = resampled_y.value_counts()
    assert counts[1] == 10
    assert counts[0] == 30
    assert len(resampled_X) == 40


def test_ratio_of_one_is_exactly_the_random_strategy():
    # A razão de ser do registro ter duas chaves para a mesma classe: "random"
    # é o caso particular de "ratio" com teto igual à minoritária. Se as duas
    # divergissem, uma das duas documentações estaria mentindo.
    X = pd.DataFrame({"f": list(range(60))})
    y = pd.Series([0] * 50 + [1] * 10)

    balanced_X, balanced_y = RandomUndersampler(random_state=3).fit_resample(X, y)
    ratio_X, ratio_y = RandomUndersampler(random_state=3, target_ratio=1.0).fit_resample(X, y)

    pd.testing.assert_frame_equal(balanced_X, ratio_X)
    pd.testing.assert_series_equal(balanced_y, ratio_y)


def test_ratio_keeps_more_benign_rows_than_balancing_does():
    # O argumento inteiro da estratégia: com prevalência baixa, balancear
    # descarta quase todo o tráfego benigno.
    X = pd.DataFrame({"f": list(range(1000))})
    y = pd.Series([0] * 990 + [1] * 10)

    _, balanced_y = RandomUndersampler(random_state=42).fit_resample(X, y)
    _, capped_y = RandomUndersampler(random_state=42, target_ratio=5.0).fit_resample(X, y)

    assert balanced_y.value_counts()[0] == 10
    assert capped_y.value_counts()[0] == 50


def test_ratio_never_empties_a_class_even_with_a_single_minority_row():
    X = pd.DataFrame({"f": list(range(21))})
    y = pd.Series([0] * 20 + [1] * 1)

    _, resampled_y = RandomUndersampler(random_state=42, target_ratio=1.5).fit_resample(X, y)

    # 1 * 1.5 trunca para 1 — e o piso de uma linha é o que impede o modelo de
    # treinar sem nunca ver uma das classes que existem no treino.
    assert set(resampled_y) == {0, 1}
    assert resampled_y.value_counts()[0] >= 1


def test_target_ratio_below_one_is_refused():
    with pytest.raises(UndersamplerError, match="inverter o desequilíbrio"):
        RandomUndersampler(target_ratio=0.5)


# --------------------------------------------------------------------------- #
# Registro                                                                     #
# --------------------------------------------------------------------------- #
def test_registry_exposes_the_three_strategies_in_order():
    assert list_undersampler_keys() == ("none", "random", "ratio")


def test_unknown_undersampler_names_the_registered_ones():
    with pytest.raises(UndersamplerError, match="Undersampler desconhecido"):
        undersampler_spec("tomek_links")


def test_build_refuses_ratio_without_a_target_ratio():
    with pytest.raises(UndersamplerError, match="exige target_ratio"):
        build_undersampler("ratio")


def test_build_refuses_a_target_ratio_where_it_does_not_apply():
    # Aceitar e ignorar seria o caminho para um manifest registrando um corte
    # que a estratégia não faz.
    with pytest.raises(UndersamplerError, match="não usa target_ratio"):
        build_undersampler("random", target_ratio=3.0)


def test_build_none_returns_the_identity_undersampler():
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([0, 0, 1])

    resampled_X, resampled_y = build_undersampler("none").fit_resample(X, y)

    assert resampled_X is X
    assert resampled_y is y


def test_built_strategies_report_the_key_they_were_built_from():
    assert build_undersampler("random").key == "random"
    assert build_undersampler("ratio", target_ratio=4.0).key == "ratio"
