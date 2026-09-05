"""Testes do FeatureSelector (épico E7).

Cobre o critério de pronto do E7: "estratégias configuráveis com ablação" —
``strategy="none"`` preserva o comportamento anterior ao épico,
``strategy="mutual_info"`` seleciona um subconjunto determinístico e
reproduzível dado o mesmo ``random_state``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversarial_ids.core.feature_selector import FeatureSelector, FeatureSelectorError


def _frame(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_none_strategy_keeps_every_candidate_feature_in_original_order():
    X = _frame(b=[1, 2, 3, 4], a=[4, 3, 2, 1])
    y = pd.Series([0, 0, 1, 1])

    selector = FeatureSelector(strategy="none").fit(X, y)

    assert selector.selected_features == ["b", "a"]
    assert selector.candidate_features == ["b", "a"]
    assert selector.scores == {}


def test_mutual_info_requires_a_stopping_criterion():
    with pytest.raises(FeatureSelectorError, match="top_k e/ou min_score"):
        FeatureSelector(strategy="mutual_info")


def test_top_k_rejects_non_positive_values():
    with pytest.raises(FeatureSelectorError, match="top_k deve ser >= 1"):
        FeatureSelector(strategy="mutual_info", top_k=0)


def test_fit_raises_on_empty_dataframe():
    with pytest.raises(FeatureSelectorError, match="vazio"):
        FeatureSelector().fit(_frame(num=[]), pd.Series([], dtype=int))


def test_fit_raises_when_x_and_y_length_mismatch():
    with pytest.raises(FeatureSelectorError, match="tamanhos diferentes"):
        FeatureSelector().fit(_frame(num=[1, 2, 3]), pd.Series([0, 1]))


def test_transform_before_fit_raises():
    with pytest.raises(FeatureSelectorError, match="antes de fit"):
        FeatureSelector().transform(_frame(num=[1]))


def test_mutual_info_top_k_selects_the_most_informative_features_and_preserves_column_order():
    # `signal` reproduz `y` perfeitamente; `noise` é constante e não carrega
    # informação nenhuma sobre a classe — mutual information deve rankear
    # `signal` muito acima de `noise`.
    n = 60
    y = pd.Series([i % 2 for i in range(n)])
    X = _frame(
        noise=[1] * n,
        signal=[label * 10 for label in y],
        extra=[i % 3 for i in range(n)],
    )

    selector = FeatureSelector(strategy="mutual_info", top_k=1, random_state=42).fit(X, y)

    assert selector.selected_features == ["signal"]
    assert set(selector.scores) == {"noise", "signal", "extra"}
    assert selector.scores["signal"] > selector.scores["noise"]


def test_mutual_info_preserves_candidate_column_order_not_ranking_order():
    n = 60
    y = pd.Series([i % 2 for i in range(n)])
    X = _frame(
        b_signal=[label * 10 for label in y],
        a_noise=[1] * n,
    )

    selector = FeatureSelector(strategy="mutual_info", top_k=2, random_state=42).fit(X, y)

    # top_k=2 mantém as duas — a ordem final é a ordem original das colunas
    # (b_signal antes de a_noise), não a ordem de ranking por pontuação.
    assert selector.selected_features == ["b_signal", "a_noise"]


def test_mutual_info_min_score_filters_out_uninformative_features():
    n = 60
    y = pd.Series([i % 2 for i in range(n)])
    X = _frame(
        noise=[1] * n,
        signal=[label * 10 for label in y],
    )

    # `noise` é constante, mas o estimador KNN de mutual_info_classif injeta
    # ruído numérico interno para desempatar — o score dele não é exatamente
    # 0.0, só muito menor que o de `signal` (~0.02 vs. ~0.70 nesta amostra).
    # O corte fica no meio dos dois, não em zero.
    selector = FeatureSelector(strategy="mutual_info", min_score=0.3, random_state=42).fit(X, y)

    assert "signal" in selector.selected_features
    assert "noise" not in selector.selected_features


def test_mutual_info_raises_when_nothing_survives_the_cutoff():
    n = 30
    y = pd.Series([i % 2 for i in range(n)])
    X = _frame(noise=[1] * n)

    with pytest.raises(FeatureSelectorError, match="Nenhuma feature sobreviveu"):
        FeatureSelector(strategy="mutual_info", min_score=999.0, random_state=42).fit(X, y)


def test_mutual_info_is_deterministic_for_the_same_seed():
    n = 80
    y = pd.Series([i % 2 for i in range(n)])
    X = _frame(
        signal=[label * 10 for label in y],
        noise_a=[i % 5 for i in range(n)],
        noise_b=[i % 7 for i in range(n)],
    )

    first = FeatureSelector(strategy="mutual_info", top_k=2, random_state=7).fit(X, y)
    second = FeatureSelector(strategy="mutual_info", top_k=2, random_state=7).fit(X, y)

    assert first.selected_features == second.selected_features
    assert first.scores == second.scores


def test_transform_keeps_only_selected_columns():
    train = _frame(signal=[1, 2, 3, 4], noise=[1, 1, 1, 1])
    y = pd.Series([0, 0, 1, 1])

    selector = FeatureSelector(strategy="mutual_info", top_k=1, random_state=42).fit(train, y)
    transformed = selector.transform(train)

    assert list(transformed.columns) == selector.selected_features
    assert len(transformed) == len(train)
