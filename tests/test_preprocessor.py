"""Testes do FeaturePreprocessor (épico E6).

Cobre o critério de pronto do E6: "fit/transform sem leakage; manifest
reproduzível" — em especial os casos em que uma decisão de preprocessamento
(coluna constante, vocabulário categórico, escala) só pode depender da
partição de treino, nunca do dataset inteiro.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversarial_ids.core.preprocessor import FeaturePreprocessor, PreprocessorError
from adversarial_ids.domain.feature_manifest import FeatureManifest


def _frame(**columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_fit_transform_imputes_and_encodes():
    train = _frame(
        num=[1.0, None, 3.0],
        cat=["a", "b", None],
    )

    preprocessor = FeaturePreprocessor()
    transformed = preprocessor.fit_transform(train, label_column="class")

    assert transformed["num"].tolist() == [1.0, 0.0, 3.0]
    # códigos ordinais de sorted(unique(["a", "b", "missing"])) -> a=0, b=1, missing=2
    assert transformed["cat"].tolist() == [0, 1, 2]
    assert preprocessor.feature_columns == ["num", "cat"]


def test_fit_drops_always_drop_and_constant_columns():
    train = _frame(
        Time=[1.0, 2.0, 3.0],  # sempre-descartar
        constant=[5, 5, 5],
        useful=[1, 2, 3],
    )

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")

    assert preprocessor.feature_columns == ["useful"]
    reasons = {column.name: column.reason for column in preprocessor.manifest().dropped_columns}
    assert reasons == {"Time": "always_drop", "constant": "constant"}


def test_fit_drops_cb_status_only_when_flag_set():
    train = _frame(cbStatus=[0, 1, 0], cbStatusDiff=[0, 1, -1], useful=[1, 2, 3])

    kept = FeaturePreprocessor(drop_cb_status=False).fit(train, label_column="class")
    assert set(kept.feature_columns) == {"cbStatus", "cbStatusDiff", "useful"}

    dropped = FeaturePreprocessor(drop_cb_status=True).fit(train, label_column="class")
    assert dropped.feature_columns == ["useful"]
    reasons = {column.name: column.reason for column in dropped.manifest().dropped_columns}
    assert reasons == {"cbStatus": "cb_status", "cbStatusDiff": "cb_status"}


def test_fit_drops_all_nan_columns_as_constant():
    # NaN conta como um único valor distinto para nunique(dropna=False), então
    # uma coluna 100% NaN já cai no check de "constant" — não há um reason
    # "all_nan" separado (ver comentário em FeaturePreprocessor.fit).
    train = _frame(empty=[None, None, None], useful=[1, 2, 3])

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")

    assert preprocessor.feature_columns == ["useful"]
    reasons = {column.name: column.reason for column in preprocessor.manifest().dropped_columns}
    assert reasons == {"empty": "constant"}


def test_fit_raises_when_no_feature_survives():
    train = _frame(Time=[1.0, 2.0, 3.0], constant=[1, 1, 1])

    with pytest.raises(PreprocessorError, match="Nenhuma feature sobrou"):
        FeaturePreprocessor().fit(train, label_column="class")


def test_fit_raises_on_empty_dataframe():
    with pytest.raises(PreprocessorError, match="vazio"):
        FeaturePreprocessor().fit(_frame(num=[]), label_column="class")


def test_transform_before_fit_raises():
    with pytest.raises(PreprocessorError, match="antes de fit"):
        FeaturePreprocessor().transform(_frame(num=[1]))


def test_manifest_before_fit_raises():
    with pytest.raises(PreprocessorError, match="antes de fit"):
        FeaturePreprocessor().manifest()


# --------------------------------------------------------------------- #
# Anti-leakage: a razão de existir do épico                              #
# --------------------------------------------------------------------- #


def test_column_constant_only_in_train_partition_is_dropped():
    # `leaky` teria variância se o fit olhasse o dataset inteiro (0,0,0,1,1)
    # — mas é constante dentro da partição de treino (as 3 primeiras linhas).
    # Fitar sobre o dataframe inteiro NUNCA descartaria essa coluna; fitar só
    # no treino (o que o chamador deve fazer, via train_test_split antes do
    # fit) descarta — é o comportamento anti-leakage que o E6 exige.
    whole_dataset = _frame(leaky=[0, 0, 0, 1, 1], useful=[1, 2, 3, 4, 5])
    train = whole_dataset.iloc[:3]

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")

    assert preprocessor.feature_columns == ["useful"]
    reasons = {column.name: column.reason for column in preprocessor.manifest().dropped_columns}
    assert reasons == {"leaky": "constant"}


def test_fitted_rows_is_train_partition_size_not_dataset_total():
    whole_dataset = _frame(num=list(range(10)))
    train = whole_dataset.iloc[:7]

    manifest = FeaturePreprocessor().fit(train, label_column="class").manifest()

    assert manifest.fitted_rows == 7


def test_unseen_category_in_transform_becomes_unseen_code():
    train = _frame(cat=["a", "b"])
    test = _frame(cat=["a", "c"])  # "c" nunca visto no fit

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")
    transformed = preprocessor.transform(test)

    assert transformed["cat"].tolist() == [preprocessor.manifest().categorical_codes["cat"]["a"], -1]


def test_missing_feature_column_at_transform_is_filled():
    train = _frame(num=[1.0, 2.0, 3.0])
    test = pd.DataFrame({"other": [1, 2]})  # não tem a coluna "num"

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")
    transformed = preprocessor.transform(test)

    assert transformed["num"].tolist() == [0.0, 0.0]


def test_standard_scaler_uses_only_train_statistics():
    train = _frame(num=[0.0, 10.0])  # média=5, std!=0
    test = _frame(num=[100.0])

    preprocessor = FeaturePreprocessor(scaler="standard").fit(train, label_column="class")
    stat = preprocessor.manifest().scaler_stats["num"]
    assert stat.mean == 5.0

    transformed = preprocessor.transform(test)
    assert transformed["num"].iloc[0] == pytest.approx((100.0 - stat.mean) / stat.std)


def test_standard_scaler_zero_train_variance_after_imputation_does_not_divide_by_zero():
    # `nunique(dropna=False)` conta NaN como valor distinto de 5.0, então esta
    # coluna passa pelo check de constante (nunique=2) — mas a imputação por
    # mediana faz as duas linhas virarem 5.0, zerando a variância no treino
    # DEPOIS do fit decidir mantê-la. Sem a salvaguarda, isto seria ZeroDivisionError.
    train = _frame(num=[5.0, None])

    preprocessor = FeaturePreprocessor(
        numeric_imputation="median", scaler="standard"
    ).fit(train, label_column="class")

    stat = preprocessor.manifest().scaler_stats["num"]
    assert stat.mean == 5.0
    assert stat.std == 1.0  # salvaguarda: nunca 0


def test_categorical_column_constant_in_train_is_dropped_even_with_two_categories_overall():
    # `cat` teria 2 categorias se o fit olhasse o dataset inteiro — mas só "a"
    # cai na partição de treino, então nunique(dropna=False)<=1 a classifica
    # como "constant" (não sobra como categórica de vocabulário {"a": 0}).
    # Uma categórica só sobrevive ao fit com >=2 valores distintos NO TREINO.
    whole_dataset = _frame(cat=["a", "a", "a", "b", "b"], num=[1, 2, 3, 4, 5])
    train = whole_dataset.iloc[:3]

    preprocessor = FeaturePreprocessor().fit(train, label_column="class")

    assert "cat" not in preprocessor.feature_columns
    reasons = {column.name: column.reason for column in preprocessor.manifest().dropped_columns}
    assert reasons["cat"] == "constant"


def test_categorical_cardinality_above_ceiling_raises():
    high_cardinality = _frame(cat=[f"id_{i}" for i in range(1001)])

    with pytest.raises(PreprocessorError, match="acima do teto"):
        FeaturePreprocessor().fit(high_cardinality, label_column="class")


def test_median_imputation_uses_train_median():
    train = _frame(num=[1.0, None, 3.0, 100.0])

    preprocessor = FeaturePreprocessor(numeric_imputation="median").fit(train, label_column="class")

    assert preprocessor.manifest().numeric_fill_values["num"] == pytest.approx(3.0)


# --------------------------------------------------------------------- #
# Manifest: determinismo e round-trip                                    #
# --------------------------------------------------------------------- #


def test_manifest_is_deterministic_for_the_same_fit():
    train = _frame(num=[1.0, 2.0, None], cat=["a", "b", "a"])

    manifest_a = FeaturePreprocessor().fit(train, label_column="class").manifest()
    manifest_b = FeaturePreprocessor().fit(train, label_column="class").manifest()

    assert manifest_a.model_dump() == manifest_b.model_dump()


def test_manifest_round_trips_through_json(tmp_path):
    train = _frame(num=[1.0, 2.0, None], cat=["a", "b", "a"])
    manifest = FeaturePreprocessor().fit(train, label_column="class").manifest()

    path = tmp_path / "feature_manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    reloaded = FeatureManifest.model_validate_json(path.read_text(encoding="utf-8"))

    assert reloaded == manifest


def test_from_manifest_reproduces_transform_without_refitting():
    train = _frame(num=[1.0, None, 30.0], cat=["a", "b", "a"])
    test = _frame(num=[None, 7.0], cat=["a", "z"])

    original = FeaturePreprocessor().fit(train, label_column="class")
    reconstructed = FeaturePreprocessor.from_manifest(original.manifest())

    pd.testing.assert_frame_equal(original.transform(test), reconstructed.transform(test))


def test_from_manifest_reconstructed_preprocessor_cannot_be_transformed_before_matching_columns():
    # Sanity: from_manifest() já vem "fitted" — transform funciona de cara,
    # sem chamar fit() de novo.
    train = _frame(num=[1.0, 2.0])
    manifest = FeaturePreprocessor().fit(train, label_column="class").manifest()

    reconstructed = FeaturePreprocessor.from_manifest(manifest)
    transformed = reconstructed.transform(_frame(num=[5.0]))

    assert transformed["num"].tolist() == [5.0]
