"""FeatureManifest — estado ajustado de um ``FeaturePreprocessor`` (épico E6).

Contrato congelado. Não é o preprocessador modular completo em si (isso é
``core/preprocessor.py``) — é o retrato do que um ``fit`` produziu: colunas
descartadas (e por quê), quais features são numéricas/categóricas, a
estratégia e os valores de imputação, os códigos categóricos e a escala
ajustados, e quantas linhas de **treino** alimentaram o ajuste. Reconstruível
sozinho via ``FeaturePreprocessor.from_manifest`` — sem pickle — o que é o que
torna o manifest "reproduzível" no sentido do critério de pronto do E6
("fit/transform sem leakage; manifest reproduzível").

``fitted_rows`` é a prova, no próprio artefato, de que o ajuste não vazou:
sempre o tamanho da partição de treino, nunca o total do dataset. Undersampling
e seleção de features (mutual information) não pertencem a este contrato —
são o épico E7 (``domain/selection_manifest.py::SelectionManifest``), que
consome ``feature_columns`` como ponto de partida.

Precedência de motivo em ``dropped_columns``: uma coluna pode se qualificar
para mais de um motivo ao mesmo tempo (no dataset real do ERENO, 21 das 33
colunas descartadas são simultaneamente ``always_drop`` e ``constant``) — o
motivo registrado segue sempre ``always_drop`` > ``cb_status`` > ``constant``,
nessa ordem, nunca uma combinação.

Ordem de transform: para cada feature numérica, imputação (``missing_column_fill``
quando a coluna nem existe no dataframe de entrada) e só depois, se
``scaler="standard"``, a padronização — um valor ausente é tratado como
qualquer outro valor bruto, não como "já padronizado". Para cada categórica,
sentinela de ausência seguida do código ordinal; nunca escalada.

Limitação conhecida (não é leakage, mas é honesto documentar): o tipo
numérico-vs-categórico de cada coluna vem do dtype que ``pandas.read_csv``
infere sobre o **arquivo inteiro**, antes de qualquer split — uma coluna não
pode trocar de tipo entre ``fit`` e ``transform`` (o que tornaria o pipeline
instável), mas a decisão de tipo, diferentemente de tudo o mais aqui, não é
exclusiva da partição de treino.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DroppedColumn(BaseModel):
    """Uma coluna descartada do dataframe original e o motivo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    reason: Literal["always_drop", "constant", "cb_status"]


class ScalerStat(BaseModel):
    """Média e desvio-padrão ajustados no treino para uma feature numérica.

    ``std`` nunca é ``0.0``: uma feature quase-constante no treino (variância
    zero após imputação) é ajustada com ``std=1.0`` em vez de deixar o
    transform dividir por zero — ver ``FeaturePreprocessor.fit``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    std: float = Field(gt=0.0)


class FeatureManifest(BaseModel):
    """Estado ajustado de um ``FeaturePreprocessor``, pronto para persistir."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    label_column: str = Field(min_length=1)
    feature_columns: tuple[str, ...] = Field(min_length=1)
    dropped_columns: tuple[DroppedColumn, ...] = ()
    numeric_features: tuple[str, ...] = ()
    categorical_features: tuple[str, ...] = ()

    numeric_imputation: Literal["zero", "median"] = "zero"
    numeric_fill_values: dict[str, float] = Field(default_factory=dict)

    categorical_imputation: Literal["missing_sentinel"] = "missing_sentinel"
    categorical_codes: dict[str, dict[str, int]] = Field(default_factory=dict)

    scaler: Literal["none", "standard"] = "none"
    scaler_stats: dict[str, ScalerStat] = Field(default_factory=dict)

    # Semântica de transform fixa, explícita no artefato em vez de implícita
    # no código: categoria nunca vista no fit -> este código; coluna ausente
    # no dataframe transformado -> este valor (bruto, antes de imputação/escala).
    unseen_category_code: Literal[-1] = -1
    missing_column_fill: Literal[0.0] = 0.0

    # Linhas de TREINO que alimentaram o fit — nunca o total do dataset. É a
    # prova, no próprio artefato, de que o ajuste não viu a partição de teste.
    fitted_rows: int = Field(gt=0)

    # sha256 de ``X_treino.to_csv(index=False, lineterminator="\n")`` — hash
    # do conteúdo exato que foi ajustado, mesma convenção de
    # ``DatasetBundle.content_hash`` (hash sobre o conteúdo real, não sobre um
    # resumo). Dois fits sobre dados idênticos produzem o mesmo hash.
    fitted_content_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _numeric_and_categorical_partition_feature_columns(self) -> "FeatureManifest":
        numeric = set(self.numeric_features)
        categorical = set(self.categorical_features)

        if numeric & categorical:
            raise ValueError(
                "numeric_features e categorical_features não podem se sobrepor "
                f"({sorted(numeric & categorical)!r})."
            )
        if (numeric | categorical) != set(self.feature_columns):
            raise ValueError(
                "numeric_features ∪ categorical_features deve cobrir exatamente "
                "feature_columns."
            )
        return self

    @model_validator(mode="after")
    def _dropped_columns_are_not_features(self) -> "FeatureManifest":
        names = [column.name for column in self.dropped_columns]
        if len(set(names)) != len(names):
            raise ValueError("dropped_columns não pode repetir o nome de uma coluna.")

        overlap = set(self.feature_columns) & set(names)
        if overlap:
            raise ValueError(
                f"feature_columns e dropped_columns não podem se sobrepor ({sorted(overlap)!r})."
            )
        return self

    @model_validator(mode="after")
    def _numeric_fill_values_cover_exactly_numeric_features(self) -> "FeatureManifest":
        if set(self.numeric_fill_values) != set(self.numeric_features):
            raise ValueError(
                "numeric_fill_values deve cobrir exatamente numeric_features "
                f"(faltam {sorted(set(self.numeric_features) - set(self.numeric_fill_values))!r}, "
                f"sobram {sorted(set(self.numeric_fill_values) - set(self.numeric_features))!r})."
            )
        return self

    @model_validator(mode="after")
    def _categorical_codes_cover_exactly_categorical_features(self) -> "FeatureManifest":
        if set(self.categorical_codes) != set(self.categorical_features):
            raise ValueError(
                "categorical_codes deve cobrir exatamente categorical_features "
                f"(faltam {sorted(set(self.categorical_features) - set(self.categorical_codes))!r}, "
                f"sobram {sorted(set(self.categorical_codes) - set(self.categorical_features))!r})."
            )
        return self

    @model_validator(mode="after")
    def _scaler_stats_reference_known_numeric_features(self) -> "FeatureManifest":
        unknown = set(self.scaler_stats) - set(self.numeric_features)
        if unknown:
            raise ValueError(
                f"scaler_stats referencia colunas fora de numeric_features ({sorted(unknown)!r})."
            )
        return self

    @model_validator(mode="after")
    def _scaler_standard_requires_stats_for_every_numeric_feature(self) -> "FeatureManifest":
        if self.scaler == "standard" and set(self.scaler_stats) != set(self.numeric_features):
            raise ValueError(
                "scaler='standard' exige scaler_stats para exatamente todas as numeric_features "
                f"(faltam {sorted(set(self.numeric_features) - set(self.scaler_stats))!r})."
            )
        return self

    @model_validator(mode="after")
    def _fitted_values_are_finite(self) -> "FeatureManifest":
        values = list(self.numeric_fill_values.values())
        values += [stat.mean for stat in self.scaler_stats.values()]
        values += [stat.std for stat in self.scaler_stats.values()]
        if any(math.isnan(value) or math.isinf(value) for value in values):
            raise ValueError(
                "numeric_fill_values e scaler_stats não podem conter NaN/Infinity — "
                "um valor ajustado não-finito não é reproduzível."
            )
        return self
