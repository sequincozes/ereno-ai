"""SelectionManifest — estado ajustado de seleção de features + undersampling (épico E7).

Contrato congelado, irmão de ``FeatureManifest`` (E6) e explicitamente fora
dele — como os dois docstrings do E6 já anunciavam, undersampling e seleção
de features por mutual information "consomem ``FeatureManifest.feature_columns``
como ponto de partida", não fazem parte do fit/transform do preprocessador.

Ordem do pipeline (``core/ids_evaluator.py::IdsEvaluator.train_baseline``):

1. ``FeaturePreprocessor.fit_transform`` (E6) sobre a partição de treino cria
   o espaço de features candidatas — é o que este manifest chama de
   ``candidate_features``.
2. ``core/feature_selector.py::FeatureSelector`` ajusta sobre **todo** o
   treino (antes de qualquer undersampling): mutual information é mais
   estável com mais linhas, e a classe minoritária de ataque não deve perder
   peso na pontuação de relevância antes mesmo de decidir quais features
   sobrevivem.
3. Só depois ``core/undersampler.py::RandomUndersampler`` reamostra as
   linhas de treino (já no espaço de features selecionado) para balancear as
   classes antes de ``model.fit`` — nunca a partição de teste/variante, que
   precisa continuar refletindo a distribuição real para a métrica fazer
   sentido.

``fitted_rows_before_undersampling``/``fitted_rows_after_undersampling`` são
a mesma prova de anti-leakage que ``FeatureManifest.fitted_rows``: contam
linhas de TREINO, nunca do dataset inteiro nem da partição de teste.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SelectionManifest(BaseModel):
    """Estado ajustado de ``FeatureSelector`` + ``RandomUndersampler``, pronto para persistir."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    # -- Seleção de features (mutual information) -------------------------- #
    feature_selection: Literal["none", "mutual_info"] = "none"
    feature_selection_top_k: int | None = Field(default=None, gt=0)
    feature_selection_min_score: float | None = None
    candidate_features: tuple[str, ...] = Field(min_length=1)
    selected_features: tuple[str, ...] = Field(min_length=1)
    # Só preenchido quando feature_selection="mutual_info" — estratégia
    # "none" não calcula pontuação nenhuma (ver FeatureSelector.fit).
    feature_scores: dict[str, float] = Field(default_factory=dict)

    # -- Undersampling da partição de treino -------------------------------- #
    undersampling: Literal["none", "random"] = "none"
    undersampling_random_state: int = 42
    # Chaves são o rótulo ORIGINAL (ex.: "normal", "masquerade_fault"), não o
    # código inteiro do LabelEncoder — o manifest deve ser legível sozinho.
    class_counts_before: dict[str, int] = Field(default_factory=dict)
    class_counts_after: dict[str, int] = Field(default_factory=dict)

    # Linhas de TREINO antes/depois do undersampling — nunca teste nem total.
    fitted_rows_before_undersampling: int = Field(gt=0)
    fitted_rows_after_undersampling: int = Field(gt=0)

    @model_validator(mode="after")
    def _selected_is_subset_of_candidate(self) -> "SelectionManifest":
        unknown = set(self.selected_features) - set(self.candidate_features)
        if unknown:
            raise ValueError(
                f"selected_features tem colunas fora de candidate_features ({sorted(unknown)!r})."
            )
        return self

    @model_validator(mode="after")
    def _feature_scores_match_strategy(self) -> "SelectionManifest":
        if self.feature_selection == "none":
            if self.feature_scores:
                raise ValueError(
                    "feature_scores só é preenchido quando feature_selection='mutual_info'."
                )
        elif set(self.feature_scores) != set(self.candidate_features):
            raise ValueError(
                "feature_scores deve cobrir exatamente candidate_features quando "
                "feature_selection='mutual_info' (faltam "
                f"{sorted(set(self.candidate_features) - set(self.feature_scores))!r}, sobram "
                f"{sorted(set(self.feature_scores) - set(self.candidate_features))!r})."
            )
        return self

    @model_validator(mode="after")
    def _mutual_info_requires_a_stopping_criterion(self) -> "SelectionManifest":
        if self.feature_selection == "mutual_info" and (
            self.feature_selection_top_k is None and self.feature_selection_min_score is None
        ):
            raise ValueError(
                "feature_selection='mutual_info' exige feature_selection_top_k ou "
                "feature_selection_min_score registrado no manifest."
            )
        return self

    @model_validator(mode="after")
    def _undersampling_after_never_exceeds_before(self) -> "SelectionManifest":
        if self.fitted_rows_after_undersampling > self.fitted_rows_before_undersampling:
            raise ValueError(
                "fitted_rows_after_undersampling não pode ser maior que "
                "fitted_rows_before_undersampling."
            )
        if self.undersampling == "none" and (
            self.fitted_rows_after_undersampling != self.fitted_rows_before_undersampling
        ):
            raise ValueError(
                "undersampling='none' não pode alterar o número de linhas de treino."
            )
        return self

    @model_validator(mode="after")
    def _class_counts_are_consistent_with_row_totals(self) -> "SelectionManifest":
        if self.class_counts_before and sum(self.class_counts_before.values()) != (
            self.fitted_rows_before_undersampling
        ):
            raise ValueError(
                "soma de class_counts_before deve bater com fitted_rows_before_undersampling."
            )
        if self.class_counts_after and sum(self.class_counts_after.values()) != (
            self.fitted_rows_after_undersampling
        ):
            raise ValueError(
                "soma de class_counts_after deve bater com fitted_rows_after_undersampling."
            )
        return self
