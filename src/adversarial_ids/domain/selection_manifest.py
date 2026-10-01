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
3. Só depois o undersampler resolvido em ``core/undersamplers.py`` reamostra
   as linhas de treino (já no espaço de features selecionado) antes de
   ``model.fit`` — nunca a partição de teste/variante, que precisa continuar
   refletindo a distribuição real para a métrica fazer sentido.

Quem escolheu o undersampler também é artefato
----------------------------------------------
``undersampling`` guarda a estratégia **resolvida** e
``undersampling_decided_by`` guarda se ela veio da configuração ou da política
declarada (``core/undersampling_policy.py``), com a razão e o desequilíbrio
observado ao lado. Sem esses três campos, um run com a política ligada é
indistinguível de um run configurado à mão — e a diferença entre "alguém
escolheu" e "uma regra escolheu, assim" é metade do que torna o resultado
auditável.

Os campos novos são **aditivos e com default**, então ``schema_version``
continua ``1``: um manifest gravado antes deles é lido com
``decided_by="config"``, que é exatamente o que ele era. Bump de schema serve
para mudança que invalida o artefato antigo (como o rename de estágios do
``LoopRecord`` v2), não para campo acrescentado.

``fitted_rows_before_undersampling``/``fitted_rows_after_undersampling`` são
a mesma prova de anti-leakage que ``FeatureManifest.fitted_rows``: contam
linhas de TREINO, nunca do dataset inteiro nem da partição de teste.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.undersampling import (
    UndersamplerKey,
    UndersamplingDecider,
    undersampler_requires_target_ratio,
)

# Vocabulário de seleção, exportado para quem precisa oferecer as opções
# (CLI, scripts) sem redigitar a lista — uma lista digitada duas vezes foi
# exatamente como o "rfe" ficou inalcançável em scripts/compare_detectors.py.
FeatureSelectionStrategy = Literal["none", "mutual_info", "rfe"]

FEATURE_SELECTION_STRATEGIES: tuple[str, ...] = get_args(FeatureSelectionStrategy)


class SelectionManifest(BaseModel):
    """Estado ajustado de ``FeatureSelector`` + o undersampler resolvido, pronto para persistir."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    # -- Seleção de features ------------------------------------------------ #
    feature_selection: FeatureSelectionStrategy = "none"
    feature_selection_top_k: int | None = Field(default=None, gt=0)
    feature_selection_min_score: float | None = None
    candidate_features: tuple[str, ...] = Field(min_length=1)
    selected_features: tuple[str, ...] = Field(min_length=1)
    # Vazio só quando feature_selection="none"; as outras duas pontuam todas
    # as candidatas. **A escala depende da estratégia**, e é o próprio
    # ``feature_selection`` que diz qual: em "mutual_info" é informação mútua
    # com o rótulo e MAIOR é melhor; em "rfe" é o ranking de eliminação, com
    # 1 para as mantidas, e MENOR é melhor. Ordenar este dicionário sem olhar
    # a estratégia produz o ranking invertido em um dos dois casos.
    feature_scores: dict[str, float] = Field(default_factory=dict)

    # -- Undersampling da partição de treino -------------------------------- #
    # A estratégia **resolvida**, nunca o modo configurado: quando a política
    # decide (``undersampling_decided_by="policy"``), o que rodou foi uma
    # destas chaves, e é ela que precisa estar no artefato — "policy" não é um
    # undersampler, é quem escolheu um.
    undersampling: UndersamplerKey = "none"
    undersampling_decided_by: UndersamplingDecider = "config"
    undersampling_target_ratio: float | None = Field(default=None, ge=1.0)
    # Desequilíbrio (maioria/minoritária) medido no treino antes de reamostrar,
    # e a razão da escolha por escrito. Registrados mesmo quando a decisão veio
    # da configuração: é o que permite julgar depois se a escolha fixa cabia
    # naquele dataset.
    undersampling_observed_imbalance: float | None = Field(default=None, ge=1.0)
    undersampling_decision_reason: str = ""
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
                    "feature_scores fica vazio quando feature_selection='none' — "
                    "nenhuma pontuação é calculada."
                )
        elif set(self.feature_scores) != set(self.candidate_features):
            raise ValueError(
                "feature_scores deve cobrir exatamente candidate_features quando "
                f"feature_selection={self.feature_selection!r} (faltam "
                f"{sorted(set(self.candidate_features) - set(self.feature_scores))!r}, sobram "
                f"{sorted(set(self.feature_scores) - set(self.candidate_features))!r})."
            )
        return self

    @model_validator(mode="after")
    def _the_strategy_requires_its_own_stopping_criterion(self) -> "SelectionManifest":
        """Cada estratégia para por um motivo diferente, e o manifest precisa
        registrar qual foi — senão não dá para reproduzir o corte."""

        if self.feature_selection == "mutual_info" and (
            self.feature_selection_top_k is None and self.feature_selection_min_score is None
        ):
            raise ValueError(
                "feature_selection='mutual_info' exige feature_selection_top_k ou "
                "feature_selection_min_score registrado no manifest."
            )
        if self.feature_selection == "rfe":
            if self.feature_selection_top_k is None:
                raise ValueError(
                    "feature_selection='rfe' exige feature_selection_top_k: o RFE "
                    "para por número de features, não por limiar."
                )
            if self.feature_selection_min_score is not None:
                raise ValueError(
                    "feature_selection='rfe' não usa feature_selection_min_score — "
                    "a pontuação é um ranking de eliminação, e um limiar sobre ela "
                    "descreveria um corte que não aconteceu."
                )
        return self

    @model_validator(mode="after")
    def _the_undersampler_records_its_own_cut(self) -> "SelectionManifest":
        """Mesma disciplina do critério de parada da seleção: cada estratégia
        corta por um motivo, e o manifest registra qual — senão o corte não se
        reproduz a partir do artefato."""

        needs_ratio = undersampler_requires_target_ratio(self.undersampling)
        if needs_ratio and self.undersampling_target_ratio is None:
            raise ValueError(
                f"undersampling={self.undersampling!r} exige "
                "undersampling_target_ratio: é o teto por classe que define o "
                "corte aplicado."
            )
        if not needs_ratio and self.undersampling_target_ratio is not None:
            raise ValueError(
                f"undersampling={self.undersampling!r} não usa "
                f"undersampling_target_ratio ({self.undersampling_target_ratio!r}) "
                "— registrá-lo descreveria um corte que não aconteceu."
            )
        return self

    @model_validator(mode="after")
    def _a_decision_by_policy_states_its_reason(self) -> "SelectionManifest":
        """Uma política que decide sem dizer por quê é indistinguível de um
        default — e é justamente a distinção que o artefato existe para
        preservar."""

        if self.undersampling_decided_by == "policy":
            if not self.undersampling_decision_reason.strip():
                raise ValueError(
                    "undersampling_decided_by='policy' exige "
                    "undersampling_decision_reason: a razão da escolha é o que "
                    "torna a decisão auditável."
                )
            if self.undersampling_observed_imbalance is None:
                raise ValueError(
                    "undersampling_decided_by='policy' exige "
                    "undersampling_observed_imbalance: é a medida sobre a qual "
                    "a política decidiu."
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
