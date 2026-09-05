"""FeatureSelector — seleção de features por mutual information (épico E7).

Consome ``FeaturePreprocessor.feature_columns`` (E6) como ponto de partida —
opera sobre o dataframe já transformado (numérico, sem categorias em texto),
nunca sobre o CSV cru. Mesmo protocolo fit/transform do E6, mesma disciplina
anti-leakage: ``fit`` só pode ver a partição de treino, ``transform`` nunca
recalcula nada.

``strategy="none"`` é o default — mantém todas as features candidatas, ou
seja, comportamento idêntico a antes do E7 existir. ``strategy="mutual_info"``
ajusta ``sklearn.feature_selection.mutual_info_classif`` sobre X_treino/y_treino
e mantém só as melhores, por ``top_k`` e/ou ``min_score`` (pelo menos um dos
dois precisa ser informado — sem critério de corte não há seleção).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class FeatureSelectorError(ValueError):
    """Erro acionável: a seleção de features não pôde ser ajustada."""


class FeatureSelector:
    """Fit/transform determinístico de seleção de features sobre X já preprocessado.

    ``fit`` decide, sobre o dataframe recebido, quais colunas sobrevivem;
    ``transform`` só recorta para essas colunas (nunca recalcula pontuação).
    A ordem de ``selected_features`` preserva a ordem original de
    ``candidate_features`` (a ordem que o preprocessador produziu), não a
    ordem de ranking — mantém o layout estável para o modelo treinado.
    """

    def __init__(
        self,
        *,
        strategy: Literal["none", "mutual_info"] = "none",
        top_k: int | None = None,
        min_score: float | None = None,
        random_state: int = 42,
    ) -> None:
        if strategy == "mutual_info" and top_k is None and min_score is None:
            raise FeatureSelectorError(
                "strategy='mutual_info' exige top_k e/ou min_score — sem critério de "
                "corte não há como decidir quantas features manter."
            )
        if top_k is not None and top_k < 1:
            raise FeatureSelectorError(f"top_k deve ser >= 1 (recebido {top_k!r}).")

        self.strategy = strategy
        self.top_k = top_k
        self.min_score = min_score
        self.random_state = random_state

        self._fitted = False
        self._candidate_features: list[str] = []
        self._selected_features: list[str] = []
        self._scores: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Ajusta sobre ``X``/``y`` — deve ser só a partição de treino (já transformada)."""

        if X.empty:
            raise FeatureSelectorError(
                "Não é possível ajustar seleção de features sobre um dataframe vazio (0 linhas)."
            )
        if len(X) != len(y):
            raise FeatureSelectorError(
                f"X e y têm tamanhos diferentes ({len(X)} != {len(y)})."
            )

        self._candidate_features = list(X.columns)

        if self.strategy == "none":
            self._selected_features = list(self._candidate_features)
            self._scores = {}
            self._fitted = True
            return self

        raw_scores = mutual_info_classif(X, y, random_state=self.random_state)
        self._scores = {
            column: float(score) for column, score in zip(self._candidate_features, raw_scores)
        }

        # Desempate determinístico por nome de coluna — mutual_info_classif
        # tem seu próprio random_state para o estimador de vizinhos, mas um
        # empate exato de pontuação (comum em colunas quase idênticas) não
        # pode depender da ordem de iteração de um dict.
        ranked = sorted(self._scores.items(), key=lambda item: (-item[1], item[0]))

        if self.top_k is not None:
            ranked = ranked[: self.top_k]
        if self.min_score is not None:
            ranked = [item for item in ranked if item[1] >= self.min_score]

        if not ranked:
            raise FeatureSelectorError(
                "Nenhuma feature sobreviveu ao critério de seleção "
                f"(top_k={self.top_k!r}, min_score={self.min_score!r})."
            )

        selected = {name for name, _ in ranked}
        self._selected_features = [
            column for column in self._candidate_features if column in selected
        ]
        self._fitted = True

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Recorta ``X`` para as features que ``fit`` selecionou."""

        if not self._fitted:
            raise FeatureSelectorError(
                "transform() chamado antes de fit(): nenhuma seleção ajustada."
            )
        return X[self._selected_features].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def candidate_features(self) -> list[str]:
        return list(self._candidate_features)

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected_features)

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._scores)
