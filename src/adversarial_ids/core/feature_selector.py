"""FeatureSelector — seleção de features, filter-based e wrapper-based (E7).

Consome ``FeaturePreprocessor.feature_columns`` (E6) como ponto de partida —
opera sobre o dataframe já transformado (numérico, sem categorias em texto),
nunca sobre o CSV cru. Mesmo protocolo fit/transform do E6, mesma disciplina
anti-leakage: ``fit`` só pode ver a partição de treino, ``transform`` nunca
recalcula nada.

Três estratégias, e a diferença entre as duas últimas é a da literatura:

- ``"none"`` (default) mantém todas as features candidatas — comportamento
  idêntico a antes do E7 existir.
- ``"mutual_info"`` é **filter-based**: pontua cada feature contra o rótulo por
  informação mútua, sem treinar modelo nenhum, e corta por ``top_k`` e/ou
  ``min_score`` (pelo menos um precisa vir — sem critério de corte não há
  seleção). Pontuação **maior é melhor**.
- ``"rfe"`` é **wrapper-based**: treina um modelo, descarta a feature menos
  importante, retreina, e repete até sobrarem ``top_k``. Pontuação é o
  *ranking de eliminação* — **menor é melhor**, ``1`` para as mantidas. Exige
  ``top_k`` (o critério de parada do RFE é um número de features, não um
  limiar) e recusa ``min_score``, que não tem significado aqui.

Por que o RFE **não** usa o detector que vai treinar
----------------------------------------------------
Seria o wrapper "de manual" — e quebraria a garantia do E8. RF/DT/SVM/XGBoost
são comparáveis porque o pipeline a montante deles é idêntico; se a seleção
usasse cada detector como seu próprio avaliador, cada um treinaria sobre um
espaço de features diferente e a comparação deixaria de medir o detector para
medir a seleção. O estimador do RFE é fixo, barato e independente do detector
(uma árvore rasa), e isso é uma decisão de desenho, não uma limitação.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

# Estimador de ranking do RFE. Raso de propósito: ele é chamado uma vez por
# feature eliminada, então o custo do wrapper é ``n_features`` treinos — o que
# o torna viável é o modelo ser barato, não o conjunto ser pequeno.
_RFE_MAX_DEPTH = 8


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
        strategy: Literal["none", "mutual_info", "rfe"] = "none",
        top_k: int | None = None,
        min_score: float | None = None,
        random_state: int = 42,
    ) -> None:
        if strategy == "mutual_info" and top_k is None and min_score is None:
            raise FeatureSelectorError(
                "strategy='mutual_info' exige top_k e/ou min_score — sem critério de "
                "corte não há como decidir quantas features manter."
            )
        if strategy == "rfe":
            if top_k is None:
                raise FeatureSelectorError(
                    "strategy='rfe' exige top_k: o critério de parada do RFE é um "
                    "número de features a manter, não um limiar de pontuação."
                )
            if min_score is not None:
                raise FeatureSelectorError(
                    "strategy='rfe' não aceita min_score — a pontuação do RFE é um "
                    "ranking de eliminação (1 = mantida), não uma força de "
                    "associação com o rótulo, então um limiar não significa nada."
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

        if self.strategy == "rfe":
            self._fit_rfe(X, y)
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

    def _fit_rfe(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Elimina recursivamente a feature menos importante até sobrarem top_k.

        ``top_k`` acima do número de candidatas não é erro: o RFE do sklearn
        recusaria, mas "quero as 20 melhores de 12" é um pedido satisfeito por
        manter as 12 — recusar faria uma configuração razoável derrubar a
        execução por um detalhe de contagem.
        """

        assert self.top_k is not None  # garantido no __init__
        keep = min(self.top_k, len(self._candidate_features))

        selector = RFE(
            estimator=DecisionTreeClassifier(
                max_depth=_RFE_MAX_DEPTH, random_state=self.random_state
            ),
            n_features_to_select=keep,
            step=1,
        )
        selector.fit(X, y)

        # ``ranking_`` é 1 para as mantidas e 2..n na ordem inversa de
        # eliminação. Guardado como float para caber no mesmo campo do
        # manifest que a informação mútua usa — com a ressalva, registrada na
        # doc e no contrato, de que aqui MENOR é melhor.
        self._scores = {
            column: float(rank)
            for column, rank in zip(self._candidate_features, selector.ranking_)
        }
        self._selected_features = [
            column
            for column, keep_it in zip(self._candidate_features, selector.support_)
            if keep_it
        ]
        if not self._selected_features:
            raise FeatureSelectorError(
                "Nenhuma feature sobreviveu ao RFE — o conjunto de candidatas "
                "estava vazio ou o estimador não produziu ranking."
            )

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
