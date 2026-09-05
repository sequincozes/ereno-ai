"""RandomUndersampler — balanceamento de classes da partição de treino (épico E7).

Só se aplica à partição de TREINO, depois de ``FeaturePreprocessor.transform``
e (se ativa) ``FeatureSelector.transform`` — nunca ao teste/variante: reamostrar
o conjunto de avaliação distorceria a própria métrica que se quer medir (a
distribuição real de ataque vs. normal é o que o detector precisa enfrentar).

``strategy="none"`` é o default — devolve ``X``/``y`` inalterados. Não usa a
dependência ``imbalanced-learn`` (fora do pyproject): a estratégia "random" é
simples o bastante — reamostragem sem reposição, sem substituto sintético —
para implementar com ``numpy``/``pandas`` puros, evitando puxar mais uma
dependência para um MVP que já tem 11 ataques + scikit-learn.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class UndersamplerError(ValueError):
    """Erro acionável: o undersampling não pôde ser aplicado."""


class RandomUndersampler:
    """Reamostragem aleatória sem reposição, determinística por ``random_state``.

    Para cada classe acima do tamanho da classe minoritária, sub-amostra até
    igualar a minoritária. A ordem relativa das linhas mantidas é preservada
    (índices reamostrados são ordenados antes de recortar) — o resultado não
    depende da ordem de iteração das classes.
    """

    def __init__(
        self,
        *,
        strategy: Literal["none", "random"] = "none",
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        if len(X) != len(y):
            raise UndersamplerError(f"X e y têm tamanhos diferentes ({len(X)} != {len(y)}).")

        if self.strategy == "none":
            return X, y

        if X.empty:
            raise UndersamplerError(
                "Não é possível reamostrar um conjunto de treino vazio (0 linhas)."
            )

        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        counts = y.value_counts()
        minority_size = int(counts.min())
        rng = np.random.RandomState(self.random_state)

        # Ordenado por rótulo (como str) para que o consumo do gerador
        # aleatório siga sempre a mesma sequência de classes, independente da
        # ordem de aparição em y — mesma seed, mesma reamostragem.
        selected_indices: list[int] = []
        for label in sorted(counts.index, key=str):
            label_indices = y.index[y == label].to_numpy()
            if len(label_indices) > minority_size:
                chosen = rng.choice(label_indices, size=minority_size, replace=False)
            else:
                chosen = label_indices
            selected_indices.extend(int(index) for index in chosen)

        selected_indices.sort()

        return (
            X.iloc[selected_indices].reset_index(drop=True),
            y.iloc[selected_indices].reset_index(drop=True),
        )

    @staticmethod
    def class_counts(y: pd.Series) -> dict[str, int]:
        """Contagem por rótulo, chaves como ``str`` — usada para o manifest (E7)."""

        return {str(label): int(count) for label, count in pd.Series(y).value_counts().items()}
