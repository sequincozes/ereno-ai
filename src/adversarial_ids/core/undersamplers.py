"""Undersamplers — registro por chave das estratégias de reamostragem (E7).

Só se aplica à partição de TREINO, depois de ``FeaturePreprocessor.transform``
e (se ativa) ``FeatureSelector.transform`` — nunca ao teste/variante:
reamostrar o conjunto de avaliação distorceria a própria métrica que se quer
medir (a distribuição real de ataque vs. normal é o que o detector precisa
enfrentar). Esse é o invariante mais importante deste módulo e ele é
responsabilidade do chamador: o que está aqui não sabe qual partição recebeu.

Terceiro registro do mesmo formato (``core/detectors.py`` no E8,
``core/generators.py`` a montante): um ``Literal`` no domínio, um
``UndersamplerSpec`` por chave aqui, e uma checagem de sincronia na importação.
Quem escolhe a chave é ``core/undersampling_policy.py`` — a configuração ou a
política declarada.

Estratégias registradas
-----------------------
- ``none`` (default): devolve ``X``/``y`` inalterados, sem copiar.
- ``random``: sub-amostra cada classe acima da minoritária até igualá-la.
- ``ratio``: sub-amostra cada classe até ``target_ratio`` vezes a minoritária,
  mantendo um desequilíbrio declarado em vez de eliminá-lo.

Nenhuma delas usa ``imbalanced-learn``: reamostragem sem reposição e sem
substituto sintético é simples o bastante para ``numpy``/``pandas`` puros, e a
dependência a mais não se pagaria. Um undersampler que precise de vizinhança
(NearMiss, Tomek) entra como uma chave a mais, e aí a dependência é dele.

Honestidade sobre o estado atual
--------------------------------
``random`` e ``ratio`` são a **mesma** classe, com ``target_ratio`` 1.0 e N —
balancear até a minoritária é o caso particular de reduzir até uma razão. Isto
é descrição, não arquitetura aspiracional: a costura que o registro abre vale
para a estratégia que não for um teto por classe. Mesma ressalva que
``core/generators.py`` faz sobre os dois backends dele.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from adversarial_ids.domain.undersampling import (
    UNDERSAMPLER_KEYS,
    undersampler_requires_target_ratio,
)

DEFAULT_UNDERSAMPLER_KEY = "none"


class UndersamplerError(ValueError):
    """Erro acionável: o undersampling não pôde ser aplicado."""


@runtime_checkable
class UndersamplerLike(Protocol):
    """A superfície que o pipeline consome de um undersampler.

    Um método só, e deliberadamente sem ``transform``: diferente do E6/E7
    (preprocessador e seletor), não há estado ajustado a reaplicar depois — a
    reamostragem acontece uma vez, na partição de treino, e nada do que ela
    decidiu pode ser reaplicado ao teste sem quebrar o invariante do módulo.
    """

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]: ...


@dataclass(frozen=True)
class UndersamplerSpec:
    """Descrição estática de uma estratégia registrada."""

    key: str
    changes_rows: bool
    requires_target_ratio: bool
    description: str
    build: Callable[..., UndersamplerLike]


def class_counts(y: pd.Series) -> dict[str, int]:
    """Contagem por rótulo, chaves como ``str`` — usada pelo manifest (E7)."""

    return {
        str(label): int(count) for label, count in pd.Series(y).value_counts().items()
    }


def _validate_pair(X: pd.DataFrame, y: pd.Series) -> None:
    if len(X) != len(y):
        raise UndersamplerError(
            f"X e y têm tamanhos diferentes ({len(X)} != {len(y)})."
        )


class NoUndersampler:
    """Identidade: devolve exatamente os objetos recebidos.

    Devolver ``X``/``y`` sem copiar é intencional — ``undersampling='none'``
    precisa ser indistinguível de não haver undersampling nenhum, inclusive em
    identidade de objeto, para que a ablação do E7 compare o pipeline com e sem
    a etapa, e não com e sem uma cópia.
    """

    key = "none"

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        _validate_pair(X, y)
        return X, y


class RandomUndersampler:
    """Reamostragem aleatória sem reposição, determinística por ``random_state``.

    Cada classe acima de ``target_ratio × minoritária`` é sub-amostrada até
    esse teto; as demais passam inteiras. A ordem relativa das linhas mantidas
    é preservada (índices reamostrados são ordenados antes de recortar), então
    o resultado não depende da ordem de iteração das classes.

    ``target_ratio=1.0`` (default) é o balanceamento total: o teto vira o
    tamanho da própria classe minoritária.
    """

    def __init__(self, *, random_state: int = 42, target_ratio: float = 1.0) -> None:
        if target_ratio < 1.0:
            raise UndersamplerError(
                f"target_ratio={target_ratio!r} é menor que 1.0, o que pediria "
                "uma classe majoritária menor que a minoritária — isso é "
                "inverter o desequilíbrio, não corrigi-lo."
            )
        self.random_state = random_state
        self.target_ratio = target_ratio

    @property
    def key(self) -> str:
        return "random" if self.target_ratio == 1.0 else "ratio"

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        _validate_pair(X, y)

        if X.empty:
            raise UndersamplerError(
                "Não é possível reamostrar um conjunto de treino vazio (0 linhas)."
            )

        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        counts = y.value_counts()
        minority_size = int(counts.min())
        # Teto por classe. ``max(1, ...)`` protege o caso de uma minoritária de
        # uma linha com razão fracionária — o piso nunca pode zerar uma classe,
        # ou o modelo deixaria de ver um rótulo que existe no treino.
        cap = max(1, int(minority_size * self.target_ratio))
        rng = np.random.RandomState(self.random_state)

        # Ordenado por rótulo (como str) para que o consumo do gerador
        # aleatório siga sempre a mesma sequência de classes, independente da
        # ordem de aparição em y — mesma seed, mesma reamostragem.
        selected_indices: list[int] = []
        for label in sorted(counts.index, key=str):
            label_indices = y.index[y == label].to_numpy()
            if len(label_indices) > cap:
                chosen = rng.choice(label_indices, size=cap, replace=False)
            else:
                chosen = label_indices
            selected_indices.extend(int(index) for index in chosen)

        selected_indices.sort()

        return (
            X.iloc[selected_indices].reset_index(drop=True),
            y.iloc[selected_indices].reset_index(drop=True),
        )


def _build_none(*, random_state: int, target_ratio: float | None) -> UndersamplerLike:
    return NoUndersampler()


def _build_random(*, random_state: int, target_ratio: float | None) -> UndersamplerLike:
    return RandomUndersampler(random_state=random_state, target_ratio=1.0)


def _build_ratio(*, random_state: int, target_ratio: float | None) -> UndersamplerLike:
    assert target_ratio is not None  # garantido por build_undersampler
    return RandomUndersampler(random_state=random_state, target_ratio=target_ratio)


_REGISTRY: dict[str, UndersamplerSpec] = {
    "none": UndersamplerSpec(
        key="none",
        changes_rows=False,
        requires_target_ratio=False,
        description=(
            "Nenhuma reamostragem — o treino mantém a distribuição real de "
            "classes. Default, e o braço de controle de qualquer ablação."
        ),
        build=_build_none,
    ),
    "random": UndersamplerSpec(
        key="random",
        changes_rows=True,
        requires_target_ratio=False,
        description=(
            "Sub-amostragem aleatória sem reposição até o tamanho da classe "
            "minoritária. Equilibra, ao custo de descartar a maior parte do "
            "tráfego benigno quando a prevalência de ataque é baixa."
        ),
        build=_build_random,
    ),
    "ratio": UndersamplerSpec(
        key="ratio",
        changes_rows=True,
        requires_target_ratio=True,
        description=(
            "Sub-amostragem até target_ratio vezes a classe minoritária: "
            "limita o desequilíbrio sem eliminá-lo, preservando volume "
            "benigno que o balanceamento total jogaria fora."
        ),
        build=_build_ratio,
    ),
}

# Mesmo contrato de sincronia do E8 e do registro de geradores: uma chave
# registrada aqui e ausente do ``Literal`` produziria um ``SelectionManifest``
# impossível de validar, e uma chave só no ``Literal`` seria uma opção de
# configuração que não constrói nada.
if tuple(_REGISTRY) != UNDERSAMPLER_KEYS:
    raise UndersamplerError(
        "Registro de undersamplers fora de sincronia com "
        f"domain.undersampling.UndersamplerKey: registro={tuple(_REGISTRY)!r}, "
        f"contrato={UNDERSAMPLER_KEYS!r}."
    )

# As duas tabelas de capacidade do domínio descrevem as mesmas estratégias que
# este registro constrói. Divergência aqui é um manifest que mente sobre o que
# a etapa pôde fazer, então também falha na importação.
for _key, _spec in _REGISTRY.items():
    if _spec.requires_target_ratio != undersampler_requires_target_ratio(_key):
        raise UndersamplerError(
            f"Undersampler {_key!r}: registro e domínio discordam sobre "
            "requires_target_ratio."
        )


def undersampler_spec(key: str) -> UndersamplerSpec:
    """``UndersamplerSpec`` registrado sob ``key``, ou ``UndersamplerError``."""

    try:
        return _REGISTRY[key]
    except KeyError:
        raise UndersamplerError(
            f"Undersampler desconhecido: {key!r}. "
            f"Registrados: {list(UNDERSAMPLER_KEYS)!r}."
        ) from None


def list_undersampler_keys() -> tuple[str, ...]:
    """Chaves registradas, na ordem do registro (para a ajuda da CLI)."""

    return tuple(_REGISTRY)


def build_undersampler(
    key: str,
    *,
    random_state: int = 42,
    target_ratio: float | None = None,
) -> UndersamplerLike:
    """Constrói a estratégia ``key``.

    Recusa ``target_ratio`` ausente onde ele é obrigatório, e também presente
    onde não se aplica — informá-lo para ``random`` descreveria um corte que a
    estratégia não faz, e é exatamente o que o ``UndersamplingDecision``
    recusaria depois, no artefato.
    """

    spec = undersampler_spec(key)
    if spec.requires_target_ratio and target_ratio is None:
        raise UndersamplerError(
            f"Undersampler {key!r} exige target_ratio, que não foi informado."
        )
    if not spec.requires_target_ratio and target_ratio is not None:
        raise UndersamplerError(
            f"Undersampler {key!r} não usa target_ratio ({target_ratio!r})."
        )
    return spec.build(random_state=random_state, target_ratio=target_ratio)
