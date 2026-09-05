"""Detector — interface única para RF/DT/SVM sob o mesmo protocolo (épico E8).

Fecha a etapa D36-46 ("mesmo split/protocolo para todos os detectores") que o
E6 e o E7 vinham anunciando: o preparo de features
(``core/preprocessor.py``), a seleção e o undersampling
(``core/feature_selector.py``/``core/undersampler.py``) já eram
detector-agnósticos; o que faltava era o próprio detector deixar de ser um
``RandomForestClassifier`` embutido em ``core/ids_evaluator.py``.

O que este módulo **não** faz: nenhum preparo de dados, nenhum split, nenhuma
métrica. Recebe um ``X`` já preprocessado/selecionado/reamostrado e um ``y``
já codificado, e devolve predições — exatamente o mesmo contrato que o
``RandomForestClassifier`` cumpria antes, agora com quatro implementações
atrás dele. Toda a comparabilidade entre detectores vem de eles serem
alimentados pelo mesmo pipeline a montante, não de nada feito aqui.

Detectores registrados
----------------------
- ``random_forest`` (default): ``RandomForestClassifier``. Comportamento
  idêntico ao de antes do E8 — mesmos hiperparâmetros, mesma ordem de fit,
  mesmas importâncias Gini. O histórico golden continua byte-estável.
- ``decision_tree``: ``DecisionTreeClassifier``. Mesma família, uma árvore só
  — a linha de base interpretável contra a qual o ganho do ensemble se mede.
- ``svm_linear``: ``LinearSVC`` (liblinear). Escala para o volume do baseline
  do ERENO e expõe ``coef_``, então ainda produz um ranking de features.
- ``svm_rbf``: ``SVC(kernel="rbf")``. É o "SVM" da literatura de IDS, mas
  custa O(n²)-O(n³) no número de linhas de treino e **não** expõe importância
  de feature nenhuma (ver ``importance_kind``).

Por que quatro chaves e não uma chave ``svm`` com um parâmetro de kernel: o
kernel muda tanto o custo quanto a capacidade de explicação do detector, e
esses dois fatos precisam estar visíveis no nome do que se está rodando.
Uma chave ``svm`` que silenciosamente fosse linear (rápida, explicável) ou
RBF (lenta, opaca) faria dois experimentos incomparáveis parecerem o mesmo.

Escala
------
Os dois SVMs declaram ``requires_scaling=True``: ``IdsEvaluator`` traduz isso
em ``FeaturePreprocessor(scaler="standard")`` quando o chamador não força o
contrário. É a razão de ``scaler="standard"`` existir desde o E6 desligado
por default (ver ``docs/preprocessing.md``). As árvores são invariantes a
transformação monotônica de feature e continuam com ``scaler="none"`` — é o
que mantém o caminho default idêntico ao de antes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from adversarial_ids.domain.detector_manifest import DETECTOR_KEYS, DetectorManifest

DEFAULT_DETECTOR_KEY = "random_forest"


class DetectorError(ValueError):
    """Erro acionável: o detector não pôde ser construído, treinado ou consultado."""


@runtime_checkable
class DetectorLike(Protocol):
    """A superfície que ``IdsEvaluator`` de fato consome de um detector.

    Diferente de ``IntentLike``/``DefenderLike`` no orquestrador, este **não é
    (ainda) um ponto de injeção**: ``IdsEvaluator`` recebe uma *chave* e
    constrói o ``Detector`` concreto sozinho, então não há como passar outra
    implementação por fora hoje. O protocolo existe como contrato explícito —
    é a lista fechada do que o avaliador chama, e portanto a superfície que um
    detector fora do scikit-learn precisaria cumprir para entrar no registro
    sem tocar em ``ids_evaluator.py``.
    """

    key: str
    model_name: str
    requires_scaling: bool

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DetectorLike": ...

    def predict(self, X: pd.DataFrame) -> Any: ...

    def feature_importances(
        self, feature_columns: list[str] | None, top_n: int = 15
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DetectorSpec:
    """Descrição estática de um detector registrado.

    ``defaults`` lista **todos** os hiperparâmetros que o chamador pode
    sobrescrever — uma chave fora dessa lista é rejeitada, mesma disciplina do
    ``extra="forbid"`` dos contratos do ``domain/``. Os valores são os defaults
    do próprio scikit-learn, salvo onde o comentário do registro diz o
    contrário; passá-los explicitamente é equivalente a omiti-los, o que é o
    que mantém o caminho ``random_forest`` byte-idêntico ao de antes do E8.
    """

    key: str
    requires_scaling: bool
    importance_kind: str
    supports_shap: bool
    build: Callable[..., Any]
    defaults: Mapping[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, DetectorSpec] = {
    "random_forest": DetectorSpec(
        key="random_forest",
        requires_scaling=False,
        importance_kind="gini",
        supports_shap=True,
        build=RandomForestClassifier,
        # Exatamente o que ``IdsEvaluator`` construía embutido antes do E8
        # (n_estimators=100, n_jobs=-1) mais dois knobs em valor default do
        # sklearn — nada aqui muda o modelo resultante.
        defaults={"n_estimators": 100, "n_jobs": -1, "max_depth": None, "class_weight": None},
    ),
    "decision_tree": DetectorSpec(
        key="decision_tree",
        requires_scaling=False,
        importance_kind="gini",
        supports_shap=True,
        build=DecisionTreeClassifier,
        defaults={
            "criterion": "gini",
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
    ),
    "svm_linear": DetectorSpec(
        key="svm_linear",
        requires_scaling=True,
        importance_kind="linear_coef",
        supports_shap=False,
        build=LinearSVC,
        # max_iter=5000 em vez dos 1000 do sklearn: com dezenas de milhares de
        # linhas o liblinear frequentemente não converge no default e emite
        # ConvergenceWarning — um modelo mal-convergido produziria uma métrica
        # que mede o otimizador, não o detector.
        defaults={"C": 1.0, "class_weight": None, "max_iter": 5000, "dual": "auto"},
    ),
    "svm_rbf": DetectorSpec(
        key="svm_rbf",
        requires_scaling=True,
        importance_kind="none",
        supports_shap=False,
        build=SVC,
        # cache_size acima do default (200 MB) porque o gargalo do libsvm
        # nesse volume é o kernel cache; não muda o modelo, só o tempo.
        defaults={"C": 1.0, "gamma": "scale", "class_weight": None, "cache_size": 500.0},
    ),
}

# O registro concreto e o vocabulário do domínio precisam descrever o mesmo
# conjunto de detectores: uma chave registrada aqui e ausente do ``Literal``
# produziria um ``DetectorManifest`` impossível de validar, e uma chave só no
# ``Literal`` seria uma opção de CLI que não constrói nada. Falhar na
# importação é mais barato do que descobrir isso no meio de uma campanha.
if tuple(_REGISTRY) != DETECTOR_KEYS:
    raise DetectorError(
        "Registro de detectores fora de sincronia com domain.detector_manifest."
        f"DetectorKey: registro={tuple(_REGISTRY)!r}, contrato={DETECTOR_KEYS!r}."
    )


def detector_spec(key: str) -> DetectorSpec:
    """``DetectorSpec`` registrado sob ``key``, ou ``DetectorError``."""

    try:
        return _REGISTRY[key]
    except KeyError:
        raise DetectorError(
            f"Detector desconhecido: {key!r}. Registrados: {list(DETECTOR_KEYS)!r}."
        ) from None


def recommended_scaler(key: str) -> str:
    """Escala que ``key`` pede do ``FeaturePreprocessor`` (E6).

    ``"standard"`` para os SVMs (sensíveis a escala), ``"none"`` para as
    árvores. É só uma recomendação: o chamador pode forçar o contrário, e o
    ``DetectorManifest`` registra os dois valores lado a lado.
    """

    return "standard" if detector_spec(key).requires_scaling else "none"


class Detector:
    """Adaptador fit/predict sobre um estimador do scikit-learn registrado.

    Espelha o protocolo que o ``IdsEvaluator`` já usava com o Random Forest
    (``fit``/``predict``/importâncias), acrescentando só o que a comparação
    entre detectores exige: de que tipo é a importância que ele produz, se é
    explicável por ``shap.TreeExplainer``, e um ``manifest()`` persistível.

    ``fit`` recebe o ``X`` **já** preprocessado (E6), com features já
    selecionadas e linhas já reamostradas (E7) — este objeto não sabe nada
    sobre split, escala ou balanceamento, e não deve saber.
    """

    def __init__(
        self,
        key: str = DEFAULT_DETECTOR_KEY,
        *,
        random_state: int = 42,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = detector_spec(key)
        self.random_state = random_state

        overrides = dict(hyperparameters or {})
        unknown = set(overrides) - set(self.spec.defaults)
        if unknown:
            raise DetectorError(
                f"Hiperparâmetro(s) não suportado(s) por {key!r}: {sorted(unknown)!r}. "
                f"Aceitos: {sorted(self.spec.defaults)!r}."
            )
        self.hyperparameters: dict[str, Any] = {**self.spec.defaults, **overrides}

        self._estimator: Any = None
        self._fitted = False
        self._trained_rows = 0
        self._trained_features: list[str] = []
        self._train_duration_seconds = 0.0

    # ------------------------------------------------------------------ #
    # Identidade / capacidades                                            #
    # ------------------------------------------------------------------ #
    @property
    def key(self) -> str:
        return self.spec.key

    @property
    def model_name(self) -> str:
        """Nome que vai para ``DetectionReport.model_name``.

        Igual à chave de propósito: é o que amarra um relatório em disco ao
        ``detector_manifest.json`` da mesma execução sem nenhuma tradução no
        meio.
        """
        return self.spec.key

    @property
    def requires_scaling(self) -> bool:
        return self.spec.requires_scaling

    @property
    def importance_kind(self) -> str:
        return self.spec.importance_kind

    @property
    def supports_shap(self) -> bool:
        return self.spec.supports_shap

    @property
    def estimator(self) -> Any:
        """Estimador do sklearn — ``None`` enquanto ``fit`` não rodou."""
        return self._estimator

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------ #
    # fit / predict                                                       #
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Detector":
        if len(X) == 0:
            raise DetectorError(
                f"Não é possível treinar {self.key!r} sobre um conjunto vazio (0 linhas)."
            )
        if len(X) != len(y):
            raise DetectorError(f"X e y têm tamanhos diferentes ({len(X)} != {len(y)}).")

        self._estimator = self.spec.build(random_state=self.random_state, **self.hyperparameters)

        start = time.perf_counter()
        self._estimator.fit(X, y)
        self._train_duration_seconds = time.perf_counter() - start

        self._trained_rows = len(X)
        self._trained_features = list(X.columns)
        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> Any:
        if not self._fitted:
            raise DetectorError("predict() chamado antes de fit(): nenhum modelo treinado.")
        return self._estimator.predict(X)

    # ------------------------------------------------------------------ #
    # Importâncias — significado varia por detector                       #
    # ------------------------------------------------------------------ #
    def feature_importances(
        self, feature_columns: list[str] | None, top_n: int = 15
    ) -> list[dict[str, Any]]:
        """Ranking decrescente ``{"feature", "importance"}``, ou ``[]``.

        Devolve ``[]`` — nunca levanta — quando não há ranking a dar: modelo
        ainda não treinado, espaço de features desconhecido, ou detector sem
        importância nenhuma (``svm_rbf``). É a mesma degradação best-effort
        que ``IdsEvaluator.get_shap_importances`` já praticava: um detector
        opaco reduz a evidência que o Blue Team pode citar, não derruba o
        loop.

        A **escala não é comparável entre detectores**: Gini (RF/DT) soma 1
        sobre as features; ``|coef_|`` (SVM linear) está na unidade do espaço
        já padronizado. ``DetectorManifest.importance_kind`` é o campo que
        diz qual das duas está em jogo.
        """

        if not self._fitted or not feature_columns:
            return []

        values = self._raw_importances()
        if values is None or len(values) != len(feature_columns):
            return []

        ranking = sorted(
            (
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(feature_columns, values)
            ),
            key=lambda item: item["importance"],
            reverse=True,
        )

        return ranking[:top_n]

    def _raw_importances(self) -> Any | None:
        if self.importance_kind == "gini":
            return getattr(self._estimator, "feature_importances_", None)

        if self.importance_kind == "linear_coef":
            coef = getattr(self._estimator, "coef_", None)
            if coef is None:
                return None
            coef = np.asarray(coef)
            # Binário: (1, n_features). Multiclasse: (n_classes, n_features) —
            # a média do módulo por coluna resume o peso da feature na
            # fronteira sem privilegiar arbitrariamente uma das classes.
            return np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)

        return None

    # ------------------------------------------------------------------ #
    # Manifest                                                            #
    # ------------------------------------------------------------------ #
    def manifest(self, *, resolved_scaler: str) -> DetectorManifest:
        """Empacota o detector treinado + o preparo sob o qual ele treinou.

        ``resolved_scaler`` vem do chamador (``IdsEvaluator``) porque é ele —
        não o detector — que decide o que o ``FeaturePreprocessor`` aplicou;
        registrar os dois lado a lado é o que deixa visível um SVM que rodou
        sem escala. **Obrigatório de propósito**: um default ``"none"`` faria
        um chamador distraído gravar "sem escala" num run padronizado — a
        exata desinformação que este campo existe para impedir.
        """

        if not self._fitted:
            raise DetectorError("manifest() chamado antes de fit(): nenhum modelo treinado.")

        return DetectorManifest(
            detector=self.key,  # type: ignore[arg-type]  # validado por detector_spec()
            model_name=self.model_name,
            hyperparameters={key: _jsonable(value) for key, value in self.hyperparameters.items()},
            random_state=self.random_state,
            requires_scaling=self.requires_scaling,
            resolved_scaler=resolved_scaler,  # type: ignore[arg-type]  # validado pelo contrato
            importance_kind=self.importance_kind,  # type: ignore[arg-type]
            supports_shap=self.supports_shap,
            trained_rows=self._trained_rows,
            trained_features=tuple(self._trained_features),
            train_duration_seconds=self._train_duration_seconds,
        )


def _jsonable(value: Any) -> float | int | str | bool | None:
    """Reduz um hiperparâmetro ao que ``DetectorManifest`` aceita persistir.

    Os registros só usam escalares hoje; o ``str()`` é a rede de segurança
    para um override exótico (um callable de kernel, um array de pesos) não
    quebrar a serialização do manifest inteiro.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
