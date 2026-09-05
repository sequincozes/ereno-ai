"""Registro e adaptador de detectores (épico E8).

Cobre o critério de pronto do E8 ("RF/DT/SVM sob o mesmo split/protocolo") no
nível do componente: os quatro detectores registrados constroem, treinam e
predizem pela mesma interface; cada um declara honestamente a escala que pede
e o tipo de importância que produz; e o ``DetectorManifest`` resultante é
consistente com essas declarações. A integração com o pipeline de dados
(preprocessador E6 + seleção/undersampling E7) é coberta em
``tests/test_ids_evaluator_detectors.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversarial_ids.core.detectors import (
    DETECTOR_KEYS,
    DEFAULT_DETECTOR_KEY,
    Detector,
    DetectorError,
    DetectorLike,
    detector_spec,
    recommended_scaler,
)
from adversarial_ids.domain.detector_manifest import DetectorManifest


def _separable_frame(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """Duas classes linearmente separáveis por ``signal``, com uma coluna ruidosa.

    Separável de propósito: o objetivo aqui não é medir qualidade de detecção
    (isso é papel do ``IdsEvaluator``), é garantir que os quatro estimadores
    convergem e predizem — inclusive o ``LinearSVC``, que num problema não
    separável dependeria de ``max_iter`` para não virar teste instável.

    As colunas saem **padronizadas** (média 0, desvio 1) porque é isso que o
    pipeline real entrega a um SVM: ``IdsEvaluator`` resolve
    ``scaler="standard"`` para os dois SVMs antes do ``fit``. Sem isso o
    ``|coef_|`` não seria comparável entre colunas de amplitudes diferentes —
    a feature de maior amplitude ganharia o menor coeficiente e o ranking
    diria o contrário do que a fronteira de decisão realmente usa. As árvores
    são indiferentes à padronização, então os quatro detectores podem partir
    do mesmo quadro.
    """

    frame = pd.DataFrame(
        {
            "signal": [float(i % 5) for i in range(n)] + [100.0 + i % 5 for i in range(n)],
            "noise": [float((i * 3) % 7) for i in range(2 * n)],
        }
    )
    X = (frame - frame.mean()) / frame.std()
    y = pd.Series([0] * n + [1] * n)
    return X, y


# --------------------------------------------------------------------------- #
# Registro                                                                     #
# --------------------------------------------------------------------------- #
def test_default_detector_is_random_forest():
    assert DEFAULT_DETECTOR_KEY == "random_forest"
    assert DETECTOR_KEYS[0] == "random_forest"


def test_unknown_detector_key_is_rejected():
    with pytest.raises(DetectorError, match="Detector desconhecido"):
        detector_spec("xgboost")
    with pytest.raises(DetectorError, match="Detector desconhecido"):
        Detector("xgboost")


def test_only_svm_detectors_ask_for_standard_scaling():
    # As árvores são invariantes a transformação monotônica de feature; os SVMs
    # não. É esta tabela que faz o caminho default (RF) continuar sem escala,
    # idêntico ao de antes do E8.
    assert recommended_scaler("random_forest") == "none"
    assert recommended_scaler("decision_tree") == "none"
    assert recommended_scaler("svm_linear") == "standard"
    assert recommended_scaler("svm_rbf") == "standard"


def test_random_forest_defaults_reproduce_the_pre_e8_estimator():
    # A construção embutida que o IdsEvaluator fazia antes do E8 era
    # RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).
    # Os demais defaults do registro são os próprios defaults do sklearn, de
    # modo que passá-los explicitamente não muda o modelo resultante.
    detector = Detector("random_forest")
    assert detector.hyperparameters["n_estimators"] == 100
    assert detector.hyperparameters["n_jobs"] == -1
    assert detector.hyperparameters["max_depth"] is None
    assert detector.random_state == 42


def test_unsupported_hyperparameter_is_rejected_at_construction():
    with pytest.raises(DetectorError, match="não suportado"):
        Detector("decision_tree", hyperparameters={"n_estimators": 10})


def test_hyperparameter_override_wins_over_registry_default():
    detector = Detector("random_forest", hyperparameters={"n_estimators": 7})
    assert detector.hyperparameters["n_estimators"] == 7
    # ...sem apagar os demais defaults do registro.
    assert detector.hyperparameters["n_jobs"] == -1


def test_detector_satisfies_the_detector_like_protocol():
    assert isinstance(Detector("random_forest"), DetectorLike)


# --------------------------------------------------------------------------- #
# fit / predict — os quatro pela mesma interface                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_every_registered_detector_trains_and_predicts(key):
    X, y = _separable_frame()
    detector = Detector(key).fit(X, y)

    assert detector.is_fitted
    assert detector.estimator is not None
    assert len(detector.predict(X)) == len(X)


@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_predict_before_fit_raises(key):
    X, _ = _separable_frame()
    with pytest.raises(DetectorError, match="antes de fit"):
        Detector(key).predict(X)


def test_fit_rejects_empty_and_mismatched_input():
    X, y = _separable_frame()
    with pytest.raises(DetectorError, match="vazio"):
        Detector("decision_tree").fit(X.iloc[:0], y.iloc[:0])
    with pytest.raises(DetectorError, match="tamanhos diferentes"):
        Detector("decision_tree").fit(X, y.iloc[:-1])


@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_same_seed_gives_the_same_predictions(key):
    X, y = _separable_frame()
    first = Detector(key, random_state=42).fit(X, y).predict(X)
    second = Detector(key, random_state=42).fit(X, y).predict(X)
    assert list(first) == list(second)


# --------------------------------------------------------------------------- #
# Importâncias — significado diferente por detector, formato igual             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ("random_forest", "decision_tree", "svm_linear"))
def test_detectors_with_importances_rank_the_informative_feature_first(key):
    X, y = _separable_frame()
    detector = Detector(key).fit(X, y)

    ranking = detector.feature_importances(list(X.columns))
    assert [item["feature"] for item in ranking][0] == "signal"
    # Formato congelado da issue #15, igual para todos os detectores.
    assert set(ranking[0]) == {"feature", "importance"}
    assert all(isinstance(item["importance"], float) for item in ranking)


def test_svm_rbf_has_no_importances_and_says_so_instead_of_raising():
    X, y = _separable_frame()
    detector = Detector("svm_rbf").fit(X, y)

    assert detector.importance_kind == "none"
    assert detector.feature_importances(list(X.columns)) == []


@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_importances_before_fit_are_empty_not_an_error(key):
    assert Detector(key).feature_importances(["signal"]) == []


def test_top_n_truncates_the_ranking():
    X, y = _separable_frame()
    detector = Detector("random_forest").fit(X, y)
    assert len(detector.feature_importances(list(X.columns), top_n=1)) == 1


def test_importances_are_empty_when_the_feature_space_disagrees_with_the_model():
    # Guarda contra zipar silenciosamente um ranking desalinhado: melhor não
    # dar ranking nenhum do que atribuir a importância de uma coluna a outra.
    X, y = _separable_frame()
    detector = Detector("random_forest").fit(X, y)
    assert detector.feature_importances(["signal", "noise", "extra"]) == []


# --------------------------------------------------------------------------- #
# Manifest                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_manifest_describes_the_trained_detector(key):
    X, y = _separable_frame()
    detector = Detector(key).fit(X, y)

    manifest = detector.manifest(resolved_scaler=recommended_scaler(key))
    assert isinstance(manifest, DetectorManifest)
    assert manifest.detector == key
    assert manifest.model_name == key
    assert manifest.trained_rows == len(X)
    assert manifest.trained_features == tuple(X.columns)
    assert manifest.train_duration_seconds >= 0.0
    # O manifest tem que sobreviver a um round-trip JSON: é persistido como
    # detector_manifest.json no estágio DETECTOR do loop intent-driven.
    assert DetectorManifest.model_validate(manifest.model_dump(mode="json")) == manifest


@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_manifest_before_fit_raises(key):
    with pytest.raises(DetectorError, match="antes de fit"):
        Detector(key).manifest(resolved_scaler=recommended_scaler(key))


def test_manifest_records_a_scaler_that_disagrees_with_the_recommendation():
    # Rodar um SVM sem escala é legítimo (ablação controlada) e por isso não é
    # erro — mas precisa ficar visível no artefato, e não some do manifest.
    X, y = _separable_frame()
    manifest = Detector("svm_linear").fit(X, y).manifest(resolved_scaler="none")

    assert manifest.requires_scaling is True
    assert manifest.resolved_scaler == "none"


def test_manifest_hyperparameters_survive_serialization():
    X, y = _separable_frame()
    manifest = (
        Detector("random_forest", hyperparameters={"n_estimators": 5})
        .fit(X, y)
        .manifest(resolved_scaler="none")
    )

    assert manifest.hyperparameters["n_estimators"] == 5
    assert manifest.hyperparameters["max_depth"] is None


def test_resolved_scaler_is_required_so_a_scaled_run_cannot_be_recorded_as_unscaled():
    # Um default "none" faria um chamador distraído gravar "sem escala" num run
    # padronizado — a exata desinformação que o campo existe para impedir.
    X, y = _separable_frame()
    with pytest.raises(TypeError, match="resolved_scaler"):
        Detector("svm_linear").fit(X, y).manifest()
