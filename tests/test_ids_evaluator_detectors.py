"""Integração do detector plugável no IdsEvaluator (épico E8).

Cobre o critério de pronto do E8 ("RF/DT/SVM sob o mesmo split/protocolo") no
ponto onde os detectores encontram o pipeline de dados: os quatro treinam e
avaliam pela mesma chamada, sob o mesmo split e o mesmo espaço de features do
E6/E7; o default segue byte-compatível com o Random Forest de antes do épico;
e a escala que cada detector pede é resolvida sem que o chamador precise
saber quem pede o quê.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.core.detectors import DETECTOR_KEYS, DetectorError
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain.detector_manifest import DetectorManifest


def _labeled_csv(path: Path, *, n_normal: int = 60, n_attack: int = 60) -> Path:
    """Mesmo formato de ``tests/test_ids_evaluator_selection.py``, porém com as
    classes balanceadas: aqui o alvo é comparar detectores, e um desbalanço
    forte faria o SVM RBF colapsar na classe majoritária por motivos que não
    têm nada a ver com o que este arquivo testa.
    """

    rows = ["signal,noise_a,noise_b,class"]
    for i in range(n_normal):
        rows.append(f"{i % 5},{i % 7},{(i * 3) % 11},normal")
    for i in range(n_attack):
        rows.append(f"{100 + i % 5},{i % 7},{(i * 3) % 11},attack_label")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _evaluator(**kwargs) -> IdsEvaluator:
    return IdsEvaluator(drop_cb_status=False, target_attack_label="attack_label", **kwargs)


# --------------------------------------------------------------------------- #
# Default: comportamento idêntico a antes do E8                                #
# --------------------------------------------------------------------------- #
def test_default_detector_is_random_forest_without_scaling(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator()
    evaluator.train_baseline(str(dataset))

    assert evaluator.detector.key == "random_forest"
    assert evaluator.scaler == "none"
    # ``model`` continua apontando para o estimador do sklearn: chamadores
    # anteriores ao E8 (e get_shap_importances) dependem disso.
    assert evaluator.model is evaluator.detector.estimator


def test_detector_manifest_is_none_before_training():
    assert IdsEvaluator().detector_manifest is None


def test_unknown_detector_is_rejected_at_construction():
    with pytest.raises(DetectorError, match="Detector desconhecido"):
        _evaluator(detector="xgboost")


def test_invalid_scaler_is_rejected_at_construction():
    with pytest.raises(ValueError, match="scaler inválido"):
        _evaluator(scaler="minmax")


# --------------------------------------------------------------------------- #
# Os quatro detectores sob o mesmo protocolo                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_every_detector_trains_and_evaluates_a_variant(key, tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector=key)

    baseline = evaluator.train_baseline(str(dataset))
    variant = evaluator.evaluate_variant(str(dataset))

    for metrics in (baseline, variant):
        assert 0.0 <= metrics["f1_score_attack"] <= 1.0
        assert metrics["tp"] is not None  # matriz binária completa
        assert metrics["used_features"] == evaluator.feature_columns


def test_all_detectors_share_the_same_split_and_feature_space(tmp_path):
    """O que torna a comparação do E8 honesta: os quatro veem exatamente as
    mesmas colunas e o mesmo tamanho de partição de teste. Só o modelo muda.
    """

    dataset = _labeled_csv(tmp_path / "baseline.csv")

    features: set[tuple[str, ...]] = set()
    supports: set[int] = set()
    for key in DETECTOR_KEYS:
        evaluator = _evaluator(detector=key)
        metrics = evaluator.train_baseline(str(dataset))
        features.add(tuple(evaluator.feature_columns))
        supports.add(metrics["support_attack"])

    assert len(features) == 1
    assert len(supports) == 1


# --------------------------------------------------------------------------- #
# Escala: recomendada pelo detector, sobrescritível pelo chamador              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("random_forest", "none"),
        ("decision_tree", "none"),
        ("svm_linear", "standard"),
        ("svm_rbf", "standard"),
    ],
)
def test_scaler_defaults_to_what_the_detector_asks_for(key, expected, tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector=key)
    evaluator.train_baseline(str(dataset))

    assert evaluator.scaler == expected
    # A recomendação tem que chegar ao preprocessador de fato, não só ao
    # atributo: é o FeatureManifest que prova o que foi aplicado.
    assert evaluator.feature_manifest.scaler == expected
    assert evaluator.detector_manifest.resolved_scaler == expected


def test_explicit_scaler_overrides_the_detector_recommendation(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector="svm_linear", scaler="none")
    evaluator.train_baseline(str(dataset))

    manifest = evaluator.detector_manifest
    # A discordância é legítima (ablação) e fica registrada nos dois campos.
    assert manifest.requires_scaling is True
    assert manifest.resolved_scaler == "none"
    assert evaluator.feature_manifest.scaler == "none"


# --------------------------------------------------------------------------- #
# Manifest do detector no fim do pipeline                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_detector_manifest_matches_what_was_actually_trained(key, tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector=key)
    evaluator.train_baseline(str(dataset))

    manifest = evaluator.detector_manifest
    assert isinstance(manifest, DetectorManifest)
    assert manifest.detector == key
    # trained_features é o espaço PÓS-seleção (E7), o mesmo que o modelo viu.
    assert list(manifest.trained_features) == evaluator.feature_columns
    # trained_rows conta linhas de TREINO pós-undersampling — nunca o dataset
    # inteiro nem a partição de teste (mesma disciplina do E6/E7).
    assert manifest.trained_rows < 120
    assert (
        manifest.trained_rows
        == evaluator.selection_manifest.fitted_rows_after_undersampling
    )


def test_detector_manifest_composes_with_feature_selection(tmp_path):
    # O detector treina sobre o que o E7 deixou passar, não sobre as
    # candidatas do E6 — se essa ordem invertesse, trained_features mentiria.
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(
        detector="decision_tree", feature_selection="mutual_info", feature_selection_top_k=1
    )
    evaluator.train_baseline(str(dataset))

    assert evaluator.detector_manifest.trained_features == ("signal",)


# --------------------------------------------------------------------------- #
# Importâncias e SHAP por detector                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ("random_forest", "decision_tree", "svm_linear"))
def test_detectors_with_importances_populate_the_metrics(key, tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector=key)
    metrics = evaluator.train_baseline(str(dataset))

    assert metrics["top_feature_importances"]
    assert metrics["top_feature_importances"][0]["feature"] == "signal"


def test_svm_rbf_reports_no_importances_without_breaking_the_metrics(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector="svm_rbf")
    metrics = evaluator.train_baseline(str(dataset))

    # Vazio, não ausente: o resto do dicionário de métricas continua completo,
    # e o Blue Team simplesmente tem menos evidência citável nessa execução.
    assert metrics["top_feature_importances"] == []
    assert metrics["f1_score_attack"] is not None


@pytest.mark.parametrize("key", ("svm_linear", "svm_rbf"))
def test_shap_is_not_attempted_on_non_tree_detectors(key, tmp_path):
    # TreeExplainer não sabe ler um SVM — melhor devolver [] pelo caminho
    # best-effort do que aplicar um explicador errado ao modelo errado.
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(detector=key)
    evaluator.train_baseline(str(dataset))

    assert evaluator.get_shap_importances(str(dataset)) == []


# --------------------------------------------------------------------------- #
# Reprodutibilidade                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_same_seed_gives_the_same_metrics(key, tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")

    first = _evaluator(detector=key)
    first.train_baseline(str(dataset))

    second = _evaluator(detector=key)
    second.train_baseline(str(dataset))

    assert first.evaluate_variant(str(dataset)) == second.evaluate_variant(str(dataset))


# --------------------------------------------------------------------------- #
# n_estimators: hiperparâmetro exclusivo do Random Forest                      #
# --------------------------------------------------------------------------- #
def test_n_estimators_reaches_the_random_forest_and_is_echoed_back(tmp_path):
    evaluator = _evaluator(n_estimators=7)
    assert evaluator.detector.hyperparameters["n_estimators"] == 7
    # O atributo espelha o que o estimador recebeu — nunca um valor que o
    # modelo não usa.
    assert evaluator.n_estimators == 7


def test_n_estimators_is_rejected_for_a_detector_that_has_no_such_concept():
    # Antes seria descartado em silêncio: o usuário acreditaria ter configurado
    # 250 árvores e receberia uma só.
    with pytest.raises(ValueError, match="só se aplica a detector='random_forest'"):
        _evaluator(n_estimators=250, detector="decision_tree")


def test_n_estimators_from_two_sources_is_rejected_instead_of_silently_picking_one():
    with pytest.raises(ValueError, match="informado duas vezes"):
        _evaluator(n_estimators=250, detector_hyperparameters={"n_estimators": 300})


def test_n_estimators_is_none_for_detectors_that_do_not_have_it():
    assert _evaluator(detector="svm_rbf").n_estimators is None


# --------------------------------------------------------------------------- #
# Limite da garantia de "mesmo espaço de features" (E7 × E8)                   #
# --------------------------------------------------------------------------- #
def test_forcing_the_same_scaler_pins_the_selected_features_across_detectors(tmp_path):
    """Com seleção de features ligada, "mesmo protocolo" exige escala forçada.

    A seleção do E7 é ajustada sobre o X **já escalado**, e
    ``mutual_info_classif`` não é estritamente invariante a escala (o ruído que
    ele injeta é proporcional à amplitude da coluna). Deixar cada detector
    resolver a própria escala pode, em princípio, entregar conjuntos de
    features diferentes a uma árvore e a um SVM. Forçar ``scaler=`` é o
    remédio — e é o que este teste fixa como garantia.
    """

    dataset = _labeled_csv(tmp_path / "baseline.csv")

    selected: set[tuple[str, ...]] = set()
    for key in DETECTOR_KEYS:
        evaluator = _evaluator(
            detector=key,
            scaler="standard",
            feature_selection="mutual_info",
            feature_selection_top_k=2,
        )
        evaluator.train_baseline(str(dataset))
        selected.add(tuple(evaluator.feature_columns))

    assert len(selected) == 1
