"""Integração de undersampling + seleção de features no IdsEvaluator (épico E7).

Cobre o critério de pronto do E7 ("estratégias configuráveis com ablação") no
ponto onde as duas estratégias realmente encontram o Random Forest: por
default (``"none"``/``"none"``) o comportamento é idêntico a antes do épico;
ativadas, produzem um ``SelectionManifest`` consistente, reduzem/balanceiam o
que o modelo vê, e reproduzem a mesma seleção/métricas com a mesma seed —
sem nunca deixar o undersampling tocar a partição de teste.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain.selection_manifest import SelectionManifest


def _labeled_csv(path: Path, *, n_normal: int = 60, n_attack: int = 20) -> Path:
    """CSV com uma feature informativa (`signal`), duas ruidosas e classes
    desbalanceadas (normal >> ataque) — o suficiente para mutual_info
    distinguir sinal de ruído e para undersampling ter o que balancear.
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
# Default ("none"/"none"): comportamento idêntico a antes do E7               #
# --------------------------------------------------------------------------- #
def test_default_strategies_keep_all_candidate_features_and_all_train_rows(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator()
    evaluator.train_baseline(str(dataset))

    manifest = evaluator.selection_manifest
    assert isinstance(manifest, SelectionManifest)
    assert manifest.feature_selection == "none"
    assert manifest.undersampling == "none"
    assert manifest.selected_features == manifest.candidate_features
    assert manifest.fitted_rows_before_undersampling == manifest.fitted_rows_after_undersampling
    assert evaluator.feature_columns == list(manifest.candidate_features)


def test_selection_manifest_is_none_before_training():
    assert IdsEvaluator().selection_manifest is None


# --------------------------------------------------------------------------- #
# Seleção de features (mutual information)                                    #
# --------------------------------------------------------------------------- #
def test_mutual_info_selection_drops_noise_features_and_shrinks_model_input(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")
    evaluator = _evaluator(feature_selection="mutual_info", feature_selection_top_k=1)
    evaluator.train_baseline(str(dataset))

    assert evaluator.feature_columns == ["signal"]

    manifest = evaluator.selection_manifest
    assert manifest.feature_selection == "mutual_info"
    assert manifest.selected_features == ("signal",)
    assert set(manifest.feature_scores) == {"signal", "noise_a", "noise_b"}
    assert manifest.feature_scores["signal"] > manifest.feature_scores["noise_a"]

    # evaluate_variant tem que recortar para o mesmo espaço selecionado —
    # sem isso o predict quebraria por descompasso de colunas.
    metrics = evaluator.evaluate_variant(str(dataset))
    assert metrics["used_features"] == ["signal"]


def test_mutual_info_selection_is_reproducible_with_the_same_seed(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv")

    first = _evaluator(feature_selection="mutual_info", feature_selection_top_k=1)
    first.train_baseline(str(dataset))

    second = _evaluator(feature_selection="mutual_info", feature_selection_top_k=1)
    second.train_baseline(str(dataset))

    assert first.selection_manifest.selected_features == second.selection_manifest.selected_features
    assert first.selection_manifest.feature_scores == second.selection_manifest.feature_scores


# --------------------------------------------------------------------------- #
# Undersampling                                                                #
# --------------------------------------------------------------------------- #
def test_random_undersampling_balances_training_classes(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv", n_normal=60, n_attack=20)
    evaluator = _evaluator(undersampling="random")
    evaluator.train_baseline(str(dataset))

    manifest = evaluator.selection_manifest
    assert manifest.undersampling == "random"
    assert manifest.fitted_rows_after_undersampling < manifest.fitted_rows_before_undersampling

    after_counts = set(manifest.class_counts_after.values())
    assert len(after_counts) == 1  # as duas classes ficam com a mesma contagem


def test_undersampling_never_touches_the_test_partition(tmp_path):
    # Sem undersampling, o dataset tem 80 linhas -> test_size=0.3 -> 24 no
    # teste. Ativar undersampling não pode mudar esse número: undersampling
    # só se aplica ao treino, depois do split.
    dataset = _labeled_csv(tmp_path / "baseline.csv", n_normal=60, n_attack=20)

    without = _evaluator()
    without.train_baseline(str(dataset))
    baseline_metrics = without.evaluate_variant(str(dataset))

    with_undersampling = _evaluator(undersampling="random")
    with_undersampling.train_baseline(str(dataset))
    undersampled_metrics = with_undersampling.evaluate_variant(str(dataset))

    assert baseline_metrics["dataset_rows"] == undersampled_metrics["dataset_rows"] == 80


def test_random_undersampling_is_reproducible_with_the_same_seed(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv", n_normal=60, n_attack=20)

    first = _evaluator(undersampling="random")
    first.train_baseline(str(dataset))

    second = _evaluator(undersampling="random")
    second.train_baseline(str(dataset))

    assert first.selection_manifest.class_counts_after == second.selection_manifest.class_counts_after
    assert (
        first.selection_manifest.fitted_rows_after_undersampling
        == second.selection_manifest.fitted_rows_after_undersampling
    )


# --------------------------------------------------------------------------- #
# Ordem do pipeline: seleção primeiro, undersampling depois                    #
# --------------------------------------------------------------------------- #
def test_feature_selection_and_undersampling_compose(tmp_path):
    dataset = _labeled_csv(tmp_path / "baseline.csv", n_normal=60, n_attack=20)
    evaluator = _evaluator(
        feature_selection="mutual_info",
        feature_selection_top_k=1,
        undersampling="random",
    )
    evaluator.train_baseline(str(dataset))

    manifest = evaluator.selection_manifest
    assert manifest.selected_features == ("signal",)
    assert manifest.fitted_rows_after_undersampling < manifest.fitted_rows_before_undersampling
    assert evaluator.feature_columns == ["signal"]

    # O modelo treinado só conhece "signal" — avaliar uma variante continua
    # funcionando (mesmo espaço de colunas dos dois lados).
    metrics = evaluator.evaluate_variant(str(dataset))
    assert metrics["used_features"] == ["signal"]
