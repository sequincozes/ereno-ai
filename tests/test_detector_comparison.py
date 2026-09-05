"""Relatório comparativo de detectores (épico E8, gate da janela D36-46).

Cobre o gate de saída que o registro de detectores sozinho não fecha: "mesmo
split/protocolo; relatório comparativo; fallback RF preservado", com o
"ranking por F1/recall/latência" que a janela pede. Três propriedades são o
alvo: o ranking é recomputável (não uma afirmação do produtor), uma falha
isolada não derruba os demais, e "mesmo protocolo" é verificado em vez de
prometido.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from adversarial_ids.core.detector_comparison import (
    compare_detectors,
    format_comparison_table,
)
from adversarial_ids.core.detectors import DETECTOR_KEYS, DetectorError
from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.domain.detector_comparison import (
    DetectorComparison,
    DetectorRun,
    rank_runs,
)

_SPLIT = "train_test_70_30_seed42"


def _labeled_csv(path: Path, *, n_normal: int = 60, n_attack: int = 60) -> Path:
    rows = ["signal,noise_a,noise_b,class"]
    for i in range(n_normal):
        rows.append(f"{i % 5},{i % 7},{(i * 3) % 11},normal")
    for i in range(n_attack):
        rows.append(f"{100 + i % 5},{i % 7},{(i * 3) % 11},attack_label")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _report(detector: str, *, f1: float, recall: float, latency: float) -> DetectionReport:
    return DetectionReport(
        model_name=detector,
        split=_SPLIT,
        accuracy=f1,
        precision=f1,
        recall=recall,
        f1=f1,
        confusion_matrix=ConfusionMatrix(tp=10, fp=1, fn=1, tn=10),
        latency_ms=latency,
    )


def _run(detector: str, *, f1=0.9, recall=0.9, latency=10.0, features=("a", "b")) -> DetectorRun:
    return DetectorRun(
        detector=detector,
        status="succeeded",
        report=_report(detector, f1=f1, recall=recall, latency=latency),
        resolved_scaler="none",
        importance_kind="gini" if detector.startswith(("random", "decision")) else "none",
        trained_rows=70,
        trained_features=features,
    )


def _failed(detector: str, error: str = "RuntimeError: estourou") -> DetectorRun:
    return DetectorRun(detector=detector, status="failed", error=error)


def _comparison(runs: tuple[DetectorRun, ...], **overrides) -> DetectorComparison:
    data: dict[str, object] = {
        "baseline_dataset": "b.csv",
        "variant_dataset": "v.csv",
        "split": _SPLIT,
        "runs": runs,
        "ranking": rank_runs(runs, metric=overrides.get("ranking_metric", "f1")),
    }
    data.update(overrides)
    return DetectorComparison.model_validate(data)


# --------------------------------------------------------------------------- #
# Ranking por F1 / recall / latência                                           #
# --------------------------------------------------------------------------- #
def test_ranking_orders_by_f1_descending_by_default():
    runs = (
        _run("random_forest", f1=0.80),
        _run("decision_tree", f1=0.95),
        _run("svm_linear", f1=0.60),
    )
    assert rank_runs(runs) == ("decision_tree", "random_forest", "svm_linear")


def test_recall_then_latency_break_an_f1_tie():
    # As três métricas da janela entram sempre: a escolhida manda, as outras
    # duas desempatam.
    runs = (
        _run("random_forest", f1=1.0, recall=1.0, latency=150.0),
        _run("decision_tree", f1=1.0, recall=1.0, latency=85.0),
        _run("svm_rbf", f1=1.0, recall=0.9, latency=10.0),
    )
    assert rank_runs(runs) == ("decision_tree", "random_forest", "svm_rbf")


def test_ranking_by_latency_prefers_the_fastest():
    runs = (
        _run("random_forest", f1=1.0, latency=150.0),
        _run("svm_linear", f1=0.5, latency=9.0),
    )
    assert rank_runs(runs, metric="latency_ms") == ("svm_linear", "random_forest")


def test_ranking_by_recall_prefers_the_highest_recall():
    runs = (
        _run("random_forest", f1=0.95, recall=0.90),
        _run("svm_rbf", f1=0.80, recall=0.99),
    )
    assert rank_runs(runs, metric="recall") == ("svm_rbf", "random_forest")


def test_an_exact_three_way_tie_breaks_deterministically_by_name():
    # Empate exato nas três métricas é comum num dataset fácil (RF e DT acertam
    # tudo). Sem o último critério, o mesmo experimento produziria rankings
    # diferentes conforme a ordem de construção da lista.
    forward = (_run("random_forest"), _run("decision_tree"))
    backward = (_run("decision_tree"), _run("random_forest"))
    assert rank_runs(forward) == rank_runs(backward) == ("decision_tree", "random_forest")


def test_failed_detectors_have_no_position_in_the_ranking():
    runs = (_run("random_forest"), _failed("svm_rbf"))
    assert rank_runs(runs) == ("random_forest",)


# --------------------------------------------------------------------------- #
# Contrato: o ranking é recomputado, não aceito na palavra                     #
# --------------------------------------------------------------------------- #
def test_contract_rejects_a_ranking_that_does_not_match_the_metrics():
    runs = (_run("random_forest", f1=0.9), _run("decision_tree", f1=0.5))
    with pytest.raises(ValidationError, match="não corresponde à ordem recomputada"):
        _comparison(runs, ranking=("decision_tree", "random_forest"))


def test_contract_is_versioned_and_survives_a_json_round_trip():
    comparison = _comparison((_run("random_forest"), _failed("svm_rbf")))
    assert comparison.schema_version == 1
    assert DetectorComparison.model_validate(comparison.model_dump(mode="json")) == comparison


def test_contract_rejects_a_report_that_names_another_detector():
    # Sem esta guarda, uma linha poderia carregar os números de outro modelo --
    # o mesmo erro que o E8 corrigiu em build_detection_report.
    with pytest.raises(ValidationError, match="diz model_name"):
        DetectorRun(
            detector="decision_tree",
            status="succeeded",
            report=_report("random_forest", f1=1.0, recall=1.0, latency=1.0),
            trained_features=("a",),
        )


def test_contract_rejects_a_split_that_differs_from_the_declared_one():
    run = _run("random_forest")
    with pytest.raises(ValidationError, match="sem partição comum não há comparação"):
        _comparison((run,), split="train_test_50_50_seed7")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"status": "succeeded", "error": "x"}, "não pode carregar error"),
        ({"status": "failed"}, "exige a causa em error"),
    ],
)
def test_contract_keeps_status_and_payload_coherent(kwargs, match):
    base: dict[str, object] = {"detector": "random_forest", "trained_features": ("a",)}
    if kwargs["status"] == "succeeded":
        base["report"] = _report("random_forest", f1=1.0, recall=1.0, latency=1.0)
    with pytest.raises(ValidationError, match=match):
        DetectorRun.model_validate({**base, **kwargs})


def test_contract_rejects_a_repeated_detector():
    with pytest.raises(ValidationError, match="detector repetido"):
        _comparison((_run("random_forest"), _run("random_forest")))


def test_protocol_flag_must_agree_with_the_feature_spaces():
    runs = (_run("random_forest", features=("a", "b")), _run("svm_linear", features=("a",)))
    with pytest.raises(ValidationError, match="espaços de features diferentes"):
        _comparison(runs, protocol_consistent=True)


def test_an_inconsistent_protocol_must_carry_an_explanation():
    with pytest.raises(ValidationError, match="exige protocol_notes"):
        _comparison((_run("random_forest"),), protocol_consistent=False)


# --------------------------------------------------------------------------- #
# compare_detectors: integração sobre o pipeline real                          #
# --------------------------------------------------------------------------- #
def test_every_registered_detector_appears_in_the_comparison(tmp_path):
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(dataset, dataset, target_attack_label="attack_label")

    assert {run.detector for run in comparison.runs} == set(DETECTOR_KEYS)
    assert comparison.protocol_consistent
    assert comparison.winner in DETECTOR_KEYS
    # Todos medidos sob a mesma partição -- é o que "mesmo protocolo" significa.
    assert {run.report.split for run in comparison.runs if run.report} == {comparison.split}


def test_each_detector_keeps_the_scaler_it_asked_for(tmp_path):
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(dataset, dataset, target_attack_label="attack_label")

    scalers = {run.detector: run.resolved_scaler for run in comparison.runs}
    assert scalers["random_forest"] == "none"
    assert scalers["decision_tree"] == "none"
    assert scalers["svm_linear"] == "standard"
    assert scalers["svm_rbf"] == "standard"


def test_an_isolated_failure_does_not_bring_down_the_others(tmp_path, monkeypatch):
    """Exigência literal da tabela de aceite da camada Detecção."""

    from adversarial_ids.core import detector_comparison as module

    real = module.IdsEvaluator

    def exploding(*args, **kwargs):
        if kwargs.get("detector") == "svm_rbf":
            raise MemoryError("kernel cache estourou")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "IdsEvaluator", exploding)

    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(dataset, dataset, target_attack_label="attack_label")

    by_key = {run.detector: run for run in comparison.runs}
    assert by_key["svm_rbf"].status == "failed"
    assert "kernel cache estourou" in by_key["svm_rbf"].error
    # Os outros três continuam medidos e ranqueados.
    assert all(by_key[k].status == "succeeded" for k in DETECTOR_KEYS if k != "svm_rbf")
    assert "svm_rbf" not in comparison.ranking
    assert len(comparison.ranking) == len(DETECTOR_KEYS) - 1
    # A falha vira nota, mas não invalida a comparação entre os que rodaram.
    assert comparison.protocol_consistent
    assert any("svm_rbf" in note for note in comparison.protocol_notes)


def test_a_comparison_where_nothing_trains_raises_instead_of_inventing_a_split(
    tmp_path, monkeypatch
):
    from adversarial_ids.core import detector_comparison as module

    monkeypatch.setattr(
        module, "IdsEvaluator", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nada roda"))
    )

    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    with pytest.raises(RuntimeError, match="não há protocolo comum"):
        compare_detectors(dataset, dataset, target_attack_label="attack_label")


def test_forcing_one_scaler_keeps_the_protocol_consistent_with_selection_on(tmp_path):
    # O furo conhecido do E7 x E8: seleção ajustada sobre X já escalado. Com a
    # escala forçada, o espaço de features é necessariamente o mesmo.
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(
        dataset,
        dataset,
        target_attack_label="attack_label",
        scaler="standard",
        evaluator_kwargs={"feature_selection": "mutual_info", "feature_selection_top_k": 2},
    )

    assert comparison.protocol_consistent
    assert len({run.trained_features for run in comparison.runs if run.report}) == 1


def test_a_divergent_feature_space_is_reported_not_hidden(tmp_path, monkeypatch):
    """Se os detectores divergirem de espaço, o relatório diz -- não afirma paridade."""

    from adversarial_ids.core import detector_comparison as module

    real = module.IdsEvaluator

    def narrowing(*args, **kwargs):
        # Só o SVM linear enxerga uma feature a menos, simulando a divergência
        # que a seleção do E7 poderia produzir entre escalas diferentes.
        if kwargs.get("detector") == "svm_linear":
            kwargs = {**kwargs, "feature_selection": "mutual_info", "feature_selection_top_k": 1}
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "IdsEvaluator", narrowing)

    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(dataset, dataset, target_attack_label="attack_label")

    assert comparison.protocol_consistent is False
    assert any("espaços de features diferentes" in note for note in comparison.protocol_notes)
    assert any("scaler=" in note for note in comparison.protocol_notes)


def test_only_the_requested_detectors_run_and_duplicates_collapse(tmp_path):
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    comparison = compare_detectors(
        dataset,
        dataset,
        detectors=["decision_tree", "random_forest", "decision_tree"],
        target_attack_label="attack_label",
    )

    assert [run.detector for run in comparison.runs] == ["decision_tree", "random_forest"]


def test_unknown_detector_is_rejected_before_any_training(tmp_path):
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    with pytest.raises(DetectorError, match="Detector desconhecido"):
        compare_detectors(dataset, dataset, detectors=["random_forest", "xgboost"])


def test_an_empty_detector_list_is_rejected(tmp_path):
    dataset = str(_labeled_csv(tmp_path / "b.csv"))
    with pytest.raises(ValueError, match="pelo menos um detector"):
        compare_detectors(dataset, dataset, detectors=[])


# --------------------------------------------------------------------------- #
# Tabela de texto                                                              #
# --------------------------------------------------------------------------- #
def test_the_table_shows_the_ranking_the_winner_and_the_failures():
    comparison = _comparison(
        (_run("random_forest", f1=0.9), _failed("svm_rbf", "MemoryError: kernel")),
        protocol_consistent=True,
    )
    table = format_comparison_table(comparison)

    assert "random_forest" in table
    assert "FALHOU" in table and "MemoryError: kernel" in table
    assert "vencedor: random_forest" in table
    assert _SPLIT in table


def test_the_table_warns_when_the_protocol_is_not_uniform():
    comparison = _comparison(
        (_run("random_forest", features=("a", "b")), _run("svm_linear", features=("a",))),
        protocol_consistent=False,
        protocol_notes=("espaços diferentes",),
    )
    assert "AVISO: protocolo NÃO uniforme" in format_comparison_table(comparison)
