"""Contrato ``DefensePlan`` v2 — técnica nomeada e teste de validação (D47-54).

A v1 já garantia que a *causa* citada era real (isso é
``validate_plan_against_report``). O que estes testes cobrem é a outra metade
do gate da janela — "100% das recomendações ligadas a evidência **e teste de
validação**" —, que na v1 era um ``str`` de texto livre e não era verificável
por tipo nenhum.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adversarial_ids.domain.defense_plan import (
    DEFENSE_BUCKETS,
    DEFENSE_TECHNIQUES,
    VALIDATION_METRICS,
    DefenseAction,
    DefensePlan,
    Evidence,
    ValidationTest,
    techniques_for_bucket,
)


def _test(**overrides: object) -> ValidationTest:
    data: dict[str, object] = {
        "metric": "recall",
        "direction": "increase",
        "target": 0.8,
        "procedure": "Reexecutar o detector no mesmo split e remedir o recall.",
    }
    data.update(overrides)
    return ValidationTest.model_validate(data)


def _action(**overrides: object) -> DefenseAction:
    data: dict[str, object] = {
        "description": "Validar monotonicidade de stNum/sqNum no barramento.",
        "technique": "goose_sequence_validation",
        "evidence": (Evidence(metric_or_feature="recall", value=0.42),),
        "validation_test": _test(),
    }
    data.update(overrides)
    return DefenseAction.model_validate(data)


def _plan(**overrides: object) -> DefensePlan:
    data: dict[str, object] = {
        "priority": "high",
        "detection_actions": (_action(),),
    }
    data.update(overrides)
    return DefensePlan.model_validate(data)


def _only(bucket: str, technique: str) -> dict[str, object]:
    """Um plano com uma única ação, no balde pedido — os outros dois vazios.

    Precisa zerar ``detection_actions`` junto, ou o padrão de ``_plan`` deixaria
    uma ação válida no plano e o teste passaria pelo motivo errado.
    """

    buckets: dict[str, object] = {name: () for name in DEFENSE_BUCKETS}
    buckets[bucket] = (_action(technique=technique),)
    return buckets


# --------------------------------------------------------------------------
# Vocabulário de técnicas e baldes
# --------------------------------------------------------------------------


def test_every_technique_is_legal_in_at_least_one_bucket():
    """Uma técnica sem balde seria inutilizável: nomeável e nunca aceita."""

    usable = {
        technique
        for bucket in DEFENSE_BUCKETS
        for technique in techniques_for_bucket(bucket)
    }

    assert usable == set(DEFENSE_TECHNIQUES)


@pytest.mark.parametrize("bucket", DEFENSE_BUCKETS)
def test_every_bucket_has_at_least_one_technique(bucket: str):
    """Nenhum balde pode ficar impossível de preencher."""

    assert techniques_for_bucket(bucket)


def test_techniques_for_bucket_keeps_the_declaration_order():
    """O prompt mostra esta lista ao vivo; ordem instável = prompt instável."""

    detection = techniques_for_bucket("detection_actions")

    assert detection == tuple(t for t in DEFENSE_TECHNIQUES if t in detection)
    assert detection == techniques_for_bucket("detection_actions")


def test_techniques_for_bucket_rejects_an_unknown_bucket():
    with pytest.raises(ValueError, match="Balde desconhecido"):
        techniques_for_bucket("mitigation_actions")


def test_action_rejects_a_technique_outside_the_vocabulary():
    with pytest.raises(ValidationError):
        _action(technique="pray_and_reboot")


@pytest.mark.parametrize(
    ("bucket", "technique"),
    [
        ("detection_actions", "network_segmentation"),
        ("detection_actions", "device_quarantine"),
        ("containment_actions", "detector_retraining"),
        ("containment_actions", "feature_engineering"),
        ("hardening_actions", "detector_threshold_tuning"),
        ("hardening_actions", "operator_alerting"),
    ],
)
def test_plan_rejects_a_technique_in_the_wrong_bucket(bucket: str, technique: str):
    """Segmentar a rede não detecta nada; retreinar não contém nada."""

    with pytest.raises(ValidationError, match="não pertence a"):
        _plan(**_only(bucket, technique))


@pytest.mark.parametrize(
    ("bucket", "technique"),
    [
        ("hardening_actions", "goose_sequence_validation"),
        ("containment_actions", "network_segmentation"),
        ("hardening_actions", "network_segmentation"),
        ("containment_actions", "publisher_binding"),
        ("hardening_actions", "publisher_binding"),
    ],
)
def test_dual_use_techniques_are_accepted_in_both_of_their_buckets(
    bucket: str, technique: str
):
    """Duplo uso é real: uma allowlist de publisher bloqueia *e* endurece."""

    plan = _plan(**_only(bucket, technique))

    assert getattr(plan, bucket)[0].technique == technique


# --------------------------------------------------------------------------
# ValidationTest — a metade do gate que a v1 não cobria
# --------------------------------------------------------------------------


def test_every_validation_metric_is_a_scalar_of_the_detection_report():
    """A lista existe para forçar o teste sobre algo remedível.

    ``model_name`` e ``split`` são texto (não há "aumentar o split") e
    importância de feature não é resultado defensivo — provar que uma feature
    ficou mais importante não prova que o ataque passou a ser detectado.
    """

    assert "model_name" not in VALIDATION_METRICS
    assert "split" not in VALIDATION_METRICS
    assert set(VALIDATION_METRICS) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "latency_ms",
        "confusion_matrix.tp",
        "confusion_matrix.fp",
        "confusion_matrix.fn",
        "confusion_matrix.tn",
    }


def test_validation_test_rejects_a_metric_that_is_not_remeasurable():
    with pytest.raises(ValidationError):
        _test(metric="stNum")


def test_validation_test_rejects_an_unknown_direction():
    with pytest.raises(ValidationError):
        _test(direction="melhorar")


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
def test_validation_test_rejects_a_rate_target_above_one(metric: str):
    """Um alvo de 1.5 para o F1 é uma promessa impossível, não uma meta."""

    with pytest.raises(ValidationError, match="fora do domínio"):
        _test(metric=metric, target=1.5)


@pytest.mark.parametrize("metric", ["latency_ms", "confusion_matrix.fn"])
def test_counts_and_latency_accept_targets_above_one(metric: str):
    """O limite de 1.0 vale só para taxas — 40ms e 12 FN são alvos legítimos."""

    assert _test(metric=metric, direction="at_most", target=40.0).target == 40.0


def test_validation_test_rejects_a_negative_target():
    with pytest.raises(ValidationError):
        _test(metric="latency_ms", direction="at_most", target=-1.0)


def test_validation_test_rejects_a_nan_target():
    """NaN passa em qualquer comparação numérica: o teste vira infalsificável."""

    with pytest.raises(ValidationError):
        _test(target=float("nan"))


def test_validation_test_rejects_an_infinite_target():
    with pytest.raises(ValidationError):
        _test(metric="latency_ms", direction="at_most", target=float("inf"))


def test_validation_test_still_requires_the_human_procedure():
    """A prosa da v1 não se perdeu — ela virou um campo obrigatório."""

    with pytest.raises(ValidationError):
        _test(procedure="")


def test_action_requires_a_validation_test():
    with pytest.raises(ValidationError):
        DefenseAction(
            description="Aumentar a vigilância.",
            technique="continuous_monitoring",
            evidence=(Evidence(metric_or_feature="recall", value=0.42),),
        )


def test_split_defaults_to_the_split_of_the_report():
    assert _test().split is None
    assert _test(split="train_test_70_30_seed42").split == "train_test_70_30_seed42"


# --------------------------------------------------------------------------
# Rastreabilidade do plano
# --------------------------------------------------------------------------


def test_plan_carries_its_own_report_reference():
    """``defense_plan.json`` é persistido avulso: sozinho, precisa se explicar."""

    plan = _plan(detection_report_ref="outputs/run-1/detection_report.json")

    assert plan.detection_report_ref == "outputs/run-1/detection_report.json"


def test_plan_rejects_evidence_pointing_at_another_run():
    action = _action(
        evidence=(
            Evidence(
                metric_or_feature="recall",
                value=0.42,
                detection_report_ref="outputs/run-2/detection_report.json",
            ),
        )
    )

    with pytest.raises(ValidationError, match="mas o plano declara"):
        _plan(
            detection_report_ref="outputs/run-1/detection_report.json",
            detection_actions=(action,),
        )


def test_evidence_without_a_reference_inherits_the_plan_silently():
    """``None`` na evidência significa "a do plano", não "de outra execução"."""

    plan = _plan(detection_report_ref="outputs/run-1/detection_report.json")

    assert plan.detection_actions[0].evidence[0].detection_report_ref is None


def test_a_plan_without_a_reference_does_not_constrain_its_evidence():
    """Sem referência no topo não há contradição a detectar — só falta de dado."""

    action = _action(
        evidence=(
            Evidence(
                metric_or_feature="recall",
                value=0.42,
                detection_report_ref="outputs/run-2/detection_report.json",
            ),
        )
    )

    assert _plan(detection_actions=(action,)).detection_report_ref is None


# --------------------------------------------------------------------------
# Invariantes herdados da v1
# --------------------------------------------------------------------------


def test_schema_version_is_two():
    """A troca de ``validation_method`` por ``ValidationTest`` quebra a v1."""

    assert _plan().schema_version == 2

    with pytest.raises(ValidationError):
        _plan(schema_version=1)


def test_plan_still_requires_at_least_one_action():
    with pytest.raises(ValidationError, match="ao menos uma ação"):
        DefensePlan(priority="low")


def test_action_still_requires_at_least_one_evidence():
    with pytest.raises(ValidationError):
        _action(evidence=())


def test_plan_is_frozen_and_forbids_extra_fields():
    plan = _plan()

    with pytest.raises(ValidationError):
        plan.priority = "low"

    with pytest.raises(ValidationError):
        _plan(applied=True)


def test_the_v1_field_name_is_gone():
    """Um plano da v1 não passa mais: `extra="forbid"` recusa o campo antigo."""

    with pytest.raises(ValidationError):
        DefenseAction(
            description="Revisar o limiar.",
            technique="detector_threshold_tuning",
            evidence=(Evidence(metric_or_feature="recall", value=0.42),),
            validation_method="Reavaliar recall no mesmo split.",
        )
