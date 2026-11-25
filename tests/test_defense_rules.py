"""Testes da avaliação por regras do DefensePlan (épico E5, janela D47-54)."""

from __future__ import annotations

import pytest

from adversarial_ids.config.defense_techniques import (
    METRIC_TECHNIQUES,
    playbook_for_attack,
    techniques_for_evidence,
    techniques_for_feature,
)
from adversarial_ids.core.defense_rules import evaluate_plan_rules
from adversarial_ids.domain import (
    DEFENSE_RULE_IDS,
    DefenseRuleReport,
    RuleFinding,
    blocking_rules,
    severity_of,
)
from adversarial_ids.domain.detection_report import DetectionReport


def build_report(
    *,
    top_features: list[dict] | None = None,
    recall: float = 0.60,
) -> DetectionReport:
    if top_features is None:
        top_features = [
            {"feature": "isbATrapAreaSum", "importance": 0.42},
            {"feature": "cbStatusDiff", "importance": 0.30},
        ]

    return DetectionReport.model_validate(
        {
            "model_name": "random_forest",
            "split": "train_test_80_20_seed42",
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": recall,
            "f1": 0.65,
            "confusion_matrix": {"tp": 12, "fp": 4, "fn": 8, "tn": 20},
            "top_features": top_features,
            "latency_ms": 3.5,
        }
    )


def build_action(
    *,
    technique: str = "detector_threshold_tuning",
    evidence: list[dict] | None = None,
) -> dict:
    if evidence is None:
        evidence = [{"metric_or_feature": "recall", "value": 0.60}]

    return {
        "description": "Ação de teste.",
        "technique": technique,
        "evidence": evidence,
        "validation_test": {
            "metric": "recall",
            "direction": "increase",
            "target": 0.80,
            "procedure": "Reavaliar o recall no mesmo split.",
        },
    }


def build_plan(**kwargs) -> dict:
    buckets = {
        "detection_actions": kwargs.pop("detection_actions", None),
        "containment_actions": kwargs.pop("containment_actions", []),
        "hardening_actions": kwargs.pop("hardening_actions", []),
    }
    if buckets["detection_actions"] is None:
        buckets["detection_actions"] = [build_action(**kwargs)]

    return {"priority": "medium", **buckets}


# --------------------------------------------------------------------------- #
# Lastro: a técnica responde à evidência que a própria ação cita              #
# --------------------------------------------------------------------------- #
def test_technique_catalogued_for_the_cited_metric_is_grounded():
    rules = evaluate_plan_rules(build_plan(), build_report())

    assert rules.is_grounded
    assert rules.grounded_actions == rules.total_actions == 1
    assert rules.grounded_fraction == 1.0
    assert rules.findings == ()


def test_technique_unrelated_to_the_cited_evidence_blocks():
    """O buraco que a janela D47-54 fecha: citar recall e autenticar publisher.

    As duas metades são defensáveis sozinhas — o recall é evidência real e
    `goose_authentication` é técnica legítima de hardening —, e por isso o
    portão de evidência aceitava a ação inteira.
    """

    plan = build_plan(
        detection_actions=[],
        hardening_actions=[
            build_action(
                technique="goose_authentication",
                evidence=[{"metric_or_feature": "recall", "value": 0.60}],
            )
        ],
    )

    rules = evaluate_plan_rules(plan, build_report())

    assert not rules.is_grounded
    assert rules.grounded_actions == 0
    (finding,) = rules.blocking_findings
    assert finding.rule == "technique_not_grounded"
    assert finding.bucket == "hardening_actions"
    assert finding.action_index == 0
    assert finding.technique == "goose_authentication"
    # A mensagem precisa dizer o que *serviria*, não só que está errado.
    assert "detector_threshold_tuning" in finding.message


def test_one_grounding_evidence_is_enough():
    """Exigir que a técnica responda a *todas* as evidências puniria o plano
    mais bem fundamentado — o que cita a métrica que dói e a feature que explica.
    """

    plan = build_plan(
        evidence=[
            {"metric_or_feature": "recall", "value": 0.60},
            {"metric_or_feature": "isbATrapAreaSum", "value": 0.42},
        ]
    )

    rules = evaluate_plan_rules(plan, build_report())

    assert rules.is_grounded
    assert rules.findings == ()


def test_grounding_is_counted_per_action_across_every_bucket():
    plan = build_plan(
        detection_actions=[
            build_action(technique="detector_threshold_tuning"),
            build_action(technique="goose_timing_analysis"),
        ],
        containment_actions=[
            build_action(
                technique="operator_alerting",
                evidence=[{"metric_or_feature": "confusion_matrix.fn", "value": 8}],
            )
        ],
    )

    rules = evaluate_plan_rules(plan, build_report())

    assert rules.total_actions == 3
    assert rules.grounded_actions == 2
    assert rules.grounded_fraction == pytest.approx(2 / 3)
    (finding,) = rules.blocking_findings
    assert finding.technique == "goose_timing_analysis"
    assert finding.action_index == 1


# --------------------------------------------------------------------------- #
# Evidência que não sustenta nada — e por dois motivos diferentes             #
# --------------------------------------------------------------------------- #
def test_descriptive_evidence_is_flagged_without_blocking_on_its_own():
    plan = build_plan(
        evidence=[
            {"metric_or_feature": "model_name", "value": "random_forest"},
            {"metric_or_feature": "recall", "value": 0.60},
        ]
    )

    rules = evaluate_plan_rules(plan, build_report())

    assert rules.is_grounded
    (finding,) = rules.findings
    assert finding.rule == "evidence_not_actionable"
    assert finding.severity == "advisory"
    assert finding.subjects == ("model_name",)


def test_descriptive_evidence_alone_leaves_the_action_ungrounded():
    plan = build_plan(
        evidence=[{"metric_or_feature": "split", "value": "train_test_80_20_seed42"}]
    )

    rules = evaluate_plan_rules(plan, build_report())

    assert not rules.is_grounded
    rule_ids = {finding.rule for finding in rules.findings}
    assert rule_ids == {"evidence_not_actionable", "technique_not_grounded"}
    (blocking,) = rules.blocking_findings
    assert "nenhuma" in blocking.message


def test_feature_outside_the_catalogue_is_advisory_not_a_lie():
    """Feature real que o catálogo ainda não cobre é lacuna do catálogo.

    Ela não pode ser confundida com evidência inventada: essa o portão de
    evidência já recusa antes, comparando com o relatório.
    """

    report = build_report(
        top_features=[{"feature": "featureNova", "importance": 0.42}]
    )
    plan = build_plan(
        evidence=[
            {"metric_or_feature": "featureNova", "value": 0.42},
            {"metric_or_feature": "recall", "value": 0.60},
        ]
    )

    rules = evaluate_plan_rules(plan, report)

    assert rules.is_grounded
    (finding,) = rules.findings
    assert finding.rule == "uncatalogued_feature"
    assert finding.subjects == ("featureNova",)


# --------------------------------------------------------------------------- #
# Playbook IEC-61850: aconselha, nunca recusa                                 #
# --------------------------------------------------------------------------- #
def test_playbook_axis_is_off_until_the_attack_is_named():
    rules = evaluate_plan_rules(build_plan(), build_report())

    assert rules.attack_key is None
    assert rules.playbook_key is None
    # Sem ataque não há playbook a cobrar, e o relatório diz isso em vez de
    # deixar a ausência de achados parecer cobertura completa.
    assert not [
        finding for finding in rules.findings if finding.rule.startswith("playbook_")
    ]


def test_playbook_reports_the_techniques_the_plan_never_proposed():
    rules = evaluate_plan_rules(
        build_plan(), build_report(), attack_key="masquerade_fault"
    )

    playbook = playbook_for_attack("masquerade_fault")
    assert rules.playbook_key == playbook.key
    (finding,) = [
        f for f in rules.findings if f.rule == "playbook_technique_missing"
    ]
    assert finding.severity == "advisory"
    assert set(finding.subjects) == set(playbook.techniques)
    # Conselho, jamais recusa: o plano continua aceitável.
    assert rules.is_grounded


def test_playbook_only_charges_a_signature_the_report_actually_surfaced():
    """Cobrar uma feature ausente de `top_features` seria cobrar do plano algo
    que o relatório não tinha como mostrar — o caso de quase todo o playbook de
    replay sob o modo de features default."""

    surfaced = build_report(
        top_features=[{"feature": "isbATrapAreaSum", "importance": 0.42}]
    )
    hidden = build_report(top_features=[])

    ignored = [
        f
        for f in evaluate_plan_rules(
            build_plan(), surfaced, attack_key="masquerade_fault"
        ).findings
        if f.rule == "playbook_signature_ignored"
    ]
    assert ignored and ignored[0].subjects == ("isbATrapAreaSum",)

    assert not [
        f
        for f in evaluate_plan_rules(
            build_plan(), hidden, attack_key="masquerade_fault"
        ).findings
        if f.rule == "playbook_signature_ignored"
    ]


def test_signature_cited_by_any_action_is_not_reported_as_ignored():
    report = build_report(
        top_features=[{"feature": "isbATrapAreaSum", "importance": 0.42}]
    )
    plan = build_plan(
        technique="physical_consistency_check",
        evidence=[{"metric_or_feature": "isbATrapAreaSum", "value": 0.42}],
    )

    rules = evaluate_plan_rules(plan, report, attack_key="masquerade_fault")

    assert not [
        f for f in rules.findings if f.rule == "playbook_signature_ignored"
    ]


def test_unknown_attack_key_raises_instead_of_silently_skipping():
    with pytest.raises(KeyError):
        evaluate_plan_rules(
            build_plan(), build_report(), attack_key="ataque_inexistente"
        )


# --------------------------------------------------------------------------- #
# O relatório em si                                                           #
# --------------------------------------------------------------------------- #
def test_report_carries_the_run_reference_it_was_given():
    rules = evaluate_plan_rules(
        build_plan(),
        build_report(),
        detection_report_ref="outputs/run/detection_report.json",
    )

    assert rules.detection_report_ref == "outputs/run/detection_report.json"


def test_report_falls_back_to_the_reference_declared_in_the_plan():
    plan = build_plan()
    plan["detection_report_ref"] = "outputs/run/detection_report.json"

    rules = evaluate_plan_rules(plan, build_report())

    assert rules.detection_report_ref == "outputs/run/detection_report.json"


def test_report_rejects_a_count_that_contradicts_its_findings():
    """O número que a janela cobra não pode ser afirmado sem os achados."""

    with pytest.raises(ValueError, match="sem lastro"):
        DefenseRuleReport.model_validate(
            {
                "total_actions": 2,
                "grounded_actions": 1,
                "findings": [],
            }
        )


def test_report_rejects_more_grounded_actions_than_it_has():
    with pytest.raises(ValueError, match="excede"):
        DefenseRuleReport.model_validate(
            {"total_actions": 1, "grounded_actions": 2, "findings": []}
        )


def test_finding_severity_cannot_disagree_with_its_rule():
    """Um achado bloqueante rotulado `advisory` passaria pelo portão."""

    with pytest.raises(ValueError, match="Severidade incoerente"):
        RuleFinding.model_validate(
            {
                "rule": "technique_not_grounded",
                "severity": "advisory",
                "bucket": "detection_actions",
                "action_index": 0,
                "message": "Achado com severidade rebaixada.",
            }
        )


def test_action_finding_needs_both_halves_of_its_location():
    with pytest.raises(ValueError, match="bucket e action_index"):
        RuleFinding.model_validate(
            {
                "rule": "uncatalogued_feature",
                "severity": "advisory",
                "bucket": "detection_actions",
                "message": "Sem índice da ação.",
            }
        )


def test_only_the_grounding_rule_blocks():
    assert blocking_rules() == ("technique_not_grounded",)
    assert all(severity_of(rule) in ("blocking", "advisory") for rule in DEFENSE_RULE_IDS)


# --------------------------------------------------------------------------- #
# A consulta única que prompt e portão compartilham                           #
# --------------------------------------------------------------------------- #
def test_evidence_lookup_dispatches_to_the_right_catalogue_half():
    assert techniques_for_evidence("recall") == METRIC_TECHNIQUES["recall"]
    assert techniques_for_evidence("sqDiff") == techniques_for_feature("sqDiff")
    # Descritores e desconhecidas caem no mesmo lugar: não sustentam nada.
    assert techniques_for_evidence("model_name") == ()
    assert techniques_for_evidence("featureNova") == ()
