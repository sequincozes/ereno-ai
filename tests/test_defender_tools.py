"""Testes das ferramentas determinísticas do Defensor (épico E5)."""

from __future__ import annotations

import pytest

from adversarial_ids.agents.defender.tools import (
    DefensePlanValidationError,
    legal_techniques_by_bucket,
    parse_defense_plan,
    priority_from_report,
    select_evidence_candidates,
    techniques_by_evidence,
    validate_plan_against_report,
)
from adversarial_ids.config.defense_techniques import techniques_for_evidence
from adversarial_ids.domain import (
    DEFENSE_BUCKETS,
    VALIDATION_METRICS,
    DetectionReport,
    techniques_for_bucket,
)


def build_report(
    *,
    f1: float = 0.65,
    recall: float = 0.60,
    accuracy: float = 0.75,
    precision: float = 0.70,
) -> DetectionReport:
    """Cria um DetectionReport mínimo para os testes."""

    return DetectionReport.model_validate(
        {
            "model_name": "random_forest",
            "split": "train_test_80_20_seed42",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": 12, "fp": 4, "fn": 8, "tn": 20},
            "top_features": [
                {"feature": "TrapAreaSum", "importance": 0.42},
                {"feature": "cbStatusDiff", "importance": 0.30},
                {"feature": "analogDelta", "importance": 0.18},
            ],
            "latency_ms": 3.5,
        }
    )


def build_plan(
    *,
    priority: str = "medium",
    metric_or_feature: str = "recall",
    value: float | int | str = 0.60,
    detection_report_ref: str | None = None,
    technique: str = "detector_threshold_tuning",
    validation_test: dict | None = None,
) -> dict:
    return {
        "priority": priority,
        "detection_actions": [
            {
                "description": "Revisar o limiar de decisão do classificador.",
                "technique": technique,
                "evidence": [
                    {
                        "metric_or_feature": metric_or_feature,
                        "value": value,
                        "detection_report_ref": detection_report_ref,
                    }
                ],
                "validation_test": validation_test
                or {
                    "metric": "recall",
                    "direction": "increase",
                    "target": 0.80,
                    "procedure": "Reavaliar recall no mesmo split.",
                },
            }
        ],
    }


# --------------------------------------------------------------------------- #
# priority_from_report                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("f1", "recall", "expected"),
    [
        (0.90, 0.90, "low"),
        (0.80, 0.90, "low"),
        (0.90, 0.40, "medium"),
        (0.65, 0.90, "medium"),
        (0.50, 0.90, "medium"),
        (0.65, 0.40, "high"),
        (0.49, 0.90, "high"),
        (0.30, 0.90, "high"),
        (0.30, 0.40, "critical"),
        (0.49, 0.49, "critical"),
        (0.50, 0.50, "medium"),  # limiar de recall não escalona em 0.50 exato
    ],
)
def test_priority_from_report(f1: float, recall: float, expected: str):
    report = build_report(f1=f1, recall=recall)
    assert priority_from_report(report) == expected


# --------------------------------------------------------------------------- #
# select_evidence_candidates                                                  #
# --------------------------------------------------------------------------- #
def test_select_evidence_candidates_includes_all_metric_keys():
    allowed = select_evidence_candidates(build_report())

    for key in (
        "model_name", "split", "accuracy", "precision", "recall", "f1",
        "latency_ms", "confusion_matrix.tp", "confusion_matrix.fp",
        "confusion_matrix.fn", "confusion_matrix.tn",
    ):
        assert key in allowed

    assert allowed["confusion_matrix.fn"] == 8
    assert isinstance(allowed["confusion_matrix.fn"], int)
    assert allowed["recall"] == 0.60


def test_select_evidence_candidates_includes_features():
    allowed = select_evidence_candidates(build_report())

    assert allowed["TrapAreaSum"] == 0.42
    assert allowed["cbStatusDiff"] == 0.30
    assert allowed["analogDelta"] == 0.18


def test_select_evidence_candidates_top_n_truncates_features_only():
    allowed = select_evidence_candidates(build_report(), top_n=1)

    assert "TrapAreaSum" in allowed
    assert "cbStatusDiff" not in allowed
    # As 11 chaves de métrica nunca são truncadas por top_n.
    assert "confusion_matrix.fn" in allowed
    assert allowed["recall"] == 0.60


def test_select_evidence_candidates_rejects_non_positive_top_n():
    with pytest.raises(ValueError, match="top_n deve ser maior que zero"):
        select_evidence_candidates(build_report(), top_n=0)


def test_select_evidence_candidates_rejects_feature_colliding_with_metric_key():
    report = build_report()
    collided = report.model_copy(
        update={
            "top_features": (
                *report.top_features,
                report.top_features[0].__class__(feature="recall", importance=0.1),
            )
        }
    )

    with pytest.raises(DefensePlanValidationError, match="colide"):
        select_evidence_candidates(collided)


# --------------------------------------------------------------------------- #
# parse_defense_plan                                                          #
# --------------------------------------------------------------------------- #
def test_parse_defense_plan_accepts_dict():
    plan = parse_defense_plan(build_plan())
    assert plan.priority == "medium"


def test_parse_defense_plan_accepts_json_string():
    import json

    plan = parse_defense_plan(json.dumps(build_plan()))
    assert plan.priority == "medium"


def test_parse_defense_plan_accepts_fenced_json_string():
    import json

    fenced = f"```json\n{json.dumps(build_plan())}\n```"
    plan = parse_defense_plan(fenced)
    assert plan.priority == "medium"


def test_parse_defense_plan_accepts_pydantic_model():
    from adversarial_ids.domain import DefensePlan

    model = DefensePlan.model_validate(build_plan())
    assert parse_defense_plan(model) is model


def test_parse_defense_plan_rejects_none():
    with pytest.raises(RuntimeError, match="não possui conteúdo"):
        parse_defense_plan(None)


def test_parse_defense_plan_rejects_invalid_json_string():
    with pytest.raises(ValueError, match="JSON inválido"):
        parse_defense_plan("não é json")


# --------------------------------------------------------------------------- #
# validate_plan_against_report                                                #
# --------------------------------------------------------------------------- #
def test_validate_accepts_grounded_plan():
    report = build_report()
    plan = build_plan(priority=priority_from_report(report))

    validated = validate_plan_against_report(plan, report)

    assert validated.priority == priority_from_report(report)
    assert validated.detection_actions[0].evidence[0].value == 0.60


def test_validate_accepts_integer_value_with_float_precision():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="confusion_matrix.fn",
        value=8.0,
    )

    validated = validate_plan_against_report(plan, report)
    assert validated.detection_actions[0].evidence[0].value == 8.0


def test_validate_rejects_invented_evidence_key():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="feature_inventada",
        value=0.5,
    )

    with pytest.raises(DefensePlanValidationError, match="Evidência inventada"):
        validate_plan_against_report(plan, report)


def test_validate_rejects_altered_float_value():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="recall",
        value=0.99,
    )

    with pytest.raises(DefensePlanValidationError, match="Valor de evidência incorreto"):
        validate_plan_against_report(plan, report)


def test_validate_rejects_altered_int_value():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="confusion_matrix.tp",
        value=999,
    )

    with pytest.raises(DefensePlanValidationError, match="Valor de evidência incorreto"):
        validate_plan_against_report(plan, report)


def test_validate_rejects_bool_cited_for_int_evidence():
    report = build_report()
    # tp == 12, então True (== 1) não deve validar mesmo comparando "igual a 1".
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="confusion_matrix.tp",
        value=True,
    )

    with pytest.raises(DefensePlanValidationError, match="Valor de evidência incorreto"):
        validate_plan_against_report(plan, report)


def test_validate_rejects_string_cited_for_float_evidence():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        metric_or_feature="recall",
        value="0.60",
    )

    with pytest.raises(DefensePlanValidationError, match="Valor de evidência incorreto"):
        validate_plan_against_report(plan, report)


def test_validate_rejects_wrong_priority():
    report = build_report()
    with pytest.raises(DefensePlanValidationError, match="Prioridade incompatível"):
        validate_plan_against_report(build_plan(priority="critical"), report)


def test_validate_rejects_duplicate_evidence_in_same_action():
    report = build_report()
    plan = build_plan(priority=priority_from_report(report))
    plan["detection_actions"][0]["evidence"].append(
        {"metric_or_feature": "recall", "value": 0.60, "detection_report_ref": None}
    )

    with pytest.raises(DefensePlanValidationError, match="Evidência duplicada"):
        validate_plan_against_report(plan, report)


@pytest.mark.parametrize(
    "bucket", ["detection_actions", "containment_actions", "hardening_actions"]
)
def test_validate_gates_all_three_buckets(bucket: str):
    report = build_report()
    plan = {
        "priority": priority_from_report(report),
        bucket: [
            {
                "description": "Ação de teste.",
                # Nenhuma técnica é legal nos três baldes, então a do balde sob
                # teste vem do mesmo mapa que o contrato usa para recusar.
                "technique": techniques_for_bucket(bucket)[0],
                "evidence": [
                    {
                        "metric_or_feature": "feature_inventada",
                        "value": 1,
                        "detection_report_ref": None,
                    }
                ],
                "validation_test": {
                    "metric": "recall",
                    "direction": "increase",
                    "target": 0.80,
                    "procedure": "Reavaliar.",
                },
            }
        ],
    }

    with pytest.raises(DefensePlanValidationError, match="Evidência inventada"):
        validate_plan_against_report(plan, report)


def test_validate_accepts_matching_detection_report_ref():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        detection_report_ref="outputs/intent_loop/run-1/detection_report.json",
    )

    validated = validate_plan_against_report(
        plan,
        report,
        expected_report_ref="outputs/intent_loop/run-1/detection_report.json",
    )
    assert (
        validated.detection_actions[0].evidence[0].detection_report_ref
        == "outputs/intent_loop/run-1/detection_report.json"
    )


def test_validate_accepts_null_detection_report_ref_regardless_of_expected():
    report = build_report()
    plan = build_plan(priority=priority_from_report(report), detection_report_ref=None)

    validated = validate_plan_against_report(
        plan, report, expected_report_ref="outputs/intent_loop/run-1/detection_report.json"
    )
    assert validated.detection_actions[0].evidence[0].detection_report_ref is None


def test_validate_rejects_mismatched_detection_report_ref():
    report = build_report()
    plan = build_plan(
        priority=priority_from_report(report),
        detection_report_ref="outputs/intent_loop/run-OUTRO/detection_report.json",
    )

    with pytest.raises(DefensePlanValidationError, match="Referência de DetectionReport inválida"):
        validate_plan_against_report(
            plan,
            report,
            expected_report_ref="outputs/intent_loop/run-1/detection_report.json",
        )


# --------------------------------------------------------------------------- #
# Sincronia entre o contrato e a allowlist                                    #
# --------------------------------------------------------------------------- #
def test_every_validation_metric_exists_in_the_evidence_allowlist():
    """``ValidationMetric`` é um subconjunto de ``select_evidence_candidates``.

    As duas listas nascem em módulos diferentes (o contrato em ``domain``, a
    allowlist aqui) porque o domínio não importa nada de ``agents``. Se um dia
    o ``DetectionReport`` renomear uma métrica e só um dos lados acompanhar, um
    plano poderia prometer remedir algo que o relatório não publica — este
    teste é a costura entre os dois, no mesmo espírito da checagem de registro
    de ``core/detectors.py`` contra ``DETECTOR_KEYS``.
    """

    allowlist = select_evidence_candidates(build_report())

    assert set(VALIDATION_METRICS) <= set(allowlist)


def test_the_allowlist_keys_left_out_of_validation_metrics_are_not_measurable():
    """O que sobra da allowlist é texto ou importância — nada que se "remeça"."""

    report = build_report()
    allowlist = select_evidence_candidates(report)
    features = {feature.feature for feature in report.top_features}

    left_out = set(allowlist) - set(VALIDATION_METRICS)

    assert left_out == {"model_name", "split"} | features


# --------------------------------------------------------------------------- #
# Teste de validação: o alvo precisa cobrar melhora                           #
# --------------------------------------------------------------------------- #
def _with_test(**test: object) -> dict:
    base: dict[str, object] = {
        "metric": "recall",
        "direction": "increase",
        "target": 0.80,
        "procedure": "Remedir.",
    }
    base.update(test)
    report = build_report()
    return build_plan(priority=priority_from_report(report), validation_test=base)


def test_validate_accepts_an_increase_target_above_the_current_value():
    report = build_report()  # recall = 0.60

    validated = validate_plan_against_report(_with_test(target=0.80), report)

    assert validated.detection_actions[0].validation_test.target == 0.80


def test_validate_rejects_an_increase_target_below_the_current_value():
    """"Subir o recall para 0.50" quando ele já é 0.60 é uma regressão."""

    report = build_report()

    with pytest.raises(DefensePlanValidationError, match="não cobra melhora"):
        validate_plan_against_report(_with_test(target=0.50), report)


def test_validate_rejects_an_increase_target_equal_to_the_current_value():
    """Prometer o valor que já se tem não é um teste, é uma tautologia."""

    report = build_report()

    with pytest.raises(DefensePlanValidationError, match="não cobra melhora"):
        validate_plan_against_report(_with_test(target=0.60), report)


def test_validate_accepts_a_decrease_target_below_the_current_value():
    report = build_report()  # confusion_matrix.fn = 8

    plan = _with_test(metric="confusion_matrix.fn", direction="decrease", target=4.0)

    assert validate_plan_against_report(plan, report).priority


def test_validate_rejects_a_decrease_target_above_the_current_value():
    report = build_report()

    plan = _with_test(metric="confusion_matrix.fn", direction="decrease", target=12.0)

    with pytest.raises(DefensePlanValidationError, match="não cobra melhora"):
        validate_plan_against_report(plan, report)


@pytest.mark.parametrize("direction", ["at_least", "at_most"])
def test_floors_and_ceilings_may_already_be_satisfied(direction: str):
    """Uma ação pode existir para *proteger* uma métrica que já está boa.

    ``at_least``/``at_most`` são guardas contra regressão enquanto outra
    métrica é atacada, então não se comparam contra o valor medido hoje.
    """

    report = build_report()  # recall = 0.60
    plan = _with_test(direction=direction, target=0.60)

    assert validate_plan_against_report(plan, report).priority


# --------------------------------------------------------------------------- #
# legal_techniques_by_bucket                                                  #
# --------------------------------------------------------------------------- #
def test_legal_techniques_covers_the_three_buckets():
    legal = legal_techniques_by_bucket()

    assert set(legal) == set(DEFENSE_BUCKETS)
    assert all(techniques for techniques in legal.values())


def test_legal_techniques_matches_what_the_contract_accepts():
    """O que o prompt oferece é literalmente o que o contrato deixa passar."""

    legal = legal_techniques_by_bucket()

    for bucket, techniques in legal.items():
        assert techniques == techniques_for_bucket(bucket)


# --------------------------------------------------------------------------- #
# Portão de regras (E5) — a técnica precisa responder à evidência citada      #
# --------------------------------------------------------------------------- #
def test_validate_rejects_a_technique_that_answers_none_of_its_evidence():
    """Todo o resto do plano está correto; só a recomendação não tem lastro.

    É o caso que passava inteiro antes da avaliação por regras: `recall` é
    evidência real, o valor bate, `goose_authentication` é técnica legítima de
    hardening e o teste cobra melhora. Nada disso torna autenticar publisher uma
    resposta a um recall baixo.
    """

    report = build_report()
    plan = build_plan()
    plan["hardening_actions"] = plan.pop("detection_actions")
    plan["hardening_actions"][0]["technique"] = "goose_authentication"

    with pytest.raises(DefensePlanValidationError, match="sem lastro"):
        validate_plan_against_report(plan, report)


def test_rejection_message_carries_the_grounded_fraction():
    """A recusa precisa dizer *quanto* do plano se sustentava, não só que caiu.

    ``grounded_actions``/``total_actions`` é o número que a janela D47-54 cobra,
    e a exceção costuma ser a única coisa que sobra num ``LoopStage`` falho.
    """

    report = build_report()
    plan = build_plan()
    plan["hardening_actions"] = plan.pop("detection_actions")
    plan["hardening_actions"][0]["technique"] = "goose_authentication"

    with pytest.raises(DefensePlanValidationError, match=r"0/1 ações sustentadas"):
        validate_plan_against_report(plan, report)


def test_validate_accepts_the_technique_the_catalogue_recommends():
    report = build_report()

    accepted = validate_plan_against_report(build_plan(), report)

    assert accepted.detection_actions[0].technique == "detector_threshold_tuning"


def test_playbook_findings_never_turn_into_a_rejection():
    """``attack_key`` só liga conselhos: passá-lo não pode mudar o veredito."""

    report = build_report()
    plan = build_plan()

    without = validate_plan_against_report(plan, report)
    with_attack = validate_plan_against_report(
        plan, report, attack_key="masquerade_fault"
    )

    assert without == with_attack


# --------------------------------------------------------------------------- #
# techniques_by_evidence                                                      #
# --------------------------------------------------------------------------- #
def test_techniques_by_evidence_only_offers_what_the_gate_accepts():
    """O que o prompt oferece por evidência é o que o portão exige dela."""

    report = build_report()

    offered = techniques_by_evidence(report)

    for key, techniques in offered.items():
        assert techniques == techniques_for_evidence(key)
        assert techniques


def test_techniques_by_evidence_omits_the_descriptive_keys():
    """Mostrar ``model_name`` com lista vazia convidaria a citá-lo sozinho."""

    offered = techniques_by_evidence(build_report())

    assert "model_name" not in offered
    assert "split" not in offered
    assert "recall" in offered


def test_techniques_by_evidence_honours_the_same_top_n_as_the_evidence():
    """Oferecer técnica ancorada numa feature que o Defensor não pode citar
    seria oferecer uma escolha que o portão de evidência recusa."""

    report = build_report()

    citable = select_evidence_candidates(report, top_n=1)
    offered = techniques_by_evidence(report, top_n=1)

    assert set(offered) <= set(citable)
    assert "cbStatusDiff" not in offered
