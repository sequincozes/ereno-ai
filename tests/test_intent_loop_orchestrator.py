"""Testes do Orchestrator v2 (IntentLoopOrchestrator, épico E3/E4/E5).

Roda o pipeline INTENT->GENERATOR->ERENO->PREPROCESS->DETECTOR->DEFENDER
ponta a ponta com stubs de ``IntentLike``/``DefenderLike`` e um seed cacheado
minúsculo — sem chave da Groq e sem Java, como os testes do loop legado
(``test_orchestrator_workflow.py``). Cobre a DoD do E3 ("estados, falhas e
artefatos persistidos por estágio"), do E4 ("nenhuma avaliação ocorre sem
DatasetBundle aprovado") e do E5 ("toda ação referencia métrica/feature e
método de verificação").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.agents.defender.tools import (
    priority_from_report,
    validate_plan_against_report,
)
from adversarial_ids.agents.orchestrator.intent_loop import (
    DefenderLike,
    IntentLike,
    IntentLoopOrchestrator,
)
from adversarial_ids.config.attack_capabilities import get_attack_capability
from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.defense_rule_report import DefenseRuleReport
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentObjective,
    IntentSpec,
)
from adversarial_ids.domain.feedback_decision import FeedbackDecision
from adversarial_ids.core.detectors import DetectorError
from adversarial_ids.domain.detector_manifest import DetectorManifest
from adversarial_ids.domain.feature_manifest import FeatureManifest
from adversarial_ids.domain.selection_manifest import SelectionManifest
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus
from adversarial_ids.shared.loop_event_store import (
    MemoryEventSink,
    load_loop_events,
    stage_timeline,
)
from adversarial_ids.shared.json_io import load_json
from adversarial_ids.shared.loop_record_store import load_loop_records

ATTACK_LABEL = "masquerade_fake_fault"


def _tiny_seed(
    path: Path,
    *,
    attack_rows: int = 10,
    normal_rows: int = 10,
    attack_label: str = ATTACK_LABEL,
) -> Path:
    """Espelha ``_tiny_labeled_seed`` de ``test_orchestrator_workflow.py``.

    Colunas nomeadas ``feat1``/``feat2`` (não ``f1``/``f2``) de propósito: o
    portão do Defensor (E5) reserva a chave ``f1`` para o F1-score do
    ``DetectionReport`` — um nome de feature real do ERENO nunca colide (são
    CamelCase, ex. ``TrapAreaSum``), mas o fixture sintético colidiria.

    ``attack_label`` é parametrizável para que o mesmo helper sirva qualquer
    ataque registrado — o gate E4 (``build_dataset_bundle``) exige que o
    rótulo do trace bata com ``AttackSpec.label`` do ataque da rodada.
    """

    rows = ["feat1,feat2,class"]
    for i in range(normal_rows):
        rows.append(f"{i % 5},{i % 3},normal")
    for i in range(attack_rows):
        rows.append(f"{100 + i % 5},{10 + i % 3},{attack_label}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _intent(seed: int = 7) -> IntentSpec:
    return IntentSpec.model_validate(
        {
            "source_prompt": "Reduza o recall variando a temporização da falha.",
            "objective": IntentObjective.EVADE_DETECTION,
            "base_attack": "masquerade_fault",
            "desired_effect": DesiredEffect.LOWER_RECALL,
            "seed": seed,
        }
    )


class _StubIntentAgent:
    """``IntentLike`` determinístico: devolve sempre a mesma intenção."""

    def __init__(self, intent: IntentSpec) -> None:
        self._intent = intent
        self.received_prompts: list[str] = []

    def interpret(self, prompt: str) -> IntentSpec:
        self.received_prompts.append(prompt)
        return self._intent


class _FailingIntentAgent:
    def interpret(self, prompt: str) -> IntentSpec:
        raise RuntimeError("a LLM não chamou submit_intent_spec")


class _StubDefenderAgent:
    """``DefenderLike`` determinístico: deriva o plano do relatório recebido.

    Não hardcoda nenhum valor de evidência — cita ``recall`` e a
    ``priority`` calculada a partir do próprio relatório recebido, o que
    exercita o portão real (``validate_plan_against_report``) de ponta a
    ponta em vez de contorná-lo com um plano fixo.
    """

    def __init__(self) -> None:
        self.received_reports: list[DetectionReport] = []
        self.received_refs: list[str | None] = []
        self.received_attack_keys: list[str | None] = []

    def defend(
        self,
        report: DetectionReport,
        *,
        report_ref: str | None = None,
        attack_key: str | None = None,
    ) -> DefensePlan:
        self.received_reports.append(report)
        self.received_refs.append(report_ref)
        self.received_attack_keys.append(attack_key)
        return DefensePlan.model_validate(
            {
                "priority": priority_from_report(report),
                "detection_actions": [
                    {
                        "description": "Revisar o limiar de decisão do classificador.",
                        "technique": "detector_threshold_tuning",
                        "evidence": [
                            {
                                "metric_or_feature": "recall",
                                "value": report.recall,
                                "detection_report_ref": report_ref,
                            }
                        ],
                        # `at_least` e não `increase`: em modo cached o relatório
                        # costuma sair com recall 1.0, e um alvo de *aumento*
                        # sobre 1.0 é impossível — o portão recusaria o plano do
                        # stub por um motivo que não é o que estes testes medem.
                        "validation_test": {
                            "metric": "recall",
                            "direction": "at_least",
                            "target": report.recall,
                            "procedure": "Reavaliar o recall no mesmo split de teste.",
                        },
                    }
                ],
            }
        )


class _UngroundedDefenderAgent:
    """``DefenderLike`` que cita uma feature inventada.

    Roda o mesmo portão determinístico que o ``DefenderAgent`` real usa
    (``validate_plan_against_report``) — a validação vive no agente, não no
    orquestrador (mesmo desenho de ``compile_intent`` dentro de
    ``IntentAgent.interpret``), então este stub precisa chamá-la para
    simular uma implementação real cujo LLM alucinou a evidência.
    """

    def defend(
        self,
        report: DetectionReport,
        *,
        report_ref: str | None = None,
        attack_key: str | None = None,
    ) -> DefensePlan:
        plan = DefensePlan.model_validate(
            {
                "priority": priority_from_report(report),
                "detection_actions": [
                    {
                        "description": "Ação sem lastro no relatório.",
                        "technique": "detector_threshold_tuning",
                        "evidence": [
                            {
                                "metric_or_feature": "feature_inventada",
                                "value": 0.99,
                                "detection_report_ref": None,
                            }
                        ],
                        # Coerente de propósito: o que este stub testa é a
                        # evidência inventada, então o teste de validação não
                        # pode ser o primeiro a falhar. Ver o comentário do
                        # ``_StubDefenderAgent`` sobre `at_least`.
                        "validation_test": {
                            "metric": "recall",
                            "direction": "at_least",
                            "target": report.recall,
                            "procedure": "Não verificável.",
                        },
                    }
                ],
            }
        )
        return validate_plan_against_report(
            plan, report, expected_report_ref=report_ref, attack_key=attack_key
        )


class _FailingDefenderAgent:
    def defend(
        self,
        report: DetectionReport,
        *,
        report_ref: str | None = None,
        attack_key: str | None = None,
    ) -> DefensePlan:
        raise RuntimeError("a LLM não retornou um DefensePlan")


def _orchestrator(
    tmp_path: Path,
    intent_agent: IntentLike,
    *,
    defender_agent: DefenderLike | None = None,
    cached_dataset_path: Path | None = None,
    min_attack_rows: int = 5,
    min_normal_rows: int = 5,
    feedback_min_delta: float = 0.01,
    detector: str = "random_forest",
    event_sink=None,
) -> IntentLoopOrchestrator:
    return IntentLoopOrchestrator(
        intent_agent=intent_agent,
        defender_agent=defender_agent or _StubDefenderAgent(),
        generator_mode="cached",
        cached_dataset_path=cached_dataset_path or _tiny_seed(tmp_path / "seed.csv"),
        output_dir=tmp_path / "artifacts",
        save_path=tmp_path / "loop_records.json",
        min_attack_rows=min_attack_rows,
        min_normal_rows=min_normal_rows,
        feedback_min_delta=feedback_min_delta,
        detector=detector,
        event_sink=event_sink,
    )


# --------------------------------------------------------------------------- #
# Construção                                                                  #
# --------------------------------------------------------------------------- #
def test_stub_intent_agent_satisfies_the_intentlike_protocol():
    assert isinstance(_StubIntentAgent(_intent()), IntentLike)


def test_stub_defender_agent_satisfies_the_defenderlike_protocol():
    assert isinstance(_StubDefenderAgent(), DefenderLike)


def test_orchestrator_rejects_unknown_generator_mode():
    with pytest.raises(ValueError, match="generator_mode inválido"):
        IntentLoopOrchestrator(
            intent_agent=_StubIntentAgent(_intent()),
            defender_agent=_StubDefenderAgent(),
            generator_mode="bogus",
        )


# --------------------------------------------------------------------------- #
# Caminho feliz — ponta a ponta                                               #
# --------------------------------------------------------------------------- #
def test_run_completes_all_seven_stages_including_feedback(tmp_path):
    agent = _StubIntentAgent(_intent())
    defender = _StubDefenderAgent()
    orchestrator = _orchestrator(tmp_path, agent, defender_agent=defender)

    record = orchestrator.run("Reduza o recall.")

    assert isinstance(record, LoopRecord)
    assert agent.received_prompts == ["Reduza o recall."]

    names = [stage.name for stage in record.stages]
    assert names == [
        "intent",
        "generator",
        "ereno",
        "preprocess",
        "detector",
        "defender",
        "feedback",
    ]

    statuses = {stage.name: stage.status for stage in record.stages}
    for implemented in (
        "intent", "generator", "ereno", "preprocess", "detector", "defender",
        "feedback",
    ):
        assert statuses[implemented] == LoopStageStatus.SUCCEEDED

    # Uma execução avulsa (run(), não run_campaign()) sempre tem max_rounds=1
    # — o FEEDBACK para na rodada 1 pelo teto de rodadas, honestamente.
    feedback = next(s for s in record.stages if s.name == "feedback")
    assert feedback.artifact_ref.endswith("feedback.json")

    # O Defender recebeu o DetectionReport real desta execução, não um stub vazio.
    assert len(defender.received_reports) == 1
    assert defender.received_reports[0].model_name == "random_forest"
    assert defender.received_refs[0] is not None
    assert defender.received_refs[0].endswith("detection_report.json")
    # O ataque-base resolvido da intenção, não um parâmetro externo separado:
    # é ele que escolhe o playbook IEC-61850 da avaliação por regras (E5).
    assert defender.received_attack_keys == ["masquerade_fault"]


@pytest.mark.parametrize("attack_key", list_attack_keys())
def test_every_catalogued_attack_runs_the_seven_stages_in_cached_mode(tmp_path, attack_key):
    """Prova que os 10 ataques novos rodam o pipeline ponta a ponta sem Java
    nem Groq — mesmo que, na prática, só sejam fisicamente mensuráveis em
    ``--generator-mode jar`` (ver README, limitação conhecida do modo cached)."""
    spec = get_attack_spec(attack_key)
    capability = get_attack_capability(attack_key)
    field = capability.fields[0]
    effect = next(iter(field.effects))

    intent = IntentSpec.model_validate(
        {
            "source_prompt": f"teste ponta a ponta: {attack_key}",
            "objective": IntentObjective.EVADE_DETECTION.value
            if effect != DesiredEffect.INCREASE_ATTACK_ACTIVITY
            else IntentObjective.ASSESS_IDS_ROBUSTNESS.value,
            "base_attack": attack_key,
            "desired_effect": effect.value,
            "restrictions": {"allowed_fields": [field.path], "max_fields_changed": 1},
        }
    )
    seed = _tiny_seed(tmp_path / "seed.csv", attack_label=spec.label)
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(intent), cached_dataset_path=seed)

    record = orchestrator.run(f"teste ponta a ponta: {attack_key}")

    statuses = {stage.name: stage.status for stage in record.stages}
    assert statuses == {
        name: LoopStageStatus.SUCCEEDED
        for name in (
            "intent", "generator", "ereno", "preprocess", "detector", "defender",
            "feedback",
        )
    }


def test_the_four_delayed_replay_variants_share_one_dataset_label():
    """Documenta em código o caveat do README: o gate E4 ancora a classe de
    ataque em ``AttackSpec.label``, e as 4 variantes uc10 compartilham o mesmo
    rótulo — uma campanha em ``delayed_replay_double_drop`` produz um
    ``DetectionReport`` indistinguível do ``delayed_replay`` base."""
    variants = [k for k in list_attack_keys() if k.startswith("delayed_replay")]
    labels = {get_attack_spec(k).label for k in variants}
    assert len(variants) == 4
    assert labels == {"delayed_replay"}


def test_run_uses_the_compiled_intents_seed_not_the_default_argument(tmp_path):
    agent = _StubIntentAgent(_intent(seed=999))
    orchestrator = _orchestrator(tmp_path, agent)

    record = orchestrator.run("prompt qualquer", seed=1)

    assert record.seed == 999


def test_run_persists_a_json_artifact_per_typed_stage(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    record = orchestrator.run("Reduza o recall.")

    run_dir = tmp_path / "artifacts" / record.run_id
    for filename in (
        "intent.json",
        "attack_candidate.json",
        "dataset_bundle.json",
        "detection_report.json",
        "feature_manifest.json",
        "selection_manifest.json",
        "detector_manifest.json",
        "defense_plan.json",
        "defense_rules.json",
        "feedback.json",
    ):
        assert (run_dir / filename).exists(), filename

    stage_by_name = {stage.name: stage for stage in record.stages}
    assert stage_by_name["intent"].artifact_ref == str(run_dir / "intent.json")
    assert stage_by_name["preprocess"].artifact_ref == str(run_dir / "dataset_bundle.json")
    assert stage_by_name["defender"].artifact_ref == str(run_dir / "defense_plan.json")
    assert stage_by_name["feedback"].artifact_ref == str(run_dir / "feedback.json")
    # O estágio ERENO referencia o trace CSV gerado, não um JSON.
    assert stage_by_name["ereno"].artifact_ref.endswith(".csv")

    # O plano persistido é um DefensePlan válido e fundamentado no relatório.
    DefensePlan.model_validate(load_json(run_dir / "defense_plan.json"))

    # A avaliação por regras (E5) fica ao lado do plano: é o artefato que
    # sustenta o gate da janela D47-54, e `grounded_actions`/`total_actions` é
    # literalmente a fração de recomendações ligadas a evidência.
    rules = DefenseRuleReport.model_validate(load_json(run_dir / "defense_rules.json"))
    assert rules.is_grounded
    assert rules.grounded_fraction == 1.0
    assert rules.detection_report_ref == str(run_dir / "detection_report.json")
    # O eixo do playbook só existe porque o ataque-base da intenção chegou até
    # aqui — sem ele o relatório não saberia de que cenário IEC-61850 falar.
    assert rules.attack_key == "masquerade_fault"
    assert rules.playbook_key == "spoofed_fault_indication"

    # A decisão persistida é um FeedbackDecision válido.
    FeedbackDecision.model_validate(load_json(run_dir / "feedback.json"))

    # O manifest (E6) é um FeatureManifest válido, e fitted_rows prova que o
    # preprocessador viu só a partição de TREINO do baseline (14 de 20 linhas
    # do seed cacheado, test_size=0.3 padrão do IdsEvaluator) — nunca as 20.
    manifest = FeatureManifest.model_validate(load_json(run_dir / "feature_manifest.json"))
    assert manifest.fitted_rows == 14

    # O manifest de seleção (E7) é um SelectionManifest válido; estratégias
    # default "none" não alteram nem colunas nem linhas de treino — as duas
    # contagens batem com o mesmo fitted_rows do FeatureManifest acima.
    selection = SelectionManifest.model_validate(load_json(run_dir / "selection_manifest.json"))
    assert selection.feature_selection == "none"
    assert selection.undersampling == "none"
    assert selection.selected_features == selection.candidate_features
    assert selection.fitted_rows_before_undersampling == 14
    assert selection.fitted_rows_after_undersampling == 14

    # O manifest do detector (E8) é um DetectorManifest válido e descreve o
    # detector default. trained_rows tem que bater com o que o E7 entregou ao
    # fit — é o elo que prova que os três manifests falam da mesma execução.
    detector = DetectorManifest.model_validate(load_json(run_dir / "detector_manifest.json"))
    assert detector.detector == "random_forest"
    assert detector.resolved_scaler == "none"
    assert detector.trained_rows == selection.fitted_rows_after_undersampling
    assert detector.trained_features == selection.selected_features


def test_run_appends_the_record_to_the_loop_record_store(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    record = orchestrator.run("Reduza o recall.")

    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert [r.run_id for r in persisted] == [record.run_id]


def test_two_runs_get_isolated_artifact_directories(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    first = orchestrator.run("prompt 1")
    second = orchestrator.run("prompt 2")

    assert first.run_id != second.run_id
    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert [r.run_id for r in persisted] == [first.run_id, second.run_id]


def test_a_single_run_is_recorded_as_round_one_with_no_parent(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    record = orchestrator.run("Reduza o recall.")

    assert record.round == 1
    assert record.parent_run_id is None


# --------------------------------------------------------------------------- #
# Campanha multi-rodada (run_campaign, E10)                                   #
# --------------------------------------------------------------------------- #
def test_run_campaign_rejects_a_non_positive_round_count(tmp_path):
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    with pytest.raises(ValueError, match="rounds"):
        orchestrator.run_campaign("Reduza o recall.", rounds=0)


def test_run_campaign_of_one_round_matches_a_plain_run(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=1)

    assert len(records) == 1
    assert records[0].round == 1
    assert agent.received_prompts == ["Reduza o recall."]


def test_run_campaign_stops_at_the_plateau_in_cached_mode(tmp_path):
    # Modo cacheado serve o mesmo dataset em toda rodada — a métrica não se
    # move, então a política para com "no_improvement" na rodada 2 mesmo
    # pedindo um teto de rodadas maior.
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=5)

    assert len(records) == 2
    first, second = records
    assert first.round == 1
    assert first.parent_run_id is None
    assert second.round == 2
    assert second.parent_run_id == first.run_id

    # A LLM só é chamada na primeira rodada — a segunda intenção nasce da
    # política de feedback, não de uma nova interpretação do prompt.
    assert agent.received_prompts == ["Reduza o recall."]

    # A rodada 2 herda a intenção sem chamar o IntentLike: o estágio intent
    # fica "skipped", nunca fabricado como se a LLM tivesse rodado de novo.
    intent_stage_round_2 = second.stages[0]
    assert intent_stage_round_2.name == "intent"
    assert intent_stage_round_2.status == LoopStageStatus.SKIPPED
    assert "herdada" in intent_stage_round_2.error
    assert intent_stage_round_2.artifact_ref.endswith("intent.json")


def test_run_campaign_runs_every_round_when_the_epsilon_allows_zero_improvement(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=3)

    assert [r.round for r in records] == [1, 2, 3]
    assert records[1].parent_run_id == records[0].run_id
    assert records[2].parent_run_id == records[1].run_id
    # A LLM ainda é chamada uma única vez pela campanha inteira.
    assert agent.received_prompts == ["Reduza o recall."]


def test_run_campaign_persists_one_loop_record_per_round(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=3)

    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert [r.run_id for r in persisted] == [r.run_id for r in records]


def test_run_campaign_keeps_the_same_source_prompt_across_rounds(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent, feedback_min_delta=0.0)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=3)

    assert {r.source_prompt for r in records} == {"Reduza o recall."}


def test_run_campaign_escalates_the_intensity_of_the_second_round(tmp_path):
    agent = _StubIntentAgent(_intent(seed=7))
    orchestrator = _orchestrator(tmp_path, agent)

    records = orchestrator.run_campaign("Reduza o recall.", rounds=5)

    run_dir = tmp_path / "artifacts" / records[1].run_id
    second_intent = load_json(run_dir / "intent.json")
    assert second_intent["intensity"] == "high"


def test_run_campaign_aborts_when_a_round_fails(tmp_path):
    orchestrator = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), defender_agent=_FailingDefenderAgent()
    )

    records = orchestrator.run_campaign("Reduza o recall.", rounds=3)

    assert len(records) == 1
    assert "feedback" not in [s.name for s in records[0].stages]
    assert records[0].stages[-1].status == LoopStageStatus.FAILED

    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert [r.run_id for r in persisted] == [records[0].run_id]


# --------------------------------------------------------------------------- #
# Falhas por estágio                                                          #
# --------------------------------------------------------------------------- #
def test_run_marks_intent_failed_and_stops_before_later_stages(tmp_path):
    orchestrator = _orchestrator(tmp_path, _FailingIntentAgent())

    record = orchestrator.run("prompt qualquer")

    assert [s.name for s in record.stages] == ["intent"]
    assert record.stages[0].status == LoopStageStatus.FAILED
    assert "submit_intent_spec" in record.stages[0].error

    # O registro (parcial) ainda é persistido — a causa não se perde.
    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert persisted[0].stages[0].status == LoopStageStatus.FAILED


def test_run_marks_preprocess_failed_when_the_trace_fails_the_gate(tmp_path):
    # Seed com só 1 linha de ataque — abaixo do piso (min_attack_rows=5).
    thin_seed = _tiny_seed(tmp_path / "thin_seed.csv", attack_rows=1, normal_rows=10)
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent, cached_dataset_path=thin_seed)

    record = orchestrator.run("Reduza o recall.")

    names = [s.name for s in record.stages]
    assert names == ["intent", "generator", "ereno", "preprocess"]
    assert record.stages[-1].status == LoopStageStatus.FAILED
    assert "abaixo do piso" in record.stages[-1].error


def test_run_never_raises_even_when_a_stage_fails(tmp_path):
    orchestrator = _orchestrator(tmp_path, _FailingIntentAgent())

    # Não deve levantar — a falha vira LoopStage(status=failed), não exceção.
    record = orchestrator.run("prompt qualquer")
    assert record.total_duration_seconds is not None


def test_run_marks_defender_failed_when_the_llm_call_raises(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent, defender_agent=_FailingDefenderAgent())

    record = orchestrator.run("Reduza o recall.")

    names = [s.name for s in record.stages]
    assert names == ["intent", "generator", "ereno", "preprocess", "detector", "defender"]
    assert record.stages[-1].status == LoopStageStatus.FAILED
    assert "DefensePlan" in record.stages[-1].error

    # O pipeline parou ali — feedback nunca chega a ser registrado.
    assert "feedback" not in names

    # O registro (parcial) ainda é persistido.
    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert persisted[0].stages[-1].status == LoopStageStatus.FAILED


def test_run_marks_defender_failed_when_the_plan_is_ungrounded(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(
        tmp_path, agent, defender_agent=_UngroundedDefenderAgent()
    )

    record = orchestrator.run("Reduza o recall.")

    names = [s.name for s in record.stages]
    assert names == ["intent", "generator", "ereno", "preprocess", "detector", "defender"]
    assert record.stages[-1].status == LoopStageStatus.FAILED
    assert "Evidência inventada" in record.stages[-1].error
    assert "feedback" not in names

    # run() nunca levanta, mesmo com o portão de evidência rejeitando o plano.
    assert record.total_duration_seconds is not None


def test_run_marks_feedback_failed_when_the_policy_raises(tmp_path, monkeypatch):
    def _boom(**_kwargs):
        raise RuntimeError("política quebrada")

    monkeypatch.setattr(
        "adversarial_ids.agents.orchestrator.intent_loop.decide_feedback", _boom
    )
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

    record = orchestrator.run("Reduza o recall.")

    names = [s.name for s in record.stages]
    assert names == [
        "intent", "generator", "ereno", "preprocess", "detector", "defender", "feedback",
    ]
    assert record.stages[-1].status == LoopStageStatus.FAILED
    assert "política quebrada" in record.stages[-1].error

    # run() nunca levanta mesmo quando a própria política falha.
    assert record.total_duration_seconds is not None


# --------------------------------------------------------------------------- #
# Detector plugável no estágio DETECTOR (épico E8)                             #
# --------------------------------------------------------------------------- #
def test_detector_choice_reaches_the_report_the_manifest_and_the_defender(tmp_path):
    """Sem isto o loop rodaria um SVM e continuaria anunciando 'random_forest'.

    Os três lugares onde o nome do detector aparece precisam concordar: o
    ``DetectionReport`` que o Defensor recebe, o ``detector_manifest.json``
    em disco, e o ``detection_report.json`` persistido.
    """

    defender = _StubDefenderAgent()
    orchestrator = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), defender_agent=defender, detector="decision_tree"
    )

    record = orchestrator.run("Reduza o recall.")
    run_dir = tmp_path / "artifacts" / record.run_id

    assert defender.received_reports[0].model_name == "decision_tree"
    assert load_json(run_dir / "detection_report.json")["model_name"] == "decision_tree"

    manifest = DetectorManifest.model_validate(load_json(run_dir / "detector_manifest.json"))
    assert manifest.detector == "decision_tree"
    assert manifest.model_name == "decision_tree"


def test_svm_detector_resolves_standard_scaling_through_the_whole_stage(tmp_path):
    # O E8 encosta no E6 exatamente aqui: a escala que o SVM pede tem que
    # chegar ao FeaturePreprocessor, e os dois manifests têm que concordar.
    orchestrator = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), detector="svm_linear"
    )

    record = orchestrator.run("Reduza o recall.")
    run_dir = tmp_path / "artifacts" / record.run_id

    detector = DetectorManifest.model_validate(load_json(run_dir / "detector_manifest.json"))
    feature = FeatureManifest.model_validate(load_json(run_dir / "feature_manifest.json"))

    assert detector.requires_scaling is True
    assert detector.resolved_scaler == "standard"
    assert feature.scaler == "standard"
    assert feature.scaler_stats  # estatísticas de fato ajustadas no treino


def test_unknown_detector_is_rejected_when_the_orchestrator_is_built(tmp_path):
    # Erro de configuração do chamador, não falha de estágio: tem que estourar
    # na construção, antes de qualquer geração de dataset.
    with pytest.raises(DetectorError, match="Detector desconhecido"):
        _orchestrator(tmp_path, _StubIntentAgent(_intent()), detector="xgboost")


# --------------------------------------------------------------------------- #
# Timeline observável (E11)                                                    #
# --------------------------------------------------------------------------- #
def test_run_emits_a_typed_timeline_from_the_first_stage_to_the_last(tmp_path):
    """A UI passa a ler evento, não a adivinhar fase por substring de print."""

    sink = MemoryEventSink()
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()), event_sink=sink)

    record = orchestrator.run("Reduza o recall.")

    events = sink.snapshot()
    assert [e.sequence for e in events] == list(range(len(events)))
    assert events[0].kind == "run_started"
    assert events[0].message == "Reduza o recall."
    assert events[-1].kind == "run_finished"
    assert events[-1].status is LoopStageStatus.SUCCEEDED
    assert all(e.run_id == record.run_id for e in events)

    # Cada estágio se anuncia ao começar e ao terminar, nessa ordem.
    for stage in ("intent", "generator", "ereno", "preprocess", "detector",
                  "defender", "feedback"):
        kinds = [e.kind for e in events if e.stage == stage]
        assert kinds == ["stage_started", "stage_finished"], stage


def test_the_timeline_and_the_record_never_disagree(tmp_path):
    """São o mesmo fato emitido duas vezes; divergir seria mentir numa das duas."""

    sink = MemoryEventSink()
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()), event_sink=sink)

    record = orchestrator.run("Reduza o recall.")

    timeline = stage_timeline(sink.snapshot())
    for stage in record.stages:
        finished = timeline[stage.name]
        assert finished.status is stage.status
        assert finished.artifact_ref == stage.artifact_ref
        assert finished.duration_seconds == stage.duration_seconds


def test_the_timeline_is_on_disk_without_anyone_asking(tmp_path):
    """Execução pela CLI também precisa deixar rastro de progresso."""

    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    record = orchestrator.run("Reduza o recall.")

    events = load_loop_events(tmp_path / "artifacts" / record.run_id / "events.jsonl")
    assert events
    assert events[-1].kind == "run_finished"


def test_a_failed_stage_names_itself_and_its_cause_in_the_timeline(tmp_path):
    """O critério de aceite da linha Operação: falha tem estágio e causa."""

    sink = MemoryEventSink()
    orchestrator = _orchestrator(
        tmp_path, _FailingIntentAgent(), event_sink=sink
    )

    orchestrator.run("Reduza o recall.")

    events = sink.snapshot()
    failed = [e for e in events if e.status is LoopStageStatus.FAILED]
    assert failed[0].stage == "intent"
    assert "submit_intent_spec" in (failed[0].message or "")
    # A execução ainda se fecha: quem observa distingue "parou" de "pensando".
    assert events[-1].kind == "run_finished"
    assert events[-1].status is LoopStageStatus.FAILED
    assert "intent" in (events[-1].message or "")


def test_each_campaign_round_carries_its_own_round_in_the_timeline(tmp_path):
    sink = MemoryEventSink()
    orchestrator = _orchestrator(
        tmp_path,
        _StubIntentAgent(_intent()),
        feedback_min_delta=0.0,
        event_sink=sink,
    )

    records = orchestrator.run_campaign("Reduza o recall.", rounds=2)

    rounds = {e.round for e in sink.snapshot()}
    assert rounds == {record.round for record in records}


def test_a_sink_that_explodes_does_not_break_the_run(tmp_path):
    def explode(_event):
        raise RuntimeError("observador quebrado")

    orchestrator = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), event_sink=explode
    )

    record = orchestrator.run("Reduza o recall.")

    assert [stage.status for stage in record.stages] == [
        LoopStageStatus.SUCCEEDED
    ] * 7
