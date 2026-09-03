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
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentObjective,
    IntentSpec,
)
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus
from adversarial_ids.shared.json_io import load_json
from adversarial_ids.shared.loop_record_store import load_loop_records

ATTACK_LABEL = "masquerade_fake_fault"


def _tiny_seed(path: Path, *, attack_rows: int = 10, normal_rows: int = 10) -> Path:
    """Espelha ``_tiny_labeled_seed`` de ``test_orchestrator_workflow.py``.

    Colunas nomeadas ``feat1``/``feat2`` (não ``f1``/``f2``) de propósito: o
    portão do Defensor (E5) reserva a chave ``f1`` para o F1-score do
    ``DetectionReport`` — um nome de feature real do ERENO nunca colide (são
    CamelCase, ex. ``TrapAreaSum``), mas o fixture sintético colidiria.
    """

    rows = ["feat1,feat2,class"]
    for i in range(normal_rows):
        rows.append(f"{i % 5},{i % 3},normal")
    for i in range(attack_rows):
        rows.append(f"{100 + i % 5},{10 + i % 3},{ATTACK_LABEL}")
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

    def defend(
        self, report: DetectionReport, *, report_ref: str | None = None
    ) -> DefensePlan:
        self.received_reports.append(report)
        self.received_refs.append(report_ref)
        return DefensePlan.model_validate(
            {
                "priority": priority_from_report(report),
                "detection_actions": [
                    {
                        "description": "Revisar o limiar de decisão do classificador.",
                        "evidence": [
                            {
                                "metric_or_feature": "recall",
                                "value": report.recall,
                                "detection_report_ref": report_ref,
                            }
                        ],
                        "validation_method": "Reavaliar o recall no mesmo split de teste.",
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
        self, report: DetectionReport, *, report_ref: str | None = None
    ) -> DefensePlan:
        plan = DefensePlan.model_validate(
            {
                "priority": priority_from_report(report),
                "detection_actions": [
                    {
                        "description": "Ação sem lastro no relatório.",
                        "evidence": [
                            {
                                "metric_or_feature": "feature_inventada",
                                "value": 0.99,
                                "detection_report_ref": None,
                            }
                        ],
                        "validation_method": "Não verificável.",
                    }
                ],
            }
        )
        return validate_plan_against_report(
            plan, report, expected_report_ref=report_ref
        )


class _FailingDefenderAgent:
    def defend(
        self, report: DetectionReport, *, report_ref: str | None = None
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
def test_run_completes_all_six_implemented_stages_and_skips_feedback(tmp_path):
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
    ):
        assert statuses[implemented] == LoopStageStatus.SUCCEEDED
    assert statuses["feedback"] == LoopStageStatus.SKIPPED

    # FEEDBACK (E10) é o único estágio fora de escopo nesta entrega — nunca
    # finge ter rodado, sempre traz o motivo.
    feedback = next(s for s in record.stages if s.name == "feedback")
    assert "E10" in (feedback.error or "")

    # O Defender recebeu o DetectionReport real desta execução, não um stub vazio.
    assert len(defender.received_reports) == 1
    assert defender.received_reports[0].model_name == "random_forest"
    assert defender.received_refs[0] is not None
    assert defender.received_refs[0].endswith("detection_report.json")


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
        "defense_plan.json",
    ):
        assert (run_dir / filename).exists(), filename

    stage_by_name = {stage.name: stage for stage in record.stages}
    assert stage_by_name["intent"].artifact_ref == str(run_dir / "intent.json")
    assert stage_by_name["preprocess"].artifact_ref == str(run_dir / "dataset_bundle.json")
    assert stage_by_name["defender"].artifact_ref == str(run_dir / "defense_plan.json")
    # O estágio ERENO referencia o trace CSV gerado, não um JSON.
    assert stage_by_name["ereno"].artifact_ref.endswith(".csv")

    # O plano persistido é um DefensePlan válido e fundamentado no relatório.
    DefensePlan.model_validate(load_json(run_dir / "defense_plan.json"))


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
