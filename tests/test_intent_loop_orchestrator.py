"""Testes do Orchestrator v2 (IntentLoopOrchestrator, épico E3/E4).

Roda o pipeline INTENT->GENERATOR->ERENO->PREPROCESS->DETECTOR ponta a ponta
com um stub de ``IntentLike`` e um seed cacheado minúsculo — sem chave da
Groq e sem Java, como os testes do loop legado (``test_orchestrator_workflow.py``).
Cobre a DoD do E3 ("estados, falhas e artefatos persistidos por estágio") e
do E4 ("nenhuma avaliação ocorre sem DatasetBundle aprovado").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.agents.orchestrator.intent_loop import (
    IntentLike,
    IntentLoopOrchestrator,
)
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentObjective,
    IntentSpec,
)
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus
from adversarial_ids.shared.loop_record_store import load_loop_records

ATTACK_LABEL = "masquerade_fake_fault"


def _tiny_seed(path: Path, *, attack_rows: int = 10, normal_rows: int = 10) -> Path:
    """Espelha ``_tiny_labeled_seed`` de ``test_orchestrator_workflow.py``."""

    rows = ["f1,f2,class"]
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


def _orchestrator(
    tmp_path: Path,
    intent_agent: IntentLike,
    *,
    cached_dataset_path: Path | None = None,
    min_attack_rows: int = 5,
    min_normal_rows: int = 5,
) -> IntentLoopOrchestrator:
    return IntentLoopOrchestrator(
        intent_agent=intent_agent,
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


def test_orchestrator_rejects_unknown_generator_mode():
    with pytest.raises(ValueError, match="generator_mode inválido"):
        IntentLoopOrchestrator(intent_agent=_StubIntentAgent(_intent()), generator_mode="bogus")


# --------------------------------------------------------------------------- #
# Caminho feliz — ponta a ponta                                               #
# --------------------------------------------------------------------------- #
def test_run_completes_all_five_implemented_stages_and_skips_the_rest(tmp_path):
    agent = _StubIntentAgent(_intent())
    orchestrator = _orchestrator(tmp_path, agent)

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
    for implemented in ("intent", "generator", "ereno", "preprocess", "detector"):
        assert statuses[implemented] == LoopStageStatus.SUCCEEDED
    assert statuses["defender"] == LoopStageStatus.SKIPPED
    assert statuses["feedback"] == LoopStageStatus.SKIPPED

    # Estágios out-of-scope (E5/E10) nunca fingem ter rodado: sempre trazem o motivo.
    defender = next(s for s in record.stages if s.name == "defender")
    assert "E5" in (defender.error or "")


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
    ):
        assert (run_dir / filename).exists(), filename

    stage_by_name = {stage.name: stage for stage in record.stages}
    assert stage_by_name["intent"].artifact_ref == str(run_dir / "intent.json")
    assert stage_by_name["preprocess"].artifact_ref == str(run_dir / "dataset_bundle.json")
    # O estágio ERENO referencia o trace CSV gerado, não um JSON.
    assert stage_by_name["ereno"].artifact_ref.endswith(".csv")


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
