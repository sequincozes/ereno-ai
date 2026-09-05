"""Testes da costura do loop real (Fase 2, integração).

Cobrem a fronteira nova sem exigir GROQ_API_KEY: normalização de saídas no
workflow, o adaptador do Estrategista real, a construção do gerador por modo e a
seleção de engine em ``create_default_runner``.
"""

from __future__ import annotations

from typing import Any

import pytest

from adversarial_ids.agents.orchestrator.live import (
    StrategistAdapter,
    _build_generator,
    run_live_workflow,
)
from adversarial_ids.agents.orchestrator.workflow import (
    AnalystLike,
    StrategistLike,
    _to_output_dict,
)
from adversarial_ids.config.attacks_registry import DEFAULT_ATTACK_KEY, get_attack_spec
from adversarial_ids.config.settings import BASELINE_DATASET_PATH
from adversarial_ids.domain.strategist_output import Change, StrategistOutput
from adversarial_ids.interfaces.experiment_runner import (
    CachedHistoryRunner,
    WorkflowAdapter,
    create_default_runner,
)


# --------------------------------------------------------------------------- #
# _to_output_dict — aceita dict OU modelo pydantic OU None                     #
# --------------------------------------------------------------------------- #
def test_to_output_dict_accepts_pydantic_dict_and_none():
    model = StrategistOutput(
        reasoning="x", persona="conservative", changes=[Change(field="fault.prob", value=0.4)]
    )

    assert _to_output_dict(model)["changes"] == [{"field": "fault.prob", "value": 0.4}]
    assert _to_output_dict({"changes": []}) == {"changes": []}
    assert _to_output_dict(None) == {}


def test_to_output_dict_rejects_unexpected_type():
    with pytest.raises(TypeError):
        _to_output_dict(["not", "a", "mapping"])


# --------------------------------------------------------------------------- #
# StrategistAdapter — faz o agente real satisfazer StrategistLike              #
# --------------------------------------------------------------------------- #
class _RecordingStrategistAgent:
    """Fake do StrategistAgent: registra os kwargs recebidos por suggest_changes."""

    def __init__(self) -> None:
        self.received: dict[str, Any] = {}

    def suggest_changes(self, **kwargs: Any) -> StrategistOutput:
        self.received = kwargs
        return StrategistOutput(reasoning="ok", persona="aggressive", changes=[])


def test_strategist_adapter_maps_contract_and_is_protocol_compatible():
    agent = _RecordingStrategistAgent()
    adapter = StrategistAdapter(agent=agent, base_prompt="BASE")

    assert isinstance(adapter, StrategistLike)

    out = adapter.propose(
        1,
        attack_config={"fault": {"prob": 0.6}},
        metrics={"f1_score_attack": 0.9},
        history=[{"iteration": 0}],
    )

    assert isinstance(out, StrategistOutput)
    assert agent.received["base_prompt"] == "BASE"
    assert agent.received["attack_json"] == {"fault": {"prob": 0.6}}
    assert agent.received["performance_results"] == {"f1_score_attack": 0.9}
    assert agent.received["history"] == [{"iteration": 0}]


# --------------------------------------------------------------------------- #
# _build_generator — modo cacheado usa a seed; modo inválido falha            #
# --------------------------------------------------------------------------- #
_SPEC = get_attack_spec(DEFAULT_ATTACK_KEY)


def test_build_generator_cached_uses_seed_dataset():
    generator = _build_generator("cached", _SPEC)
    assert generator.is_cached is True
    assert generator.cached_dataset_path == BASELINE_DATASET_PATH


def test_build_generator_jar_has_no_cache():
    generator = _build_generator("jar", _SPEC)
    assert generator.is_cached is False
    assert generator.segment_name == _SPEC.segment_name


def test_build_generator_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _build_generator("bogus", _SPEC)


# --------------------------------------------------------------------------- #
# create_default_runner — demo é o default; live liga o loop real            #
# --------------------------------------------------------------------------- #
def test_create_default_runner_demo_is_cached_history():
    assert isinstance(create_default_runner(), CachedHistoryRunner)
    assert isinstance(create_default_runner("demo"), CachedHistoryRunner)


def test_create_default_runner_live_defaults_to_team():
    runner = create_default_runner("live")
    assert isinstance(runner, WorkflowAdapter)
    # run_workflow é um partial(run_live_workflow, use_team=True, persona=...,
    # detector=...) por default. O detector (E8) entra aqui em vez de no
    # protocolo ExperimentRunner.run pelo mesmo motivo de persona/orchestration:
    # é uma escolha de montagem do loop, não um parâmetro de cada execução.
    assert runner.run_workflow.func is run_live_workflow
    assert runner.run_workflow.keywords == {
        "use_team": True,
        "persona": "conservative",
        "detector": "random_forest",
    }


def test_create_default_runner_live_direct_disables_team():
    runner = create_default_runner("live", orchestration="direct")
    assert isinstance(runner, WorkflowAdapter)
    assert runner.run_workflow.keywords == {
        "use_team": False,
        "persona": "conservative",
        "detector": "random_forest",
    }


def test_create_default_runner_live_forwards_persona():
    runner = create_default_runner("live", persona="aggressive")
    assert runner.run_workflow.keywords == {
        "use_team": True,
        "persona": "aggressive",
        "detector": "random_forest",
    }


def test_create_default_runner_rejects_unknown_persona():
    with pytest.raises(ValueError):
        create_default_runner("live", persona="bogus")


def test_create_default_runner_rejects_unknown_engine():
    with pytest.raises(ValueError):
        create_default_runner("bogus")


def test_create_default_runner_rejects_unknown_orchestration():
    with pytest.raises(ValueError):
        create_default_runner("live", orchestration="bogus")


# --------------------------------------------------------------------------- #
# Detector plugável (épico E8)                                                 #
# --------------------------------------------------------------------------- #
def test_create_default_runner_live_forwards_the_detector():
    runner = create_default_runner("live", detector="svm_rbf")
    assert runner.run_workflow.keywords["detector"] == "svm_rbf"


def test_create_default_runner_rejects_an_unknown_detector():
    from adversarial_ids.core.detectors import DetectorError

    with pytest.raises(DetectorError, match="Detector desconhecido"):
        create_default_runner("live", detector="xgboost")


def test_create_default_runner_demo_rejects_a_detector_choice():
    # No caminho demo o histórico golden já está gravado: nenhum detector é
    # treinado, então aceitar a escolha em silêncio seria mentir sobre o que
    # a execução faz.
    with pytest.raises(ValueError, match="não se aplica ao engine 'demo'"):
        create_default_runner("demo", detector="svm_linear")
