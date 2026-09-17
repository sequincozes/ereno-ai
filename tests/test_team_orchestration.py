"""Testes do caminho de orquestração via ``agno.team.Team`` (route mode).

Exercitam a extração da saída estruturada dos membros a partir de um
``TeamRunOutput`` **falso** — sem Groq nem chave. A construção real do ``Team``
também é verificada (só instanciação, sem ``run``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from adversarial_ids.agents.orchestrator.live import (
    ANALYST_MEMBER,
    STRATEGIST_MEMBER,
    TeamAnalyst,
    TeamStrategist,
    _build_team,
    _member_response,
)
from adversarial_ids.agents.orchestrator.workflow import AnalystLike, StrategistLike
from adversarial_ids.domain.strategist_output import StrategistOutput


# --------------------------------------------------------------------------- #
# Dublês do TeamRunOutput / RunOutput de membro                                #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeTool:
    tool_name: str
    result: str


@dataclass
class _FakeMemberResponse:
    agent_name: str
    content: Any = None
    tools: list[_FakeTool] = field(default_factory=list)


@dataclass
class _FakeTeamOutput:
    member_responses: list[_FakeMemberResponse] = field(default_factory=list)
    content: Any = None
    tools: list[_FakeTool] = field(default_factory=list)


class _FakeTeam:
    """Registra a última entrada e devolve um ``TeamRunOutput`` pré-fabricado."""

    def __init__(self, output: _FakeTeamOutput) -> None:
        self._output = output
        self.last_input: str | None = None

    def run(self, input: str, **_: Any) -> _FakeTeamOutput:
        self.last_input = input
        return self._output


class _FakeStrategistAgent:
    def build_prompt(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "STRATEGIST_PROMPT"


class _FakeAnalystAgent:
    def build_prompt(self, *, iteration: int, metrics: Any) -> str:
        self.iteration = iteration
        return "ANALYST_PROMPT"


# --------------------------------------------------------------------------- #
# _member_response                                                            #
# --------------------------------------------------------------------------- #
def test_member_response_matches_by_agent_name():
    strat = _FakeMemberResponse(agent_name=STRATEGIST_MEMBER)
    analyst = _FakeMemberResponse(agent_name=ANALYST_MEMBER)
    output = _FakeTeamOutput(member_responses=[strat, analyst])

    assert _member_response(output, ANALYST_MEMBER) is analyst
    assert _member_response(output, STRATEGIST_MEMBER) is strat


def test_member_response_falls_back_to_team_output_when_empty():
    output = _FakeTeamOutput(member_responses=[])
    assert _member_response(output, STRATEGIST_MEMBER) is output


# --------------------------------------------------------------------------- #
# TeamStrategist — extrai o StrategistOutput da tool do membro                 #
# --------------------------------------------------------------------------- #
def test_team_strategist_extracts_tool_output_and_is_protocol_compatible():
    payload = StrategistOutput(
        reasoning="via team",
        persona="aggressive",
        changes=[{"field": "fault.prob", "value": 0.7}],
    ).model_dump_json()

    member = _FakeMemberResponse(
        agent_name=STRATEGIST_MEMBER,
        tools=[_FakeTool(tool_name="submit_strategist_output", result=payload)],
    )
    team = _FakeTeam(_FakeTeamOutput(member_responses=[member]))
    adapter = TeamStrategist(team, _FakeStrategistAgent(), base_prompt="BASE")

    assert isinstance(adapter, StrategistLike)

    out = adapter.propose(1, attack_config={"fault": {"prob": 0.6}}, metrics={}, history=[])

    assert isinstance(out, StrategistOutput)
    assert out.persona == "aggressive"
    assert out.changes[0].field == "fault.prob"
    assert out.changes[0].value == 0.7
    # a tarefa foi encaminhada com a diretiva de roteamento ao membro certo.
    assert STRATEGIST_MEMBER in (team.last_input or "")


def test_team_strategist_falls_back_to_content_without_tool():
    payload = StrategistOutput(
        reasoning="sem tool", persona="conservative", changes=[]
    ).model_dump_json()
    member = _FakeMemberResponse(agent_name=STRATEGIST_MEMBER, content=payload, tools=[])
    team = _FakeTeam(_FakeTeamOutput(member_responses=[member]))
    adapter = TeamStrategist(team, _FakeStrategistAgent(), base_prompt="BASE")

    out = adapter.propose(1, attack_config={}, metrics={}, history=[])

    assert isinstance(out, StrategistOutput)
    assert out.persona == "conservative"
    assert out.changes == []


# --------------------------------------------------------------------------- #
# TeamAnalyst — extrai/valida o AnalystOutput do membro                        #
# --------------------------------------------------------------------------- #
def test_team_analyst_extracts_and_validates_output():
    metrics = {
        "f1_score_attack": 0.9,  # severidade esperada: low
        "top_feature_importances": [{"feature": "stDiff", "importance": 0.5}],
    }
    analyst_json = json.dumps(
        {
            "iteration": 2,
            "deceptive_features": [
                {
                    "feature": "stDiff",
                    "importance": 0.5,
                    "explanation": "A feature stDiff dominou a decisão do RF.",
                }
            ],
            "diagnosis": "O ataque explorou a feature temporal stDiff para evadir.",
            "mitigations": [
                {"type": "threshold", "recommendation": "Recalibrar o limiar do RF."}
            ],
            "severity": "low",
        }
    )
    member = _FakeMemberResponse(agent_name=ANALYST_MEMBER, content=analyst_json)
    team = _FakeTeam(_FakeTeamOutput(member_responses=[member]))
    adapter = TeamAnalyst(team, _FakeAnalystAgent())

    assert isinstance(adapter, AnalystLike)

    out = adapter.analyze(2, metrics)

    assert out.iteration == 2
    assert out.severity == "low"
    assert out.deceptive_features[0].feature == "stDiff"
    assert ANALYST_MEMBER in (team.last_input or "")


def test_team_analyst_rejects_iteration_mismatch():
    analyst_json = json.dumps(
        {
            "iteration": 99,
            "deceptive_features": [],
            "diagnosis": "Diagnóstico com tamanho suficiente para validação.",
            "mitigations": [
                {"type": "retrain", "recommendation": "Retreinar o modelo base."}
            ],
            "severity": "low",
        }
    )
    member = _FakeMemberResponse(agent_name=ANALYST_MEMBER, content=analyst_json)
    team = _FakeTeam(_FakeTeamOutput(member_responses=[member]))
    adapter = TeamAnalyst(team, _FakeAnalystAgent())

    with pytest.raises(ValueError):
        adapter.analyze(2, {"f1_score_attack": 0.9, "top_feature_importances": []})


# --------------------------------------------------------------------------- #
# _build_team — constrói um Team real em route mode (sem run/chave)            #
# --------------------------------------------------------------------------- #
def test_build_team_constructs_route_team_with_named_members():
    from adversarial_ids.agents.analyst.agent import AnalystAgent
    from adversarial_ids.agents.strategist.agent import StrategistAgent

    strat = StrategistAgent(model_id="openai/gpt-oss-120b")
    analyst = AnalystAgent(model_id="openai/gpt-oss-120b")

    team = _build_team("openai/gpt-oss-120b", strat, analyst)

    assert str(getattr(team.mode, "value", team.mode)) == "route"
    assert team.respond_directly is True
    # crítico: o líder NÃO pode reescrever o input dos membros, senão corrompe os
    # prompts estruturados (lista de campos do Estrategista, iteração do Analista).
    assert team.determine_input_for_members is False
    member_names = {m.name for m in team.members}
    assert member_names == {STRATEGIST_MEMBER, ANALYST_MEMBER}
