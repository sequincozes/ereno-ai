"""orchestrator/live.py — monta o loop adversarial com os agentes **reais**.

O ``AdversarialWorkflow`` (workflow.py) é agente-agnóstico: fala com qualquer
objeto que satisfaça ``StrategistLike`` / ``AnalystLike``. Este módulo é a costura
que faltava (Fase 2): embrulha os agentes reais (``StrategistAgent`` #7 / M1 e
``AnalystAgent`` #11 / M2) nesses contratos, constrói o núcleo a partir das
``settings`` e expõe ``run_live_workflow`` — a função pública que a camada de
interface (CLI/Dashboard, M4) liga via ``WorkflowAdapter``.

É o caminho **opt-in**: exige ``GROQ_API_KEY`` (agentes reais) e, no modo ``jar``,
o JAR do ERENO. O caminho de demonstração cacheada (golden) continua sendo o
default das interfaces e não passa por aqui — por isso o import de ``agno`` (via
os agentes) fica confinado a este módulo, carregado só quando o loop real roda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adversarial_ids.agents.analyst.agent import AnalystAgent
from adversarial_ids.agents.analyst.tools import validate_output_against_metrics
from adversarial_ids.agents.orchestrator.workflow import (
    AdversarialWorkflow,
    build_agno_team,
)
from adversarial_ids.agents.strategist.agent import StrategistAgent
from adversarial_ids.config.attacks_registry import (
    DEFAULT_ATTACK_KEY,
    AttackSpec,
    get_attack_spec,
)
from adversarial_ids.config.settings import (
    BASELINE_DATASET_PATH,
    GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RUN_COMMAND,
    GENERATOR_RUNTIME_DIR,
    ITERATION_HISTORY_PATH,
    MODEL_ID,
    OUTPUTS_DIR,
    PROMPT_PATH,
)
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain import IterationRecord, Metrics
from adversarial_ids.domain.strategist_output import StrategistOutput
from adversarial_ids.shared.json_io import load_json

# Nomes dos membros do Team — usados pelo líder (route mode) para encaminhar a
# tarefa e por nós para recuperar a resposta certa em ``member_responses``.
STRATEGIST_MEMBER = "Estrategista"
ANALYST_MEMBER = "Analista"


# --------------------------------------------------------------------------- #
# Adaptador do Estrategista real ao contrato do orquestrador                   #
# --------------------------------------------------------------------------- #
class StrategistAdapter:
    """Faz ``StrategistAgent`` satisfazer ``StrategistLike``.

    O orquestrador chama ``propose(iteration, *, attack_config, metrics, history)``
    e espera de volta o ``StrategistOutput`` (o workflow normaliza para dict). O
    agente real expõe ``suggest_changes(base_prompt, attack_json, ...)``, então a
    tradução mora aqui: carrega o ``base_prompt`` versionado uma única vez e mapeia
    os nomes dos argumentos.
    """

    def __init__(self, agent: StrategistAgent, base_prompt: str) -> None:
        self._agent = agent
        self._base_prompt = base_prompt

    def propose(
        self,
        iteration: int,
        *,
        attack_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> Any:
        del iteration  # o histórico nativo do Agno cuida da memória entre iterações
        return self._agent.suggest_changes(
            base_prompt=self._base_prompt,
            attack_json=attack_config or {},
            performance_results=metrics or {},
            history=history or [],
        )


class AnalystAdapter:
    """Faz ``AnalystAgent`` satisfazer ``AnalystLike``.

    O agente real já expõe ``analyze(iteration, metrics, shap_importances=None)``
    e devolve ``AnalystOutput``; este adaptador só fixa a assinatura do contrato
    (``**context`` extra que o orquestrador pode passar) sem repassar kwargs que o
    agente não conhece.
    """

    def __init__(self, agent: AnalystAgent) -> None:
        self._agent = agent

    def analyze(
        self,
        iteration: int,
        metrics: dict[str, Any],
        **_: Any,
    ) -> Any:
        return self._agent.analyze(iteration, metrics)


# --------------------------------------------------------------------------- #
# Adaptadores Team-backed (route mode) — o Team dirige as chamadas dos agentes  #
# --------------------------------------------------------------------------- #
def _member_response(team_output: Any, agent_name: str) -> Any:
    """Recupera o ``RunOutput`` do membro roteado a partir do ``TeamRunOutput``.

    Em ``route`` + ``respond_directly`` o membro alvo aparece em
    ``member_responses``; casamos por ``agent_name``. Se não achar (ex.: só um
    membro respondeu), caímos no último ``member_response`` e, por fim, no próprio
    ``team_output`` — que também expõe ``.tools`` e ``.content``.
    """

    responses = list(getattr(team_output, "member_responses", None) or [])
    for response in responses:
        if getattr(response, "agent_name", None) == agent_name:
            return response
    if responses:
        return responses[-1]
    return team_output


class TeamStrategist:
    """``StrategistLike`` que obtém a jogada roteando pelo ``agno.team.Team``.

    Reusa ``StrategistAgent`` para montar o prompt e para extrair a saída
    estruturada (a tool ``submit_strategist_output``) da resposta do membro — a
    diferença é que a execução passa pelo líder do Team (``team.run``) em vez do
    ``agent.run`` direto.
    """

    def __init__(
        self,
        team: Any,
        agent: StrategistAgent,
        base_prompt: str,
        member_name: str = STRATEGIST_MEMBER,
    ) -> None:
        self._team = team
        self._agent = agent
        self._base_prompt = base_prompt
        self._member_name = member_name

    def propose(
        self,
        iteration: int,
        *,
        attack_config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> StrategistOutput:
        del iteration
        prompt = self._agent.build_prompt(
            base_prompt=self._base_prompt,
            attack_json=attack_config or {},
            performance_results=metrics or {},
            history=history or [],
        )
        directive = (
            f"Encaminhe a tarefa a seguir ao membro {self._member_name} "
            "(Red Team) e devolva a resposta dele.\n\n"
        )
        output = self._team.run(directive + prompt)
        response = _member_response(output, self._member_name)

        tool_result = StrategistAgent._extract_tool_result(response)
        if tool_result is not None:
            return StrategistOutput.model_validate_json(tool_result)
        return StrategistAgent._fallback_parse(getattr(response, "content", "") or "")


class TeamAnalyst:
    """``AnalystLike`` que obtém o diagnóstico roteando pelo ``agno.team.Team``.

    Reusa ``AnalystAgent`` (prompt) e as regras determinísticas de
    ``analyst/tools.py`` para validar a saída do membro contra as métricas.
    """

    def __init__(
        self,
        team: Any,
        agent: AnalystAgent,
        member_name: str = ANALYST_MEMBER,
    ) -> None:
        self._team = team
        self._agent = agent
        self._member_name = member_name

    def analyze(self, iteration: int, metrics: dict[str, Any], **_: Any) -> Any:
        metrics_model = Metrics.model_validate(metrics)
        prompt = self._agent.build_prompt(iteration=iteration, metrics=metrics_model)
        directive = (
            f"Encaminhe a tarefa a seguir ao membro {self._member_name} "
            "(Blue Team) e devolva a resposta dele.\n\n"
        )
        output = self._team.run(directive + prompt)
        response = _member_response(output, self._member_name)

        parsed = AnalystAgent._parse_response_content(getattr(response, "content", None))
        if parsed.iteration != iteration:
            raise ValueError(
                "A iteração retornada pelo Analista (via Team) é incompatível: "
                f"esperado={iteration}, recebido={parsed.iteration}"
            )
        return validate_output_against_metrics(
            output=parsed,
            metrics=metrics_model,
            shap_importances=None,
        )


def _build_team(
    model_id: str,
    strategist_agent: StrategistAgent,
    analyst_agent: AnalystAgent,
) -> Any:
    """Compõe os dois agentes reais num ``agno.team.Team`` em modo ``route``.

    Nomeia os membros para que o líder saiba encaminhar (route) e para casarmos a
    resposta em ``member_responses``. O modelo coordenador reusa o mesmo
    ``model_id`` dos agentes.
    """

    from agno.models.groq import Groq

    strategist_agent.agent.name = STRATEGIST_MEMBER
    strategist_agent.agent.role = (
        "Red Team: propõe alterações parametrizadas no ataque sintético do ERENO."
    )
    analyst_agent.agent.name = ANALYST_MEMBER
    analyst_agent.agent.role = (
        "Blue Team: explica quais features enganaram o IDS e recomenda mitigação."
    )

    return build_agno_team(
        members=[strategist_agent.agent, analyst_agent.agent],
        model=Groq(id=model_id),
    )


# --------------------------------------------------------------------------- #
# Núcleo                                                                        #
# --------------------------------------------------------------------------- #
def _build_generator(generator_mode: str, spec: AttackSpec) -> GeneratorRunner:
    """Constrói o ``GeneratorRunner`` no modo pedido (``cached`` ou ``jar``).

    No modo ``jar`` o runner seleciona o segmento do ataque (``spec.segment_name``)
    no action config a cada geração, então trocar de ataque é só trocar a ``spec``.
    """

    if generator_mode not in ("cached", "jar"):
        raise ValueError(
            f"generator_mode inválido: {generator_mode!r}. Use 'cached' ou 'jar'."
        )

    cached_dataset = BASELINE_DATASET_PATH if generator_mode == "cached" else None

    return GeneratorRunner(
        runtime_dir=GENERATOR_RUNTIME_DIR,
        output_dataset_path=GENERATOR_OUTPUT_DATASET_PATH,
        run_command=GENERATOR_RUN_COMMAND,
        suggested_config_path=str(OUTPUTS_DIR / "suggested_attack_config.json"),
        action_config_relative_path=GENERATOR_ACTION_CONFIG_RELATIVE_PATH,
        segment_name=spec.segment_name,
        cached_dataset_path=cached_dataset,
    )


# --------------------------------------------------------------------------- #
# Fábrica pública                                                              #
# --------------------------------------------------------------------------- #
def _attack_context(spec: AttackSpec) -> str:
    """Bloco injetado no prompt do Estrategista com o contexto do ataque atual."""
    return (
        "\n\n## Ataque desta execução\n"
        f"- Tipo (`attackType`): `{spec.attack_type}`\n"
        f"- Classe rotulada no dataset: `{spec.label}`\n"
        f"- Descrição: {spec.description}\n"
        "Os campos editáveis abaixo são derivados automaticamente da configuração "
        "deste ataque — proponha alterações apenas nesses caminhos. Seu objetivo é "
        "reduzir `f1_score_attack` (detecção da classe de ataque) sem descaracterizar "
        "o ataque (não zere frequência/intensidade a ponto de a variante degenerar).\n"
    )


def build_live_workflow(
    *,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
    persona: str = "conservative",
    use_team: bool = True,
    save_path: str | Path | None = ITERATION_HISTORY_PATH,
    attack: str = DEFAULT_ATTACK_KEY,
) -> AdversarialWorkflow:
    """Monta o ``AdversarialWorkflow`` com os agentes reais + núcleo real.

    Exige ``GROQ_API_KEY`` (agentes reais). Em ``generator_mode='jar'`` também
    exige o JAR do ERENO em ``generator_runtime/``.

    ``use_team=True`` (default) dirige as chamadas dos agentes por um
    ``agno.team.Team`` (route mode): a cada iteração o líder encaminha ao
    Estrategista e, depois do núcleo determinístico, ao Analista. ``use_team=False``
    mantém o encadeamento direto (``StrategistAdapter``/``AnalystAdapter``), útil
    como fallback determinístico e para diagnóstico.
    """

    spec = get_attack_spec(attack)
    baseline_attack_config = load_json(spec.baseline_path)
    base_prompt = Path(PROMPT_PATH).read_text(encoding="utf-8") + _attack_context(spec)

    strategist_agent = StrategistAgent(model_id=model_id, persona=persona)
    analyst_agent = AnalystAgent(model_id=model_id)

    if use_team:
        team = _build_team(model_id, strategist_agent, analyst_agent)
        strategist: Any = TeamStrategist(team, strategist_agent, base_prompt)
        analyst: Any = TeamAnalyst(team, analyst_agent)
    else:
        strategist = StrategistAdapter(agent=strategist_agent, base_prompt=base_prompt)
        analyst = AnalystAdapter(analyst_agent)

    return AdversarialWorkflow(
        strategist=strategist,
        analyst=analyst,
        generator=_build_generator(generator_mode, spec),
        evaluator=IdsEvaluator(drop_cb_status=False, target_attack_label=spec.label),
        baseline_attack_config=baseline_attack_config,
        save_path=save_path,
    )


def run_live_workflow(
    *,
    iterations: int,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
    use_team: bool = True,
    persona: str = "conservative",
    attack: str = DEFAULT_ATTACK_KEY,
) -> list[IterationRecord]:
    """Roda o loop real e devolve os ``IterationRecord`` (baseline + iterações).

    Assinatura compatível com ``interfaces.experiment_runner.WorkflowCallable``,
    para que ``create_default_runner('live')`` a ligue via ``WorkflowAdapter``.
    """

    workflow = build_live_workflow(
        model_id=model_id,
        generator_mode=generator_mode,
        use_team=use_team,
        persona=persona,
        attack=attack,
    )
    memory = workflow.run(iterations)
    return memory.get_records()
