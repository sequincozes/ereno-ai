"""orchestrator/workflow.py — encadeia Estrategista → Analista por iteração (#16, M3).

O orquestrador é o **controlador** do loop adversarial: a cada iteração pede uma
jogada ao Estrategista (Red Team), materializa a variante pelo núcleo
(patch → validação → gerador → avaliador do IDS) e entrega as métricas ao
Analista (Blue Team), persistindo tudo como ``IterationRecord`` (#2) via
``ExperimentMemory`` (#17).

Encaixe injetável
-----------------
O workflow não depende das implementações concretas dos agentes: recebe
qualquer objeto que satisfaça ``StrategistLike`` / ``AnalystLike``. Isso permite
rodar **hoje** com os stubs determinísticos (``FakeStrategist`` / ``FakeAnalyst``,
#3) — sem chave nem Java — e, quando o Estrategista com tool calling (#7, M1) e o
Analista real (#11, M2) ficarem prontos, plugá-los sem tocar nesta classe.

Agno Team
---------
``build_agno_team`` compõe os agentes reais num ``agno.team.Team`` (o caminho
"Agno Team" da issue). É opcional e lazy: só é chamado no caminho real (exige
chave da Groq e os agentes concretos), então o caminho de stubs continua
executável sem dependências de rede.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from adversarial_ids.core.experiment_memory import ExperimentMemory
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.shared.json_patch import apply_patch_to_json
from adversarial_ids.shared.validator import validate_and_clamp_attack_config

# Abaixo desta razão o ataque é considerado degenerado (raro/fraco demais).
_DEGENERATE_ATTACK_RATIO = 0.5


# --------------------------------------------------------------------------- #
# Contratos de encaixe (a fronteira do orquestrador com os agentes)           #
# --------------------------------------------------------------------------- #
@runtime_checkable
class StrategistLike(Protocol):
    """Qualquer Red Team que proponha uma jogada no formato ``StrategistOutput``.

    Retorna ``{"reasoning", "persona", "changes": [{"field", "value"}, ...]}``.
    ``FakeStrategist`` (#3) e o agente real (#7) satisfazem esta interface.
    """

    def propose(self, iteration: int, **context: Any) -> dict[str, Any]: ...


@runtime_checkable
class AnalystLike(Protocol):
    """Qualquer Blue Team que analise as ``Metrics`` no formato ``AnalystOutput``.

    ``FakeAnalyst`` (#3) e o agente real (#11) satisfazem esta interface.
    """

    def analyze(
        self, iteration: int, metrics: dict[str, Any], **context: Any
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Adaptador de contrato                                                       #
# --------------------------------------------------------------------------- #
def _to_output_dict(value: Any) -> dict[str, Any]:
    """Normaliza a saída de um agente para ``dict``.

    A fronteira do orquestrador com os agentes é frouxa de propósito: os stubs
    (``FakeStrategist``/``FakeAnalyst``, #3) e os agentes reais (#7/#11) tanto
    podem devolver um modelo Pydantic tipado (``StrategistOutput``/``AnalystOutput``)
    quanto um ``dict`` cru. O núcleo do loop fala em ``dict`` (``.get("changes")``,
    persistência via ``add_iteration``), então convertemos aqui num único ponto.
    """

    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(
        "Saída de agente inesperada: esperava dict ou pydantic BaseModel, "
        f"recebi {type(value).__name__}."
    )


def changes_to_patch(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``StrategistOutput.changes`` → patch do ``shared/json_patch``.

    A fronteira exata: o Estrategista fala em ``{field, value}``; o núcleo aplica
    ``{operation, field, old_value, new_value, reason}``. Mesma conversão que o
    ``FakeStrategist.to_patch`` faz, aqui como adaptador canônico do orquestrador.
    """

    patch: list[dict[str, Any]] = []
    for change in changes or []:
        if "field" not in change or "value" not in change:
            continue
        patch.append(
            {
                "operation": "replace",
                "field": change["field"],
                "old_value": None,
                "new_value": change["value"],
                "reason": "strategist_change",
            }
        )
    return patch


def _skipped_metrics() -> dict[str, Any]:
    """Métricas mínimas para uma iteração sem mudança de config (ERENO pulado)."""

    return {
        "evaluation_type": "skipped_no_config_change",
        "config_changed": False,
        "attack_count_ratio_vs_baseline": None,
        "degenerate_variant": None,
    }


# --------------------------------------------------------------------------- #
# O orquestrador                                                              #
# --------------------------------------------------------------------------- #
class AdversarialWorkflow:
    """Controlador do loop Red vs. Blue, agente-agnóstico."""

    def __init__(
        self,
        *,
        strategist: StrategistLike,
        analyst: AnalystLike,
        generator: GeneratorRunner,
        evaluator: IdsEvaluator,
        baseline_attack_config: dict[str, Any],
        memory: ExperimentMemory | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        self.strategist = strategist
        self.analyst = analyst
        self.generator = generator
        self.evaluator = evaluator
        self.baseline_attack_config = baseline_attack_config
        self.memory = memory if memory is not None else ExperimentMemory()
        self.save_path = Path(save_path) if save_path is not None else None

        self._baseline_attack_count: int | None = None

    # ------------------------------------------------------------------ #
    # Passo 0 — baseline                                                  #
    # ------------------------------------------------------------------ #
    def run_baseline(self) -> dict[str, Any]:
        """Treina o IDS no baseline e registra a iteração 0."""

        print("\n[ORCH] === BASELINE (iteração 0) ===")
        dataset_path = self.generator.generate_dataset(
            attack_config=self.baseline_attack_config,
            iteration=0,
        )
        metrics = self.evaluator.train_baseline(dataset_path)
        metrics["config_changed"] = False
        metrics["attack_count_ratio_vs_baseline"] = 1.0
        metrics["degenerate_variant"] = False

        self._baseline_attack_count = (
            metrics.get("full_attack_count")
            or metrics.get("attack_count")
            or metrics.get("support_attack")
        )

        analyst_output = self._safe_analyze(0, metrics)

        self.memory.add_iteration(
            iteration=0,
            attack_json=self.baseline_attack_config,
            metrics=metrics,
            strategist_output=None,
            analyst_output=analyst_output,
        )
        self._persist()
        return metrics

    # ------------------------------------------------------------------ #
    # Loop principal                                                      #
    # ------------------------------------------------------------------ #
    def run(self, iterations: int) -> ExperimentMemory:
        """Roda o baseline + ``iterations`` rodadas adversariais."""

        baseline_metrics = self.run_baseline()

        attack_config = dict(self.baseline_attack_config)
        performance = baseline_metrics

        for iteration in range(1, iterations + 1):
            attack_config, performance = self._run_iteration(
                iteration=iteration,
                attack_config=attack_config,
                performance=performance,
            )

        return self.memory

    def _run_iteration(
        self,
        *,
        iteration: int,
        attack_config: dict[str, Any],
        performance: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        print(f"\n[ORCH] === ITERAÇÃO {iteration} ===")

        # 1) Estrategista propõe (Red Team) -------------------------------
        strategist_output = self._safe_propose(
            iteration,
            attack_config=attack_config,
            performance=performance,
        )
        patch = changes_to_patch(strategist_output.get("changes", []))

        # 2) Aplica + valida (núcleo) -------------------------------------
        candidate = apply_patch_to_json(attack_config, patch) if patch else attack_config
        validated, warnings = validate_and_clamp_attack_config(
            candidate_config=candidate,
            baseline_config=self.baseline_attack_config,
        )
        for warning in warnings:
            print(f"[VALIDATOR] {warning}")

        config_changed = validated != attack_config

        # 3) Gera a variante e avalia o IDS (núcleo) ----------------------
        if config_changed:
            print("[ORCH] Configuração alterada — gerando e avaliando variante.")
            dataset_path = self.generator.generate_dataset(validated, iteration)
            metrics = self.evaluator.evaluate_variant(dataset_path)
            metrics["config_changed"] = True
            self._annotate_attack_ratio(metrics)
        else:
            print("[ORCH] Configuração inalterada — pulando o gerador.")
            metrics = _skipped_metrics()

        # 4) Analista explica (Blue Team) ---------------------------------
        analyst_output = self._safe_analyze(iteration, metrics)

        # 5) Persiste o IterationRecord -----------------------------------
        self.memory.add_iteration(
            iteration=iteration,
            attack_json=validated,
            metrics=metrics,
            strategist_output=strategist_output,
            analyst_output=analyst_output,
        )
        self._persist()

        return validated, metrics

    # ------------------------------------------------------------------ #
    # Chamadas resilientes aos agentes                                    #
    # ------------------------------------------------------------------ #
    # Um output ruim de um LLM (tool call rejeitada, feature alucinada que a
    # validação recusa) não deve derrubar o experimento inteiro: registramos o
    # ocorrido e seguimos — o Estrategista vira um no-op (config inalterada) e o
    # Analista fica ausente naquela iteração.
    def _safe_propose(
        self,
        iteration: int,
        *,
        attack_config: dict[str, Any],
        performance: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return _to_output_dict(
                self.strategist.propose(
                    iteration,
                    attack_config=attack_config,
                    metrics=performance,
                    history=self.memory.get_history(),
                )
            )
        except Exception as exc:  # fronteira do agente
            print(
                f"[ORCH] Estrategista falhou na iteração {iteration} "
                f"({exc}); mantendo a configuração atual."
            )
            return {"reasoning": f"[erro] {exc}", "persona": "conservative", "changes": []}

    def _safe_analyze(
        self, iteration: int, metrics: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            return _to_output_dict(self.analyst.analyze(iteration, metrics)) or None
        except Exception as exc:  # fronteira do agente
            print(
                f"[ORCH] Analista falhou na iteração {iteration} "
                f"({exc}); registrando a iteração sem análise."
            )
            return None

    # ------------------------------------------------------------------ #
    # Auxiliares                                                          #
    # ------------------------------------------------------------------ #
    def _annotate_attack_ratio(self, metrics: dict[str, Any]) -> None:
        current = metrics.get("attack_count") or metrics.get("support_attack")
        baseline = self._baseline_attack_count

        if baseline and current:
            ratio = current / baseline
        else:
            ratio = None

        metrics["attack_count_ratio_vs_baseline"] = ratio
        metrics["degenerate_variant"] = (
            ratio is not None and ratio < _DEGENERATE_ATTACK_RATIO
        )
        if metrics["degenerate_variant"]:
            print(
                f"[VALIDATOR] Variante degenerada: attack_count em {ratio:.2%} do baseline."
            )

    def _persist(self) -> None:
        if self.save_path is not None:
            self.memory.save(self.save_path)
            print(f"[ORCH] Histórico salvo em: {self.save_path}")


# --------------------------------------------------------------------------- #
# Fábrica do Agno Team (caminho dos agentes reais)                            #
# --------------------------------------------------------------------------- #
def build_agno_team(
    members: list[Any],
    *,
    model: Any,
    name: str = "AdversarialTeam",
    instructions: str | None = None,
    mode: str = "route",
    respond_directly: bool = True,
    determine_input_for_members: bool = False,
) -> Any:
    """Compõe os agentes reais (Estrategista, Analista) num ``agno.team.Team``.

    Import lazy de ``agno`` para não acoplar o caminho de stubs à biblioteca nem
    exigir chave em tempo de import. Os ``members`` são ``agno.agent.Agent`` já
    construídos pelos donos dos agentes (M1/#7 e M2/#11); ``model`` é o LLM
    coordenador (ex.: ``agno.models.groq.Groq``). Executar o time exige
    ``GROQ_API_KEY``.

    ``mode="route"`` + ``respond_directly=True`` fazem o líder **encaminhar** a
    tarefa ao membro certo e devolver a resposta dele sem reescrever — é o que
    permite ao orquestrador recuperar a saída estruturada de cada agente
    (``StrategistOutput`` via tool / ``AnalystOutput`` via ``output_schema``)
    intacta a partir de ``TeamRunOutput.member_responses`` (ver ``live.py``).

    ``determine_input_for_members=False`` é crítico: por padrão o líder do time
    **reescreve** o input antes de passá-lo ao membro, o que corrompe os prompts
    estruturados (a lista exata de campos editáveis do Estrategista, o número da
    iteração do Analista). Desligando isso, o membro recebe o prompt montado pelo
    dono do agente intacto.
    """

    from agno.team import Team

    return Team(
        members=members,
        model=model,
        name=name,
        mode=mode,
        respond_directly=respond_directly,
        determine_input_for_members=determine_input_for_members,
        instructions=instructions
        or (
            "Coordene o loop adversarial: o Estrategista (Red Team) propõe a "
            "alteração do ataque e o Analista (Blue Team) explica e recomenda "
            "mitigação a cada iteração. Encaminhe cada tarefa ao membro indicado "
            "na mensagem."
        ),
    )
