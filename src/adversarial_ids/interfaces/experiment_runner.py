"""Contrato de execução compartilhado pela CLI e pelo dashboard."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from adversarial_ids.config.attacks_registry import DEFAULT_ATTACK_KEY
from adversarial_ids.config.settings import DETECTOR_MODE, GOLDEN_HISTORY_PATH
from adversarial_ids.domain import IterationRecord


class ExperimentRunner(Protocol):
    """Porta mínima que a camada de interface espera do orquestrador."""

    def run(
        self,
        *,
        iterations: int,
        model_id: str,
        generator_mode: str,
        attack: str = DEFAULT_ATTACK_KEY,
    ) -> list[IterationRecord]: ...


class CachedHistoryRunner:
    """Runner provisório que serve a fixture golden sem executar agentes ou Java."""

    def __init__(self, history_path: Path = GOLDEN_HISTORY_PATH) -> None:
        self.history_path = history_path

    def run(
        self,
        *,
        iterations: int,
        model_id: str,
        generator_mode: str,
        attack: str = DEFAULT_ATTACK_KEY,
    ) -> list[IterationRecord]:
        del model_id  # O golden já foi produzido e não depende do modelo selecionado.

        if attack != DEFAULT_ATTACK_KEY:
            # O histórico golden foi gravado para o uc03; o replay não regenera
            # dataset, então outros ataques só têm efeito no motor 'live'.
            print(
                "[DEMO] --attack é ignorado no motor de demonstração (replay do "
                f"golden uc03). Use --engine live para rodar '{attack}'."
            )

        if generator_mode != "cached":
            raise ValueError(
                "O runner de demonstração suporta somente generator_mode='cached'."
            )
        if iterations < 0:
            raise ValueError("iterations não pode ser negativo.")
        if not self.history_path.exists():
            raise FileNotFoundError(
                f"Histórico cacheado não encontrado: {self.history_path}"
            )

        with self.history_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        history = payload.get("history")
        if not isinstance(history, list):
            raise ValueError("O histórico cacheado deve conter uma lista em 'history'.")

        records = [IterationRecord.model_validate(item) for item in history]
        # Iteração zero é o baseline; N significa baseline + até N variantes.
        return records[: iterations + 1]


WorkflowCallable = Callable[..., Iterable[IterationRecord | dict[str, Any]]]


class WorkflowAdapter:
    """Adapta uma função pública do workflow ao contrato das interfaces.

    A função concreta será fornecida pelo M3 na integração. Manter essa tradução
    aqui evita que CLI e dashboard dependam diretamente de detalhes do Agno.
    """

    def __init__(self, run_workflow: WorkflowCallable) -> None:
        self.run_workflow = run_workflow

    def run(
        self,
        *,
        iterations: int,
        model_id: str,
        generator_mode: str,
        attack: str = DEFAULT_ATTACK_KEY,
    ) -> list[IterationRecord]:
        result = self.run_workflow(
            iterations=iterations,
            model_id=model_id,
            generator_mode=generator_mode,
            attack=attack,
        )
        return [IterationRecord.model_validate(item) for item in result]


def create_default_runner(
    engine: str = "demo",
    *,
    orchestration: str = "team",
    persona: str = "conservative",
    detector: str | None = None,
) -> ExperimentRunner:
    """Seleciona a implementação usada pelas interfaces.

    - ``demo`` (default) → ``CachedHistoryRunner``: replay do histórico golden,
      sem Groq nem Java. É o caminho da demonstração.
    - ``live`` → ``WorkflowAdapter`` sobre o loop real do orquestrador
      (``run_live_workflow``). Exige ``GROQ_API_KEY`` e, em ``generator_mode='jar'``,
      o JAR do ERENO. O import é lazy para o caminho ``demo`` não depender de
      ``agno`` nem exigir chave.

    ``orchestration`` só vale para ``live``: ``team`` (default) dirige os agentes
    por um ``agno.team.Team`` (route mode); ``direct`` usa o encadeamento
    determinístico como fallback. ``persona`` seleciona o perfil do Estrategista
    (``conservative`` = 1 alteração/iteração; ``aggressive`` = até 3).
    ``detector`` (E8) também só vale para ``live``: escolhe o modelo que o
    ``IdsEvaluator`` treina (``decision_tree``, ``svm_linear``, ``svm_rbf`` —
    ver ``core/detectors.py``); ``None`` (default) herda
    ``settings.DETECTOR_MODE``. A distinção entre ``None`` e uma chave
    explícita importa: no caminho ``demo`` o histórico golden já está gravado
    e nenhum detector é treinado, então **qualquer** escolha explícita ali é
    erro — comparar com o default de ambiente em vez de com ``None`` faria a
    rejeição depender de ``DETECTOR_MODE`` e recusar até
    ``detector="random_forest"``.
    """

    if engine == "live":
        from functools import partial

        from adversarial_ids.agents.orchestrator.live import run_live_workflow
        from adversarial_ids.core.detectors import detector_spec

        if orchestration not in ("team", "direct"):
            raise ValueError(
                f"orchestration inválido: {orchestration!r}. Use 'team' ou 'direct'."
            )
        if persona not in ("conservative", "aggressive"):
            raise ValueError(
                f"persona inválida: {persona!r}. Use 'conservative' ou 'aggressive'."
            )

        # Levanta ``DetectorError`` para uma chave desconhecida, na mesma
        # posição em que orchestration/persona são validados — antes de
        # qualquer trabalho caro.
        resolved_detector = detector or DETECTOR_MODE
        detector_spec(resolved_detector)

        run = partial(
            run_live_workflow,
            use_team=orchestration == "team",
            persona=persona,
            detector=resolved_detector,
        )
        return WorkflowAdapter(run)

    if engine != "demo":
        raise ValueError(f"engine inválido: {engine!r}. Use 'demo' ou 'live'.")

    if detector is not None:
        raise ValueError(
            f"detector={detector!r} não se aplica ao engine 'demo': o histórico "
            "golden é um replay já gravado, nenhum detector é treinado."
        )

    return CachedHistoryRunner()
