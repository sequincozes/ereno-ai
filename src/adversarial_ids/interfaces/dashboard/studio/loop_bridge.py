"""Ponte de execução do pipeline intent-driven para a UI (épico E11).

Irmã de ``bridge.py``, e deliberadamente diferente dela num ponto: o loop
legado comunica progresso por ``print``, e aquela ponte precisa capturar
``stdout`` e adivinhar a fase por substring. Aqui o núcleo emite ``LoopEvent``
tipado, então esta ponte só registra um sink e lê eventos — nenhuma string é
interpretada, e renomear uma mensagem de log não quebra a barra de progresso.

Nada aqui importa ``streamlit``: é lógica pura, testável sem subir a UI.

## O resultado de uma execução não mora no LoopRecord

O ``LoopRecord`` guarda o caminho de cada artefato, não o conteúdo. Para a tela
mostrar "resultado" — que é metade do critério de pronto do E11 — alguém precisa
abrir `detection_report.json`, `defense_plan.json` e `defense_rules.json` do
diretório da execução. ``RunOutcome`` faz isso uma vez e entrega já tipado,
tolerando ausência: uma execução que falhou no estágio ERENO não tem relatório
de detecção, e isso é informação, não erro.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from adversarial_ids.config.settings import (
    DETECTOR_MODE,
    INTENT_LOOP_DEFAULT_ROUNDS,
    LOOP_RECORDS_PATH,
    MODEL_ID,
)
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.defense_rule_report import DefenseRuleReport
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.loop_event import LoopEvent
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus
from adversarial_ids.shared.json_io import load_json
from adversarial_ids.shared.loop_event_store import MemoryEventSink, load_loop_events
from adversarial_ids.shared.loop_record_store import load_loop_records

RunIntentLoop = Callable[..., tuple[LoopRecord, ...]]


@dataclass
class IntentLoopConfig:
    """Parâmetros de uma execução intent-driven — o que o formulário monta."""

    prompt: str = ""
    rounds: int = INTENT_LOOP_DEFAULT_ROUNDS
    generator_mode: str = "cached"
    detector: str = DETECTOR_MODE
    model_id: str = MODEL_ID

    def as_summary(self) -> dict[str, str]:
        return {
            "Rodadas": str(self.rounds),
            "Gerador": self.generator_mode,
            "Detector": self.detector,
            "Modelo": self.model_id,
        }


@dataclass(frozen=True)
class RunOutcome:
    """O resultado de uma execução, lido dos artefatos que ela deixou.

    Todo campo é opcional porque toda execução pode parar antes de produzi-lo. A
    ausência é exibida como ausência — inventar um valor neutro faria uma
    execução interrompida no ERENO parecer uma que detectou zero.
    """

    run_dir: Path | None = None
    detection_report: DetectionReport | None = None
    defense_plan: DefensePlan | None = None
    defense_rules: DefenseRuleReport | None = None

    @property
    def has_result(self) -> bool:
        return self.detection_report is not None


def _load_artifact(path: Path, model: Any) -> Any | None:
    """Lê um artefato tipado da execução, ou ``None`` se ele não existe/não vale.

    Tolerante de propósito: esta é a camada de apresentação de uma execução que
    pode ter morrido em qualquer estágio, e uma tela em branco ensina menos do
    que uma tela que mostra o que sobrou.
    """

    if not path.exists():
        return None
    try:
        return model.model_validate(load_json(path))
    except Exception:  # fronteira da interface
        return None


def run_directory(record: LoopRecord) -> Path | None:
    """Deduz o diretório da execução a partir de qualquer artefato registrado.

    O ``LoopRecord`` não carrega o diretório: carrega o caminho de cada
    artefato, e todos moram no mesmo lugar. Derivar do primeiro é mais honesto
    que recompor a partir de ``settings.INTENT_LOOP_OUTPUT_DIR``, que ignoraria
    um ``output_dir`` customizado.
    """

    for stage in record.stages:
        ref = stage.artifact_ref
        if ref and ref.endswith(".json"):
            return Path(ref).parent

    return None


def outcome_for(record: LoopRecord) -> RunOutcome:
    """Abre os artefatos da execução e devolve o que existir deles."""

    run_dir = run_directory(record)
    if run_dir is None:
        return RunOutcome()

    return RunOutcome(
        run_dir=run_dir,
        detection_report=_load_artifact(
            run_dir / "detection_report.json", DetectionReport
        ),
        defense_plan=_load_artifact(run_dir / "defense_plan.json", DefensePlan),
        defense_rules=_load_artifact(
            run_dir / "defense_rules.json", DefenseRuleReport
        ),
    )


def saved_records(path: Path | str = LOOP_RECORDS_PATH) -> list[LoopRecord]:
    """Execuções já persistidas, da mais recente para a mais antiga."""

    return list(reversed(load_loop_records(path)))


def timeline_for(record: LoopRecord) -> list[LoopEvent]:
    """A timeline gravada daquela execução (vazia se o arquivo não existir)."""

    run_dir = run_directory(record)
    if run_dir is None:
        return []

    return load_loop_events(run_dir / "events.jsonl")


@dataclass
class IntentLoopJob:
    """Uma campanha intent-driven rodando em background, observável ao vivo."""

    config: IntentLoopConfig
    runner: RunIntentLoop | None = None
    records: tuple[LoopRecord, ...] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _sink: MemoryEventSink = field(init=False, default_factory=MemoryEventSink)
    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ereno-intent-loop", daemon=True
        )

    # -- ciclo de vida ----------------------------------------------------- #
    def start(self) -> "IntentLoopJob":
        self._thread.start()
        return self

    def _resolve_runner(self) -> RunIntentLoop:
        """Importa o caminho ao vivo só na hora de usar.

        ``intent_live`` importa ``agno`` no topo; resolvê-lo na construção faria
        a página inteira exigir a dependência para apenas *abrir*.
        """

        if self.runner is not None:
            return self.runner

        from adversarial_ids.agents.orchestrator.intent_live import run_intent_loop

        return run_intent_loop

    def _run(self) -> None:
        try:
            cfg = self.config
            self.records = self._resolve_runner()(
                prompt=cfg.prompt,
                model_id=cfg.model_id,
                generator_mode=cfg.generator_mode,
                rounds=cfg.rounds,
                detector=cfg.detector,
                event_sink=self._sink,
            )
        except Exception as exc:  # fronteira da interface
            self.error = str(exc)
        finally:
            self.finished_at = time.time()

    # -- inspeção ---------------------------------------------------------- #
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def events(self) -> tuple[LoopEvent, ...]:
        return self._sink.snapshot()

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def current_stage(self) -> str | None:
        """O estágio que está rodando agora, ou ``None`` entre um e outro.

        Lido do evento, não deduzido de texto: é um ``stage_started`` sem o
        ``stage_finished`` correspondente.
        """

        running: str | None = None
        for event in self.events():
            if event.kind == "stage_started":
                running = event.stage
            elif event.kind == "stage_finished" and event.stage == running:
                running = None

        return running

    def progress(self) -> float:
        """Fração concluída: estágios fechados sobre os esperados na campanha.

        Sem heurística de texto e sem teto artificial — a campanha tem um número
        conhecido de estágios (sete por rodada), e um estágio só conta depois de
        anunciar que terminou.
        """

        expected = 7 * max(self.config.rounds, 1)
        finished = sum(1 for e in self.events() if e.kind == "stage_finished")

        return min(1.0, finished / expected)

    def failed_stage(self) -> LoopEvent | None:
        """O primeiro estágio que falhou, com a causa — ou ``None``."""

        for event in self.events():
            if event.status is LoopStageStatus.FAILED and event.stage is not None:
                return event

        return None
