"""Ponte de execução: roda o loop adversarial e captura os logs para a UI.

O núcleo (``AdversarialWorkflow``) comunica progresso via ``print`` com prefixos
(``[ORCH]``, ``[VALIDATOR]``, ``[DEMO]``). Para transformar isso num painel de
logs ao vivo sem tocar no núcleo, executamos o *runner* numa thread e redirecionamos
``sys.stdout``/``sys.stderr`` para um buffer thread-safe que também repassa para o
console real (tee). A página de execução faz *polling* do buffer enquanto a thread
vive.

Nada aqui importa ``streamlit`` — é lógica pura, testável e reusável.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from adversarial_ids.domain import IterationRecord
from adversarial_ids.interfaces.experiment_runner import create_default_runner


class _TeeBuffer:
    """Coletor de linhas thread-safe que também escreve no stream original."""

    def __init__(self, mirror: Any) -> None:
        self._mirror = mirror
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._partial = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        try:
            if self._mirror is not None:
                self._mirror.write(text)
        except Exception:
            pass
        with self._lock:
            self._partial += text
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._lines.append(line)
        return len(text)

    def flush(self) -> None:
        try:
            if self._mirror is not None:
                self._mirror.flush()
        except Exception:
            pass

    def snapshot(self) -> list[str]:
        with self._lock:
            lines = list(self._lines)
            if self._partial:
                lines.append(self._partial)
            return lines


@dataclass
class ExperimentConfig:
    """Parâmetros de uma execução — o que a UI de configuração monta."""

    engine: str = "demo"                 # demo | live
    iterations: int = 3
    attack: str = "masquerade_fault"
    persona: str = "conservative"        # conservative | aggressive
    orchestration: str = "team"          # team | direct
    generator_mode: str = "cached"       # cached | jar
    model_id: str = "llama-3.1-8b-instant"

    def as_summary(self) -> dict[str, str]:
        return {
            "Motor": self.engine,
            "Iterações": str(self.iterations),
            "Ataque": self.attack,
            "Persona": self.persona,
            "Orquestração": self.orchestration,
            "Gerador": self.generator_mode,
            "Modelo": self.model_id,
        }


@dataclass
class ExperimentJob:
    """Uma execução em andamento (ou concluída) rodando em background."""

    config: ExperimentConfig
    _buffer: _TeeBuffer = field(init=False)
    _thread: threading.Thread = field(init=False)
    records: list[IterationRecord] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def __post_init__(self) -> None:
        self._buffer = _TeeBuffer(sys.__stdout__)
        self._thread = threading.Thread(target=self._run, name="ereno-exp", daemon=True)

    # -- ciclo de vida ----------------------------------------------------- #
    def start(self) -> "ExperimentJob":
        self._thread.start()
        return self

    def _run(self) -> None:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = self._buffer          # type: ignore[assignment]
        sys.stderr = self._buffer          # type: ignore[assignment]
        try:
            cfg = self.config
            generator_mode = "cached" if cfg.engine == "demo" else cfg.generator_mode
            runner = create_default_runner(
                cfg.engine,
                orchestration=cfg.orchestration,
                persona=cfg.persona,
            )
            print(f"[ORCH] Iniciando execução · motor={cfg.engine} · "
                  f"ataque={cfg.attack} · iterações={cfg.iterations}")
            records = runner.run(
                iterations=cfg.iterations,
                model_id=cfg.model_id,
                generator_mode=generator_mode,
                attack=cfg.attack,
            )
            self.records = list(records)
            if cfg.engine == "demo":
                # O replay golden não passa pelo AdversarialWorkflow, então
                # sintetizamos um recap por iteração para o painel contar a história.
                for rec in self.records:
                    f1 = rec.metrics.f1_score_attack
                    f1_txt = "—" if f1 is None else f"{f1:.4f}"
                    changed = rec.metrics.config_changed
                    tag = "baseline" if rec.iteration == 0 else (
                        "config alterada" if changed else "config inalterada")
                    print(f"[ORCH] Iteração {rec.iteration} · F1(ataque)={f1_txt} · {tag}")
                    if rec.analyst_output is not None:
                        print(f"[VALIDATOR] Analista presente · severidade="
                              f"{rec.analyst_output.severity}")
            print(f"[ORCH] Execução concluída · {len(self.records)} registro(s).")
        except Exception as exc:  # fronteira da interface
            self.error = str(exc)
            print(f"[ERRO] {exc}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self.finished_at = time.time()

    # -- inspeção ---------------------------------------------------------- #
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def logs(self) -> list[str]:
        return self._buffer.snapshot()

    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    def progress(self) -> tuple[int, str]:
        """Heurística de progresso a partir dos logs (iteração atual / fase)."""
        current = 0
        phase = "Inicializando…"
        for line in self.logs():
            s = line.strip()
            if "BASELINE" in s:
                current = 0
                phase = "Baseline (iteração 0)"
            elif "=== ITERAÇÃO" in s:
                try:
                    current = int(s.split("ITERAÇÃO", 1)[1].split("=", 1)[0].strip())
                    phase = f"Iteração {current}"
                except (ValueError, IndexError):
                    pass
            elif "gerando e avaliando" in s.lower():
                phase = f"Iteração {current} · gerando + avaliando IDS"
            elif "concluída" in s.lower():
                phase = "Concluído"
        if not self.is_running():
            return 100, ("Falhou" if self.error else "Concluído")
        total = max(self.config.iterations, 1)
        fraction = min(0.98, (current + 1) / (total + 1))
        return int(fraction * 100), phase
