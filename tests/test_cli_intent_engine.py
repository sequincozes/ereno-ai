"""Testes de --engine intent na CLI (feature flag do E3/E4).

Injeta ``run_intent_loop`` para não depender de GROQ_API_KEY/Java — mesmo
espírito de ``RecordingRunner``/``FailingRunner`` em ``test_cli_runner.py``.
"""

from __future__ import annotations

from io import StringIO

from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.interfaces.cli import run_cli


def _record(*, run_id: str = "run-1", failed: bool = False) -> LoopRecord:
    stages = [
        LoopStage(
            name="intent",
            status=LoopStageStatus.SUCCEEDED,
            artifact_ref="outputs/intent_loop/run-1/intent.json",
        ),
        LoopStage(
            name="generator",
            status=LoopStageStatus.FAILED if failed else LoopStageStatus.SUCCEEDED,
            error="falha controlada" if failed else None,
        ),
    ]
    return LoopRecord(
        run_id=run_id,
        source_prompt="Reduza o recall.",
        seed=42,
        stages=tuple(stages),
    )


def test_intent_engine_requires_prompt():
    stderr = StringIO()

    exit_code = run_cli(argv=["--engine", "intent"], stderr=stderr)

    assert exit_code == 2
    assert "--prompt" in stderr.getvalue()


def test_intent_engine_forwards_arguments_and_reports_success():
    received = {}

    def fake_run_intent_loop(**kwargs):
        received.update(kwargs)
        return _record()

    stdout = StringIO()
    exit_code = run_cli(
        argv=[
            "--engine", "intent",
            "--prompt", "Reduza o recall.",
            "--model-id", "modelo-teste",
            "--generator-mode", "cached",
        ],
        stdout=stdout,
        run_intent_loop=fake_run_intent_loop,
    )

    assert exit_code == 0
    # Sem "attack": o ataque-base vem da própria intenção compilada, não de
    # um --attack redundante (ver IntentLoopOrchestrator.run).
    assert received == {
        "prompt": "Reduza o recall.",
        "model_id": "modelo-teste",
        "generator_mode": "cached",
    }
    output = stdout.getvalue()
    assert "run_id=run-1" in output
    assert "[OK] intent" in output
    assert "intent.json" in output


def test_intent_engine_reports_nonzero_exit_when_a_stage_failed():
    stdout = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _record(failed=True),
    )

    assert exit_code == 1
    assert "[FALHOU] generator" in stdout.getvalue()
    assert "falha controlada" in stdout.getvalue()


def test_intent_engine_turns_exception_into_nonzero_exit_code():
    stderr = StringIO()

    def failing(**_):
        raise RuntimeError("sem GROQ_API_KEY")

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stderr=stderr,
        run_intent_loop=failing,
    )

    assert exit_code == 1
    assert "sem GROQ_API_KEY" in stderr.getvalue()


def test_intent_engine_handles_keyboard_interrupt():
    stderr = StringIO()

    def interrupted(**_):
        raise KeyboardInterrupt

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stderr=stderr,
        run_intent_loop=interrupted,
    )

    assert exit_code == 130


def test_demo_engine_is_unaffected_by_the_new_flag():
    """--engine demo continua funcionando exatamente como antes (regressão)."""

    stdout = StringIO()
    exit_code = run_cli(argv=["--iterations", "1"], stdout=stdout)

    assert exit_code == 0
    assert "registro(s)" in stdout.getvalue()
