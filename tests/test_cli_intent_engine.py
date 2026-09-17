"""Testes de --engine intent na CLI (feature flag do E3/E4).

Injeta ``run_intent_loop`` para não depender de GROQ_API_KEY/Java — mesmo
espírito de ``RecordingRunner``/``FailingRunner`` em ``test_cli_runner.py``.
"""

from __future__ import annotations

from io import StringIO

from adversarial_ids.config.settings import INTENT_LOOP_DEFAULT_ROUNDS
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.interfaces.cli import run_cli


def _record(
    *, run_id: str = "run-1", failed: bool = False, round: int = 1,
    parent_run_id: str | None = None,
) -> LoopRecord:
    stages = [
        LoopStage(
            name="intent",
            status=LoopStageStatus.SUCCEEDED,
            duration_seconds=0.75,
            artifact_ref="outputs/intent_loop/run-1/intent.json",
        ),
        LoopStage(
            name="generator",
            status=LoopStageStatus.FAILED if failed else LoopStageStatus.SUCCEEDED,
            duration_seconds=1.5,
            error="falha controlada" if failed else None,
        ),
    ]
    return LoopRecord(
        run_id=run_id,
        source_prompt="Reduza o recall.",
        seed=42,
        stages=tuple(stages),
        total_duration_seconds=2.25,
        total_tokens=1540,
        cost_usd=0.0031,
        round=round,
        parent_run_id=parent_run_id,
    )


def _records(*, count: int = 1, failed: bool = False) -> tuple[LoopRecord, ...]:
    records: list[LoopRecord] = []
    parent_run_id: str | None = None
    for index in range(1, count + 1):
        run_id = f"run-{index}"
        records.append(
            _record(
                run_id=run_id,
                failed=failed and index == count,
                round=index,
                parent_run_id=parent_run_id,
            )
        )
        parent_run_id = run_id
    return tuple(records)


def test_intent_engine_requires_prompt():
    stderr = StringIO()

    exit_code = run_cli(argv=["--engine", "intent"], stderr=stderr)

    assert exit_code == 2
    assert "--prompt" in stderr.getvalue()


def test_intent_engine_forwards_arguments_and_reports_success():
    received = {}

    def fake_run_intent_loop(**kwargs):
        received.update(kwargs)
        return _records()

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
        "rounds": INTENT_LOOP_DEFAULT_ROUNDS,
        # E8: o detector default vai junto, para que a campanha registre qual
        # modelo rodou sem depender de uma env var lida lá dentro.
        "detector": "random_forest",
    }
    output = stdout.getvalue()
    assert "run_id=run-1" in output
    assert "[OK]" in output
    assert "intent" in output
    assert "intent.json" in output


def test_intent_engine_forwards_the_round_count_from_the_flag():
    received = {}

    def fake_run_intent_loop(**kwargs):
        received.update(kwargs)
        return _records(count=3)

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x", "--rounds", "3"],
        stdout=StringIO(),
        run_intent_loop=fake_run_intent_loop,
    )

    assert exit_code == 0
    assert received["rounds"] == 3


def test_intent_engine_rejects_a_non_positive_round_count():
    stderr = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x", "--rounds", "0"],
        stderr=stderr,
    )

    assert exit_code == 2
    assert "--rounds" in stderr.getvalue()


def test_intent_engine_ignores_the_legacy_iterations_flag():
    received = {}

    def fake_run_intent_loop(**kwargs):
        received.update(kwargs)
        return _records()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x", "--iterations", "30"],
        stdout=StringIO(),
        run_intent_loop=fake_run_intent_loop,
    )

    assert exit_code == 0
    assert "iterations" not in received


def test_intent_engine_prints_one_summary_per_round():
    stdout = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _records(count=2),
    )

    assert exit_code == 0
    output = stdout.getvalue()
    assert "run_id=run-1" in output
    assert "run_id=run-2" in output
    assert "Rodada 1/2" in output
    assert "Rodada 2/2" in output


def test_intent_engine_reports_nonzero_exit_when_a_stage_failed():
    stdout = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _records(failed=True),
    )

    assert exit_code == 1
    output = stdout.getvalue()
    assert "[FALHOU]" in output
    assert "generator" in output
    assert "falha controlada" in output
    assert "falha controlada" in stdout.getvalue()


def test_intent_engine_reports_nonzero_exit_when_any_round_has_a_failed_stage():
    stdout = StringIO()

    exit_code = run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _records(count=2, failed=True),
    )

    assert exit_code == 1


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


def test_intent_engine_forwards_the_chosen_detector():
    received = {}

    def fake_run_intent_loop(**kwargs):
        received.update(kwargs)
        return _records()

    exit_code = run_cli(
        argv=[
            "--engine", "intent",
            "--prompt", "Reduza o recall.",
            "--detector", "svm_linear",
        ],
        stdout=StringIO(),
        run_intent_loop=fake_run_intent_loop,
    )

    assert exit_code == 0
    assert received["detector"] == "svm_linear"


def test_intent_summary_prints_the_duration_of_each_stage_and_of_the_run():
    """O critério de pronto do E11 cobra duração, e o resumo media sem mostrar."""

    stdout = StringIO()

    run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _records(),
    )

    output = stdout.getvalue()
    assert "0.75s" in output
    assert "1.50s" in output
    assert "total: 2.25s" in output


def test_intent_summary_prints_the_consumption_when_the_agents_reported_it():
    """"Execução cabe no orçamento definido" exige que o gasto seja visível."""

    stdout = StringIO()

    run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: _records(),
    )

    output = stdout.getvalue()
    assert "1540 tokens" in output
    assert "US$ 0.0031" in output


def test_a_run_with_no_reported_consumption_says_nothing_instead_of_zero():
    """Um "0 tokens" impresso porque ninguém contou seria pior que a ausência."""

    stdout = StringIO()
    silent = LoopRecord(
        run_id="run-1",
        source_prompt="Reduza o recall.",
        seed=42,
        stages=(LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),),
        total_duration_seconds=1.0,
    )

    run_cli(
        argv=["--engine", "intent", "--prompt", "x"],
        stdout=stdout,
        run_intent_loop=lambda **_: (silent,),
    )

    output = stdout.getvalue()
    assert "total: 1.00s" in output
    assert "tokens" not in output
    assert "US$" not in output
