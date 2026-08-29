"""Testes do contrato que desacopla as interfaces do workflow."""

from io import StringIO

from adversarial_ids.interfaces.cli import run_cli
from adversarial_ids.interfaces.experiment_runner import (
    CachedHistoryRunner,
    WorkflowAdapter,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.options = None

    def run(
        self,
        *,
        iterations: int,
        model_id: str,
        generator_mode: str,
        attack: str = "masquerade_fault",
    ):
        self.options = (iterations, model_id, generator_mode)
        return CachedHistoryRunner().run(
            iterations=iterations,
            model_id=model_id,
            generator_mode=generator_mode,
            attack=attack,
        )


class FailingRunner:
    def run(self, **_):
        raise RuntimeError("falha controlada")


def test_cli_forwards_arguments_and_reports_success():
    runner = RecordingRunner()
    stdout = StringIO()

    exit_code = run_cli(
        runner,
        ["--iterations", "1", "--model-id", "modelo-teste", "--generator-mode", "cached"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert runner.options == (1, "modelo-teste", "cached")
    assert "2 registro(s)" in stdout.getvalue()


def test_cli_turns_runner_error_into_nonzero_exit_code():
    stderr = StringIO()

    exit_code = run_cli(FailingRunner(), [], stderr=stderr)

    assert exit_code == 1
    assert "falha controlada" in stderr.getvalue()


def test_cached_runner_is_explicitly_cached_and_respects_iteration_limit():
    runner = CachedHistoryRunner()

    records = runner.run(iterations=2, model_id="ignorado", generator_mode="cached")

    assert [record.iteration for record in records] == [0, 1, 2]


def test_workflow_adapter_normalizes_records_to_domain_contract():
    raw_records = [
        record.model_dump(mode="json")
        for record in CachedHistoryRunner().run(
            iterations=0,
            model_id="modelo-teste",
            generator_mode="cached",
        )
    ]
    received = {}

    def workflow(**options):
        received.update(options)
        return raw_records

    records = WorkflowAdapter(workflow).run(
        iterations=3,
        model_id="modelo-real",
        generator_mode="cached",
    )

    assert records[0].iteration == 0
    assert received == {
        "iterations": 3,
        "model_id": "modelo-real",
        "generator_mode": "cached",
        "attack": "masquerade_fault",
    }
