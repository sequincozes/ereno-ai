"""Testes do contrato que desacopla as interfaces do workflow."""

from io import StringIO

import pytest

from adversarial_ids.interfaces import cli
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


def test_demo_engine_rejects_a_detector_choice_with_a_clean_error():
    # Única combinação de flags inválida que o argparse não barra por choices:
    # precisa sair com exit code 2 e mensagem, não com traceback.
    stderr = StringIO()
    exit_code = run_cli(argv=["--engine", "demo", "--detector", "svm_rbf"], stderr=stderr)

    assert exit_code == 2
    assert "não se aplica a --engine demo" in stderr.getvalue()


def test_demo_rejection_does_not_depend_on_the_environment_default(monkeypatch):
    """A rejeição é sobre "passou a flag", não sobre "divergiu de DETECTOR_MODE".

    Sem a sentinela ``default=None`` os dois casos abaixo se invertem: com
    ``DETECTOR_MODE=svm_linear``, ``--detector random_forest`` seria recusado
    (por "divergir" do ambiente) e a omissão da flag passaria batida.
    """

    monkeypatch.setattr(cli, "DETECTOR_MODE", "svm_linear")

    stderr = StringIO()
    assert run_cli(argv=["--engine", "demo", "--detector", "random_forest"], stderr=stderr) == 2
    assert "não se aplica a --engine demo" in stderr.getvalue()

    # E omitir a flag continua sendo o caminho válido do demo.
    assert run_cli(argv=["--iterations", "1"], stdout=StringIO()) == 0


@pytest.mark.parametrize("engine_args", [["--engine", "live"], ["--engine", "intent", "--prompt", "p"]])
def test_invalid_detector_env_default_fails_cleanly_for_every_engine(monkeypatch, engine_args):
    # argparse só aplica `choices` a valores vindos de argv, nunca ao default —
    # um erro de digitação em DETECTOR_MODE chegaria cru ao núcleo e viraria
    # traceback (e de forma inconsistente entre live e intent).
    monkeypatch.setattr(cli, "DETECTOR_MODE", "bogus_detector")

    stderr = StringIO()
    assert run_cli(argv=engine_args, stderr=stderr) == 2
    assert "não é um detector conhecido" in stderr.getvalue()
