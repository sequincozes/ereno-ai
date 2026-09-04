"""Timeout/retry robustness of GeneratorRunner's JAR subprocess calls.

Covers the gap flagged for the D15-24 window of the roadmap ("loop confiável
e multi-ataque"): the JAR call had no timeout and no retry, so a hang or a
transient crash propagated straight up with no recovery. See
``core/generator_runner.py::GeneratorRunner._run_generator_command``.
"""

import json
import subprocess

import pytest

from adversarial_ids.core import generator_runner as generator_runner_module
from adversarial_ids.core.generator_runner import GeneratorRunner


def _runner(tmp_path, *, max_retries=0, timeout_seconds=None, retry_backoff_seconds=0.0):
    runtime = tmp_path / "runtime"
    training_path = runtime / "target/training/training.csv"

    # Action config with a benign dataset that already exists, so
    # `_ensure_benign_dataset` returns early and every subprocess.run call
    # observed by these tests is the main generator command — the one whose
    # timeout/retry behavior is under test here (already covered on its own
    # in test_generator_runner_benign.py).
    action_path = runtime / "config/actions/attack.json"
    benign_path = runtime / "target/benign_data/existing.csv"
    action_path.parent.mkdir(parents=True)
    benign_path.parent.mkdir(parents=True)
    benign_path.write_text("feature,class\n1,normal\n", encoding="utf-8")
    action_path.write_text(
        json.dumps({"input": {"benignDataPath": "target/benign_data/existing.csv"}}),
        encoding="utf-8",
    )

    return GeneratorRunner(
        runtime_dir=runtime,
        output_dataset_path=training_path,
        run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
        suggested_config_path=str(tmp_path / "outputs/suggested.json"),
        attack_config_relative_path="config/attacks/attack.json",
        action_config_relative_path="config/actions/attack.json",
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _no_sleep(monkeypatch):
    """Retries in these tests don't need real backoff delay."""
    monkeypatch.setattr(generator_runner_module.time, "sleep", lambda _seconds: None)


def test_default_construction_keeps_single_attempt_no_timeout(tmp_path):
    runner = GeneratorRunner(
        runtime_dir=tmp_path / "runtime",
        output_dataset_path=tmp_path / "runtime/target/training/training.csv",
        run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
        suggested_config_path=str(tmp_path / "outputs/suggested.json"),
    )
    assert runner.max_retries == 0
    assert runner.timeout_seconds is None


def test_negative_max_retries_rejected(tmp_path):
    with pytest.raises(ValueError):
        GeneratorRunner(
            runtime_dir=tmp_path / "runtime",
            output_dataset_path=tmp_path / "runtime/target/training/training.csv",
            run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
            suggested_config_path=str(tmp_path / "outputs/suggested.json"),
            max_retries=-1,
        )


def test_single_attempt_failure_raises_immediately_when_retries_disabled(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="out", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _no_sleep(monkeypatch)
    runner = _runner(tmp_path, max_retries=0)

    with pytest.raises(RuntimeError, match="falhou em todas as 1 tentativa"):
        runner.generate_dataset({"attackType": "test"}, iteration=1)

    assert len(calls) == 1


def test_retries_after_transient_process_failure_then_succeeds(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="flaky")
        training_path = tmp_path / "runtime/target/training/training.csv"
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.write_text("feature,class\n1,attack\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _no_sleep(monkeypatch)
    runner = _runner(tmp_path, max_retries=1)

    result = runner.generate_dataset({"attackType": "test"}, iteration=1)

    assert len(calls) == 2
    from pathlib import Path

    assert Path(result).exists()


def test_all_attempts_failing_raises_after_exhausting_retries(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="out", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _no_sleep(monkeypatch)
    runner = _runner(tmp_path, max_retries=2)

    with pytest.raises(RuntimeError, match="falhou em todas as 3 tentativa"):
        runner.generate_dataset({"attackType": "test"}, iteration=1)

    assert len(calls) == 3


def test_timeout_retries_then_succeeds(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))
        training_path = tmp_path / "runtime/target/training/training.csv"
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.write_text("feature,class\n1,attack\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _no_sleep(monkeypatch)
    runner = _runner(tmp_path, max_retries=1, timeout_seconds=5.0)

    from pathlib import Path

    result = runner.generate_dataset({"attackType": "test"}, iteration=1)
    assert len(calls) == 2
    assert Path(result).exists()


def test_all_attempts_timing_out_raises_timeout_error(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    _no_sleep(monkeypatch)
    runner = _runner(tmp_path, max_retries=1, timeout_seconds=5.0)

    with pytest.raises(TimeoutError, match="expirou em todas as 2 tentativa"):
        runner.generate_dataset({"attackType": "test"}, iteration=1)

    assert len(calls) == 2


def test_retry_backoff_grows_with_attempt_number(monkeypatch, tmp_path):
    sleeps = []
    monkeypatch.setattr(
        generator_runner_module.time, "sleep", lambda seconds: sleeps.append(seconds)
    )

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = _runner(tmp_path, max_retries=2, retry_backoff_seconds=2.0)

    with pytest.raises(RuntimeError):
        runner.generate_dataset({"attackType": "test"}, iteration=1)

    # 3 tentativas -> 2 pausas, crescendo com o número da tentativa que falhou.
    assert sleeps == [2.0, 4.0]
