import json
import subprocess
from pathlib import Path

from adversarial_ids.core.generator_runner import GeneratorRunner


def test_jar_mode_generates_missing_benign_dataset(
    monkeypatch, tmp_path, capsys
):
    runtime = tmp_path / "runtime"
    action_path = runtime / "config/actions/attack.json"
    benign_config_path = runtime / "config/actions/benign.json"
    attack_config_path = runtime / "config/attacks/attack.json"
    training_path = runtime / "target/training/training.csv"

    action_path.parent.mkdir(parents=True)
    action_path.write_text(
        json.dumps(
            {
                "input": {
                    "benignDataPath": "target/benign_data/missing.csv",
                }
            }
        ),
        encoding="utf-8",
    )
    benign_config_path.write_text("{}", encoding="utf-8")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "config/actions/benign.json":
            generated = (
                runtime / "target/benign_data/123_5%fault_benign_data.csv"
            )
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("feature,class\n1,normal\n", encoding="utf-8")
        else:
            training_path.parent.mkdir(parents=True, exist_ok=True)
            training_path.write_text("feature,class\n1,attack\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = GeneratorRunner(
        runtime_dir=runtime,
        output_dataset_path=training_path,
        run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
        suggested_config_path=str(tmp_path / "outputs/suggested.json"),
        attack_config_relative_path="config/attacks/attack.json",
        action_config_relative_path="config/actions/attack.json",
        benign_action_config_relative_path="config/actions/benign.json",
    )

    result = runner.generate_dataset({"attackType": "test"}, iteration=1)

    assert len(calls) == 2
    assert calls[0][-1] == "config/actions/benign.json"
    assert calls[1][-1] == "config/actions/attack.json"
    assert Path(result).exists()
    updated_action = json.loads(action_path.read_text(encoding="utf-8"))
    assert updated_action["input"]["benignDataPath"] == (
        "target/benign_data/123_5%fault_benign_data.csv"
    )
    output = capsys.readouterr().out
    assert "[GEN:BENIGN] AUSENTE" in output
    assert "[GEN:BENIGN] CRIADO" in output


def test_jar_mode_reuses_existing_benign_dataset(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    action_path = runtime / "config/actions/attack.json"
    benign_path = runtime / "target/benign_data/existing.csv"
    training_path = runtime / "target/training/training.csv"
    attack_config_path = runtime / "config/attacks/attack.json"

    action_path.parent.mkdir(parents=True)
    benign_path.parent.mkdir(parents=True)
    benign_path.write_text("feature,class\n1,normal\n", encoding="utf-8")
    action_path.write_text(
        json.dumps({"input": {"benignDataPath": "target/benign_data/existing.csv"}}),
        encoding="utf-8",
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.write_text("feature,class\n1,attack\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = GeneratorRunner(
        runtime_dir=runtime,
        output_dataset_path=training_path,
        run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
        suggested_config_path=str(tmp_path / "outputs/suggested.json"),
        attack_config_relative_path=str(attack_config_path.relative_to(runtime)),
        action_config_relative_path="config/actions/attack.json",
        benign_action_config_relative_path="config/actions/benign.json",
    )

    runner.generate_dataset({"attackType": "test"}, iteration=2)

    assert calls == [
        ["java", "-jar", "generator.jar", "config/actions/attack.json"]
    ]


def test_jar_mode_extracts_normal_rows_from_seed(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    action_path = runtime / "config/actions/attack.json"
    benign_path = runtime / "target/benign_data/extracted.csv"
    training_path = runtime / "target/training/training.csv"
    seed_path = tmp_path / "baseline.csv"

    action_path.parent.mkdir(parents=True)
    action_path.write_text(
        json.dumps({"input": {"benignDataPath": "target/benign_data/extracted.csv"}}),
        encoding="utf-8",
    )
    seed_path.write_text(
        "feature,class\n1,normal\n2,attack\n3,normal\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.write_text("feature,class\n1,attack\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = GeneratorRunner(
        runtime_dir=runtime,
        output_dataset_path=training_path,
        run_command=["java", "-jar", "generator.jar", "config/actions/attack.json"],
        suggested_config_path=str(tmp_path / "outputs/suggested.json"),
        attack_config_relative_path="config/attacks/attack.json",
        action_config_relative_path="config/actions/attack.json",
        benign_seed_path=seed_path,
    )

    runner.generate_dataset({"attackType": "test"}, iteration=3)

    assert calls == [
        ["java", "-jar", "generator.jar", "config/actions/attack.json"]
    ]
    assert benign_path.read_text(encoding="utf-8") == (
        "feature,class\n1,normal\n3,normal\n"
    )
