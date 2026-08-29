"""Testes do modo cacheado + seeds/fixtures/stubs (issue #3).

Rápidos e sem Java: validam a mecânica do cache, a presença do seed real, o
golden como IterationRecord (schemas #2) e o determinismo dos stubs.
"""

import json
from pathlib import Path

import pytest

from adversarial_ids.config.settings import BASELINE_DATASET_PATH, GOLDEN_HISTORY_PATH
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.domain import IterationRecord
from tests.fakes import FakeAnalyst, FakeStrategist

BASE_DIR = Path(__file__).resolve().parents[1]


def _cached_runner(cached_path: Path, out_dir: Path) -> GeneratorRunner:
    return GeneratorRunner(
        runtime_dir=out_dir / "runtime",
        attack_config_relative_path="config/attacks/x.json",
        output_dataset_path=out_dir / "unused.csv",
        run_command=["java", "-jar", "nao-deve-rodar.jar"],
        suggested_config_path=str(out_dir / "suggested.json"),
        cached_dataset_path=cached_path,
    )


# --------------------------------------------------------------------------- #
# Modo cacheado                                                                #
# --------------------------------------------------------------------------- #
def test_cached_generator_serves_seed_without_java(tmp_path):
    seed = tmp_path / "seed.csv"
    seed.write_text("a,b,class\n1,2,normal\n3,4,masquerade_fake_fault\n", encoding="utf-8")

    runner = _cached_runner(seed, tmp_path)
    assert runner.is_cached is True

    out = runner.generate_dataset({"attackType": "x"}, iteration=7)

    assert Path(out).exists()
    assert Path(out).name == "dataset_iteration_7.csv"
    assert Path(out).read_text(encoding="utf-8") == seed.read_text(encoding="utf-8")


def test_cached_generator_missing_seed_raises(tmp_path):
    runner = _cached_runner(tmp_path / "nao_existe.csv", tmp_path)

    with pytest.raises(FileNotFoundError):
        runner.generate_dataset({"attackType": "x"}, iteration=0)


# --------------------------------------------------------------------------- #
# Seeds versionados                                                            #
# --------------------------------------------------------------------------- #
def test_baseline_seed_exists_and_is_labeled():
    assert BASELINE_DATASET_PATH.exists(), "data/baseline_dataset.csv não versionado"

    with open(BASELINE_DATASET_PATH, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")

    assert "class" in header


def test_golden_history_validates_as_iteration_records():
    assert GOLDEN_HISTORY_PATH.exists(), "data/iteration_history.json não versionado"

    data = json.loads(GOLDEN_HISTORY_PATH.read_text(encoding="utf-8"))
    history = data["history"]

    assert len(history) >= 2
    assert history[0]["iteration"] == 0

    # cada registro precisa casar com o contrato IterationRecord (#2)
    for raw in history:
        record = IterationRecord.model_validate(raw)
        assert record.attack_config["attackType"] == "masquerade_fault"
        assert record.metrics.f1_score_attack is not None


# --------------------------------------------------------------------------- #
# Stubs determinísticos                                                        #
# --------------------------------------------------------------------------- #
def test_fake_strategist_is_deterministic_and_patchable():
    strategist = FakeStrategist()

    out_a = strategist.propose(1)
    out_b = strategist.propose(1)
    assert out_a == out_b  # determinismo

    assert out_a.persona in {"conservative", "aggressive"}

    patch = FakeStrategist.to_patch(out_a.model_dump())
    assert patch[0]["operation"] == "replace"
    assert patch[0]["field"] == out_a.changes[0].field
    assert patch[0]["new_value"] == out_a.changes[0].value


def test_fake_analyst_severity_and_shape():
    analyst = FakeAnalyst()

    strong = analyst.analyze(1, {"f1_score_attack": 1.0, "top_feature_importances": []})
    weak = analyst.analyze(2, {"f1_score_attack": 0.3, "top_feature_importances": []})

    assert strong["severity"] == "low"
    assert weak["severity"] == "high"
    assert weak["iteration"] == 2
    assert {"deceptive_features", "diagnosis", "mitigations", "severity"} <= weak.keys()
