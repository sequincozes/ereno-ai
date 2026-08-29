"""Testes do AdversarialWorkflow encadeando strategist → analyst (issue #16).

Roda o loop **ponta a ponta com os stubs** (``FakeStrategist`` / ``FakeAnalyst``,
#3) sobre um seed cacheado minúsculo — sem chave da Groq e sem Java. Cobre a DoD:
"encadeia strategist → analyst por iteração" e "roda ponta a ponta com stubs".
"""

import json
from pathlib import Path

import pytest

from adversarial_ids.agents.orchestrator import (
    AdversarialWorkflow,
    changes_to_patch,
)
from adversarial_ids.core.experiment_memory import ExperimentMemory
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain import IterationRecord
from tests.fakes import FakeAnalyst, FakeStrategist

BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_ATTACK_JSON = BASE_DIR / "inputs" / "uc03_masquerade_fault.json"


def _tiny_labeled_seed(path: Path) -> Path:
    """CSV rotulado pequeno (2 classes, com sinal) para treinar o RF rápido."""

    rows = ["f1,f2,class"]
    for i in range(30):
        # normal: f1 baixo; ataque: f1 alto — separável, RF treina em ms.
        rows.append(f"{i % 5},{i % 3},normal")
        rows.append(f"{100 + i % 5},{10 + i % 3},masquerade_fake_fault")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _cached_workflow(tmp_path: Path, iterations_memory=None) -> AdversarialWorkflow:
    seed = _tiny_labeled_seed(tmp_path / "seed.csv")

    generator = GeneratorRunner(
        runtime_dir=tmp_path / "runtime",
        attack_config_relative_path="config/attacks/x.json",
        output_dataset_path=tmp_path / "unused.csv",
        run_command=["java", "-jar", "nao-deve-rodar.jar"],
        suggested_config_path=str(tmp_path / "suggested.json"),
        cached_dataset_path=seed,  # força modo cacheado (sem Java)
    )

    baseline_attack = json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))

    return AdversarialWorkflow(
        strategist=FakeStrategist(),
        analyst=FakeAnalyst(),
        generator=generator,
        evaluator=IdsEvaluator(drop_cb_status=False),
        baseline_attack_config=baseline_attack,
        memory=iterations_memory,
        save_path=tmp_path / "iteration_history.json",
    )


# --------------------------------------------------------------------------- #
# changes_to_patch                                                            #
# --------------------------------------------------------------------------- #
def test_changes_to_patch_maps_contract_to_json_patch():
    patch = changes_to_patch([{"field": "fault.prob", "value": 0.4}])

    assert patch == [
        {
            "operation": "replace",
            "field": "fault.prob",
            "old_value": None,
            "new_value": 0.4,
            "reason": "strategist_change",
        }
    ]


def test_changes_to_patch_ignores_malformed_changes():
    assert changes_to_patch([{"field": "fault.prob"}, {"value": 1}, {}]) == []


# --------------------------------------------------------------------------- #
# Loop ponta a ponta com stubs                                                #
# --------------------------------------------------------------------------- #
def test_workflow_runs_end_to_end_with_stubs(tmp_path):
    workflow = _cached_workflow(tmp_path)

    memory = workflow.run(iterations=2)

    records = memory.get_records()
    # baseline (0) + 2 iterações
    assert [r.iteration for r in records] == [0, 1, 2]

    # baseline: sem estrategista, com analista; iterações: ambos presentes
    assert records[0].strategist_output is None
    assert records[0].analyst_output is not None
    for record in records[1:]:
        assert record.strategist_output is not None
        assert record.analyst_output is not None
        # analyst_output é tipado (AnalystOutput), acesso por atributo.
        assert record.analyst_output.severity is not None
        assert record.metrics.f1_score_attack is not None


def test_workflow_persists_golden_shaped_history(tmp_path):
    workflow = _cached_workflow(tmp_path)
    workflow.run(iterations=1)

    saved = tmp_path / "iteration_history.json"
    assert saved.exists()

    data = json.loads(saved.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"history"}
    # cada registro persistido revalida como IterationRecord (#2)
    for raw in data["history"]:
        IterationRecord.model_validate(raw)


def test_workflow_chains_strategist_changes_into_config(tmp_path):
    """A jogada do Estrategista deve alterar o attack_config persistido."""

    workflow = _cached_workflow(tmp_path)
    memory = workflow.run(iterations=1)

    baseline_record, iter1 = memory.get_records()
    # FakeStrategist na iteração 1 mexe em fault.prob (0.6 -> 0.75)
    assert baseline_record.attack_config["fault"]["prob"] == 0.6
    assert iter1.attack_config["fault"]["prob"] == 0.75
    assert iter1.metrics.config_changed is True


def test_workflow_accepts_injected_memory(tmp_path):
    memory = ExperimentMemory()
    workflow = _cached_workflow(tmp_path, iterations_memory=memory)

    returned = workflow.run(iterations=1)

    assert returned is memory
    assert len(memory) == 2
