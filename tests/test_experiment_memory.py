"""Testes do ExperimentMemory persistindo IterationRecord (issue #17).

Cobrem os três caminhos do núcleo (M3):

- ``add_iteration`` (adaptador frouxo usado por ``interfaces/cli.py``) monta um
  ``IterationRecord`` válido a partir de dicts;
- ``save``/``load`` fazem round-trip no formato do golden ``{"history": [...]}``;
- ``get_history`` continua compatível com quem lê ``item["iteration"]`` e
  ``item["metrics"]`` (ex.: ``StrategistAgent._compact_history``).
"""

import json
from pathlib import Path

import pytest

from adversarial_ids.core.experiment_memory import ExperimentMemory
from adversarial_ids.domain import AttackConfig, IterationRecord, Metrics

BASE_DIR = Path(__file__).resolve().parents[1]
BASELINE_ATTACK_JSON = BASE_DIR / "inputs" / "uc03_masquerade_fault.json"


def _attack_json() -> dict:
    return json.loads(BASELINE_ATTACK_JSON.read_text(encoding="utf-8"))


def _variant_metrics() -> dict:
    return {
        "evaluation_type": "variant_external_test",
        "f1_score_attack": 0.91,
        "recall_attack": 0.88,
        "precision_attack": 0.94,
        "attack_count": 300,
        "normal_count": 700,
        "tp": 264,
        "fp": 17,
        "fn": 36,
        "tn": 683,
        "config_changed": True,
        "degenerate_variant": False,
    }


# --------------------------------------------------------------------------- #
# add_iteration (caminho da CLI)                                              #
# --------------------------------------------------------------------------- #
def test_add_iteration_builds_valid_iteration_record():
    memory = ExperimentMemory()

    record = memory.add_iteration(
        iteration=1,
        attack_json=_attack_json(),
        metrics=_variant_metrics(),
        llm_response="ALTERACOES_APLICAVEIS\nfault.prob: 0.4\n",
    )

    assert isinstance(record, IterationRecord)
    assert len(memory) == 1
    assert record.attack_config["fault"]["prob"] == 0.6
    assert record.metrics.f1_score_attack == 0.91
    # resposta crua da LLM preservada no reasoning do StrategistOutput
    assert record.strategist_output is not None
    assert record.strategist_output.reasoning == (
        "ALTERACOES_APLICAVEIS\nfault.prob: 0.4\n"
    )


def test_add_record_accepts_prebuilt_typed_record():
    memory = ExperimentMemory()

    record = IterationRecord(
        iteration=0,
        attack_config=AttackConfig.model_validate(_attack_json()),
        metrics=Metrics.model_validate(_variant_metrics()),
    )
    memory.add_record(record)

    assert memory.get_records() == [record]


def test_add_iteration_stores_attack_config_as_is():
    """A memória guarda o attack_config como dado de schema variável (multi-ataque).

    A validação/clamp por campo acontece no workflow (``validate_and_clamp_attack_config``),
    não nesta camada — então ``add_iteration`` aceita a configuração de qualquer
    tipo de ataque e a preserva fielmente no registro.
    """
    memory = ExperimentMemory()
    other_attack = {
        "attackType": "grayhole",
        "enabled": True,
        "dropRate": {"min": 0.25, "max": 0.35},
        "burstDropProb": 0.2,
    }

    record = memory.add_iteration(
        iteration=1, attack_json=other_attack, metrics=_variant_metrics()
    )

    assert record.attack_config == other_attack


# --------------------------------------------------------------------------- #
# get_history — compatível com StrategistAgent._compact_history               #
# --------------------------------------------------------------------------- #
def test_get_history_is_json_serializable_and_readable():
    memory = ExperimentMemory()
    memory.add_iteration(1, _attack_json(), _variant_metrics(), llm_response="x")

    history = memory.get_history()

    assert json.dumps(history)  # não levanta
    item = history[0]
    # as chaves que o Estrategista lê do histórico
    assert item["iteration"] == 1
    assert item["metrics"]["f1_score_attack"] == 0.91


# --------------------------------------------------------------------------- #
# save / load — round-trip no formato do golden                              #
# --------------------------------------------------------------------------- #
def test_save_writes_golden_shape(tmp_path):
    memory = ExperimentMemory()
    memory.add_iteration(0, _attack_json(), _variant_metrics())

    out = tmp_path / "iteration_history.json"
    memory.save(out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"history"}
    assert data["history"][0]["iteration"] == 0
    # cada registro persistido revalida como IterationRecord
    IterationRecord.model_validate(data["history"][0])


def test_load_round_trips_saved_history(tmp_path):
    memory = ExperimentMemory()
    memory.add_iteration(0, _attack_json(), _variant_metrics())
    memory.add_iteration(1, _attack_json(), _variant_metrics(), llm_response="y")

    out = tmp_path / "iteration_history.json"
    memory.save(out)

    reloaded = ExperimentMemory.load(out)

    assert len(reloaded) == 2
    assert reloaded.get_history() == memory.get_history()


def test_load_missing_file_returns_empty_memory(tmp_path):
    reloaded = ExperimentMemory.load(tmp_path / "nao_existe.json")

    assert isinstance(reloaded, ExperimentMemory)
    assert len(reloaded) == 0
