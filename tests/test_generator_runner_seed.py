"""Propagação da semente determinística ao action config (Fase 2.0).

O ERENO lê ``randomSeed`` do topo do action config e a aplica antes de gerar
(``CreateAttackDatasetAction``); com ela, a mesma config produz os mesmos bytes
— o piloto E0 media o oposto. Aqui testamos só o lado Python: que
``GeneratorRunner`` escreve/remove ``randomSeed`` ao selecionar o segmento, sem
depender do JAR nem do Java.
"""

from __future__ import annotations

import json

from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.shared.json_io import load_json


def _runner(tmp_path, *, random_seed):
    runtime = tmp_path / "runtime"
    action_rel = "config/actions/attack.json"
    action_path = runtime / action_rel
    action_path.parent.mkdir(parents=True)
    action_path.write_text(
        json.dumps(
            {
                "attackSegments": [
                    {"name": "uc03_masquerade_fault", "enabled": False,
                     "attackConfig": "config/attacks/uc03.json"},
                    {"name": "uc01_random_replay", "enabled": True,
                     "attackConfig": "config/attacks/uc01.json"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return GeneratorRunner(
        runtime_dir=runtime,
        output_dataset_path=runtime / "out.csv",
        run_command=["java", "-jar", "g.jar"],
        suggested_config_path=str(runtime / "attack_config.json"),
        action_config_relative_path=action_rel,
        segment_name="uc03_masquerade_fault",
        random_seed=random_seed,
    ), action_path


def test_the_seed_is_written_to_the_action_config_when_set(tmp_path):
    runner, action_path = _runner(tmp_path, random_seed=123)

    runner._select_segment_and_resolve_attack_path()

    action = load_json(action_path)
    assert action["randomSeed"] == 123
    # E o segmento escolhido continua sendo o único habilitado.
    enabled = [s["name"] for s in action["attackSegments"] if s["enabled"]]
    assert enabled == ["uc03_masquerade_fault"]


def test_no_seed_leaves_no_randomseed_key(tmp_path):
    """Sem seed, a chave não é inventada — o comportamento anterior é preservado."""

    runner, action_path = _runner(tmp_path, random_seed=None)

    runner._select_segment_and_resolve_attack_path()

    assert "randomSeed" not in load_json(action_path)


def test_a_stale_seed_from_a_previous_run_is_removed(tmp_path):
    """Um action config versionado não pode guardar a seed de uma execução
    anterior: sem seed nesta rodada, a chave antiga sai."""

    runner, action_path = _runner(tmp_path, random_seed=None)
    action = load_json(action_path)
    action["randomSeed"] = 999
    action_path.write_text(json.dumps(action), encoding="utf-8")

    runner._select_segment_and_resolve_attack_path()

    assert "randomSeed" not in load_json(action_path)


def test_the_seed_default_is_none_so_construction_is_unchanged(tmp_path):
    runner, _ = _runner(tmp_path, random_seed=None)
    assert runner.random_seed is None
