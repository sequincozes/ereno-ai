"""Testes de shared/loop_record_store.py — persistência append-only (E3)."""

from __future__ import annotations

import json

import pytest

from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.shared.json_io import save_json
from adversarial_ids.shared.loop_record_store import (
    append_loop_record,
    load_loop_records,
)


def _record(run_id: str) -> LoopRecord:
    return LoopRecord(
        run_id=run_id,
        source_prompt="Reduza o recall variando a temporização da falha.",
        seed=42,
        stages=(LoopStage(name="intent", status=LoopStageStatus.SUCCEEDED),),
    )


def test_load_loop_records_returns_empty_list_when_file_missing(tmp_path):
    assert load_loop_records(tmp_path / "missing.json") == []


def test_append_loop_record_persists_and_round_trips(tmp_path):
    path = tmp_path / "loop_records.json"
    append_loop_record(path, _record("run-1"))

    records = load_loop_records(path)
    assert [r.run_id for r in records] == ["run-1"]


def test_append_loop_record_never_overwrites_previous_records(tmp_path):
    path = tmp_path / "loop_records.json"
    append_loop_record(path, _record("run-1"))
    append_loop_record(path, _record("run-2"))
    append_loop_record(path, _record("run-3"))

    records = load_loop_records(path)
    assert [r.run_id for r in records] == ["run-1", "run-2", "run-3"]


def test_append_loop_record_disk_format_is_a_plain_list_under_loop_records(tmp_path):
    path = tmp_path / "loop_records.json"
    append_loop_record(path, _record("run-1"))

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"loop_records"}
    assert raw["loop_records"][0]["run_id"] == "run-1"


def test_records_persisted_before_the_e10_lineage_fields_still_load(tmp_path):
    """Justifica manter schema_version=1 (E10): um registro sem round/
    parent_run_id (gravado antes desta entrega) continua carregando, com os
    defaults assumindo a rodada 1 sem campanha."""

    path = tmp_path / "loop_records.json"
    pre_e10 = _record("run-0").model_dump(mode="json")
    del pre_e10["round"]
    del pre_e10["parent_run_id"]
    path.write_text(
        json.dumps({"loop_records": [pre_e10]}), encoding="utf-8"
    )

    records = load_loop_records(path)

    assert len(records) == 1
    assert records[0].round == 1
    assert records[0].parent_run_id is None


# --------------------------------------------------------------------------- #
# Versão do contrato (v2 renomeou dois estágios)                              #
# --------------------------------------------------------------------------- #
def test_a_v1_ledger_is_refused_with_the_reason(tmp_path):
    """Sem isto o erro seria "'ereno' não é um LoopStageName" — verdade que não
    ajuda ninguém a perceber que o arquivo é de outra versão do contrato."""

    path = tmp_path / "loop_records.json"
    save_json(
        path,
        {"loop_records": [{"schema_version": 1, "run_id": "antigo", "stages": []}]},
    )

    with pytest.raises(ValueError, match="schema_version"):
        load_loop_records(path)


def test_the_refusal_says_where_the_file_is_and_what_changed(tmp_path):
    path = tmp_path / "loop_records.json"
    save_json(path, {"loop_records": [{"schema_version": 1, "run_id": "antigo"}]})

    with pytest.raises(ValueError) as excinfo:
        load_loop_records(path)

    message = str(excinfo.value)
    assert "loop_records.json" in message
    assert "compiler" in message and "generator" in message
    # Ledgers são artefatos descartáveis: a mensagem precisa dizer isso, ou
    # alguém vai procurar uma migração que não existe.
    assert "init_state" in message
