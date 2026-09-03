"""Testes do gate de integração do ERENO (épico E4).

Cobre o critério de aceite D6-14: "trace novo tem hash, classes e volume
válidos" — nenhum ``DatasetBundle`` deve sair de ``build_dataset_bundle``
sem passar pelos três checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adversarial_ids.core.dataset_bundle_builder import (
    DatasetGateError,
    build_dataset_bundle,
)
from adversarial_ids.domain.dataset_bundle import DatasetBundle


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _valid_trace(path: Path, *, attack_rows: int = 6, normal_rows: int = 6) -> Path:
    rows = ["f1,f2,class"]
    for i in range(attack_rows):
        rows.append(f"{i},1,attack_label")
    for i in range(normal_rows):
        rows.append(f"{i},0,normal")
    return _write_csv(path, rows)


def test_build_dataset_bundle_approves_a_valid_trace(tmp_path):
    trace = _valid_trace(tmp_path / "trace.csv")

    bundle = build_dataset_bundle(
        trace,
        lineage_run_id="run-1",
        expected_attack_label="attack_label",
    )

    assert isinstance(bundle, DatasetBundle)
    assert bundle.trace_path == str(trace)
    assert bundle.classes == ("attack_label", "normal")
    assert bundle.class_counts == {"attack_label": 6, "normal": 6}
    assert len(bundle.content_hash) == 64
    assert bundle.lineage_run_id == "run-1"
    assert bundle.columns == ("f1", "f2", "class")


def test_build_dataset_bundle_is_deterministic_for_the_same_content(tmp_path):
    trace_a = _valid_trace(tmp_path / "a.csv")
    trace_b = _valid_trace(tmp_path / "b.csv")

    bundle_a = build_dataset_bundle(
        trace_a, lineage_run_id="run-1", expected_attack_label="attack_label"
    )
    bundle_b = build_dataset_bundle(
        trace_b, lineage_run_id="run-2", expected_attack_label="attack_label"
    )

    assert bundle_a.content_hash == bundle_b.content_hash


def test_build_dataset_bundle_rejects_missing_file(tmp_path):
    with pytest.raises(DatasetGateError, match="não encontrado"):
        build_dataset_bundle(
            tmp_path / "missing.csv",
            lineage_run_id="run-1",
            expected_attack_label="attack_label",
        )


def test_build_dataset_bundle_rejects_empty_file(tmp_path):
    trace = tmp_path / "empty.csv"
    trace.write_text("", encoding="utf-8")

    with pytest.raises(DatasetGateError, match="vazio"):
        build_dataset_bundle(
            trace, lineage_run_id="run-1", expected_attack_label="attack_label"
        )


def test_build_dataset_bundle_rejects_missing_attack_class(tmp_path):
    trace = _write_csv(
        tmp_path / "trace.csv",
        ["f1,class"] + [f"{i},normal" for i in range(10)],
    )

    with pytest.raises(DatasetGateError, match="ausente"):
        build_dataset_bundle(
            trace, lineage_run_id="run-1", expected_attack_label="attack_label"
        )


def test_build_dataset_bundle_rejects_missing_normal_class(tmp_path):
    trace = _write_csv(
        tmp_path / "trace.csv",
        ["f1,class"] + [f"{i},attack_label" for i in range(10)],
    )

    with pytest.raises(DatasetGateError, match="'normal' ausente"):
        build_dataset_bundle(
            trace, lineage_run_id="run-1", expected_attack_label="attack_label"
        )


def test_build_dataset_bundle_rejects_volume_below_floor(tmp_path):
    trace = _valid_trace(tmp_path / "trace.csv", attack_rows=2, normal_rows=10)

    with pytest.raises(DatasetGateError, match="abaixo do piso"):
        build_dataset_bundle(
            trace,
            lineage_run_id="run-1",
            expected_attack_label="attack_label",
            min_attack_rows=5,
        )


def test_build_dataset_bundle_finds_the_class_column_case_insensitively(tmp_path):
    trace = _write_csv(
        tmp_path / "trace.csv",
        ["f1,Class"] + [f"{i},attack_label" for i in range(6)] + [f"{i},normal" for i in range(6)],
    )

    bundle = build_dataset_bundle(
        trace, lineage_run_id="run-1", expected_attack_label="attack_label"
    )
    assert bundle.class_counts == {"attack_label": 6, "normal": 6}
