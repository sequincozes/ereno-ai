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


def test_build_dataset_bundle_rejects_attack_class_too_rare_to_measure(tmp_path):
    """Piso proporcional: o absoluto passa, a prevalência não.

    Forma real observada numa variante de ``injection`` gerada pelo JAR —
    27 linhas de ataque contra ~50k normais. Antes do piso de prevalência o
    trace era aprovado e o detector reportava recall sobre 27 amostras.
    """

    trace = _valid_trace(tmp_path / "trace.csv", attack_rows=27, normal_rows=49_998)

    with pytest.raises(DatasetGateError, match="Prevalência") as excinfo:
        build_dataset_bundle(
            trace,
            lineage_run_id="run-1",
            expected_attack_label="attack_label",
        )

    message = str(excinfo.value)
    # O piso absoluto não é o que reprovou — a mensagem precisa dizer isso, ou
    # quem lê vai mexer em min_attack_rows e não resolver nada.
    assert "27/50025" in message
    assert "dataset, não a intenção" in message


def test_build_dataset_bundle_approves_attack_class_at_the_prevalence_floor(tmp_path):
    trace = _valid_trace(tmp_path / "trace.csv", attack_rows=10, normal_rows=990)

    bundle = build_dataset_bundle(
        trace,
        lineage_run_id="run-1",
        expected_attack_label="attack_label",
        min_attack_prevalence=0.01,
    )

    assert bundle.class_counts == {"attack_label": 10, "normal": 990}


def test_build_dataset_bundle_prevalence_floor_can_be_disabled(tmp_path):
    """``0.0`` volta ao comportamento anterior ao piso proporcional."""

    trace = _valid_trace(tmp_path / "trace.csv", attack_rows=27, normal_rows=49_998)

    bundle = build_dataset_bundle(
        trace,
        lineage_run_id="run-1",
        expected_attack_label="attack_label",
        min_attack_prevalence=0.0,
    )

    assert bundle.class_counts["attack_label"] == 27


def test_build_dataset_bundle_rejects_an_out_of_range_prevalence_floor(tmp_path):
    """Erro de configuração do chamador, não trace reprovado — ``ValueError``
    puro, não ``DatasetGateError``."""

    trace = _valid_trace(tmp_path / "trace.csv")

    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]") as excinfo:
        build_dataset_bundle(
            trace,
            lineage_run_id="run-1",
            expected_attack_label="attack_label",
            min_attack_prevalence=1.5,
        )

    assert not isinstance(excinfo.value, DatasetGateError)


def test_build_dataset_bundle_prevalence_counts_every_labeled_row(tmp_path):
    """A prevalência é sobre o total rotulado, não sobre ataque+normal — um
    trace com uma terceira classe não pode inflar a fração do ataque."""

    rows = ["f1,f2,class"]
    rows += [f"{i},1,attack_label" for i in range(10)]
    rows += [f"{i},0,normal" for i in range(500)]
    rows += [f"{i},2,other_attack" for i in range(490)]
    trace = _write_csv(tmp_path / "trace.csv", rows)

    with pytest.raises(DatasetGateError, match="10/1000"):
        build_dataset_bundle(
            trace,
            lineage_run_id="run-1",
            expected_attack_label="attack_label",
            min_attack_prevalence=0.02,
        )
