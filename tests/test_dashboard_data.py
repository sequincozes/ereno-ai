"""Testes das transformações puras usadas pelo dashboard."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from adversarial_ids.config.settings import GOLDEN_HISTORY_PATH
from adversarial_ids.interfaces.dashboard.app import (
    compare_configurations,
    load_cached_demo,
    prepare_confusion_matrix,
    prepare_f1_data,
    prepare_iteration_table,
)


def test_cached_demo_is_valid_and_complete_for_dashboard():
    records = load_cached_demo(GOLDEN_HISTORY_PATH)

    assert len(records) == 4
    assert all(record.metrics.f1_score_attack is not None for record in records)
    assert all(record.analyst_output for record in records)


def test_f1_and_iteration_tables_use_domain_values():
    records = load_cached_demo()

    f1_data = prepare_f1_data(records)
    iteration_data = prepare_iteration_table(records)

    assert f1_data["Iteração"].tolist() == [0, 1, 2, 3]
    assert f1_data["F1"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert iteration_data["Analista disponível"].all()


def test_confusion_matrix_has_real_rows_and_predicted_columns():
    record = load_cached_demo()[0]

    matrix = prepare_confusion_matrix(record)

    assert matrix is not None
    assert matrix.loc["Real: normal", "Previsto: normal"] == record.metrics.tn
    assert matrix.loc["Real: normal", "Previsto: ataque"] == record.metrics.fp
    assert matrix.loc["Real: ataque", "Previsto: normal"] == record.metrics.fn
    assert matrix.loc["Real: ataque", "Previsto: ataque"] == record.metrics.tp


def test_config_diff_compares_serialized_attack_configs():
    records = load_cached_demo()

    differences = compare_configurations(
        records[0].attack_config,
        records[-1].attack_config,
    )

    assert set(differences["Campo"]) == {
        "analog.deltaAbs.max",
        "fault.prob",
        "trapArea.spikeProb",
    }


def test_streamlit_dashboard_renders_typed_analyst_output():
    app = AppTest.from_file(
        "src/adversarial_ids/interfaces/dashboard/app.py"
    )

    app.run(timeout=20)
    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert app.session_state["cached_records"]
    assert app.success
