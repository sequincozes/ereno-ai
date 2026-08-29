"""Dashboard Streamlit da demonstração cacheada do adversarial-ids."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from adversarial_ids.config.settings import (
    GOLDEN_HISTORY_PATH,
    ITERATION_HISTORY_PATH,
    MODEL_ID,
    TOTAL_ITERATIONS,
)
from adversarial_ids.domain import IterationRecord
from adversarial_ids.interfaces.experiment_runner import (
    CachedHistoryRunner,
    ExperimentRunner,
    create_default_runner,
)


def load_cached_demo(path: Path = GOLDEN_HISTORY_PATH) -> list[IterationRecord]:
    """Carrega e valida a fixture produzida pelo pipeline cacheado oficial."""
    return run_experiment(CachedHistoryRunner(path))


def run_experiment(
    runner: ExperimentRunner,
    *,
    iterations: int = TOTAL_ITERATIONS,
    model_id: str = MODEL_ID,
    generator_mode: str = "cached",
) -> list[IterationRecord]:
    """Executa qualquer implementação compatível com a porta da interface."""
    return runner.run(
        iterations=iterations,
        model_id=model_id,
        generator_mode=generator_mode,
    )


def _run_into_session(
    runner: ExperimentRunner,
    *,
    iterations: int = TOTAL_ITERATIONS,
    generator_mode: str = "cached",
) -> None:
    """Executa o runner e guarda o resultado (ou o erro) no ``session_state``.

    Concentra o tratamento de erro da fronteira da interface, compartilhado
    entre a demonstração cacheada e a execução ao vivo.
    """

    try:
        st.session_state.cached_records = run_experiment(
            runner,
            iterations=iterations,
            generator_mode=generator_mode,
        )
    except Exception as error:  # fronteira da interface
        st.session_state.cached_records = None
        st.session_state.dashboard_error = {
            "message": str(error),
            "traceback": traceback.format_exc(),
        }


def prepare_f1_data(records: list[IterationRecord]) -> pd.DataFrame:
    rows = [
        {
            "Iteração": record.iteration,
            "F1": record.metrics.f1_score_attack,
        }
        for record in records
        if record.metrics.f1_score_attack is not None
    ]
    return pd.DataFrame(rows, columns=["Iteração", "F1"]).sort_values(
        "Iteração", ignore_index=True
    )


def prepare_confusion_matrix(record: IterationRecord) -> pd.DataFrame | None:
    metrics = record.metrics
    if any(value is None for value in (metrics.tn, metrics.fp, metrics.fn, metrics.tp)):
        return None

    return pd.DataFrame(
        [[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]],
        index=["Real: normal", "Real: ataque"],
        columns=["Previsto: normal", "Previsto: ataque"],
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flattened.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        flattened[prefix] = value
    return flattened


def compare_configurations(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> pd.DataFrame:
    previous_values = _flatten(previous)
    current_values = _flatten(current)
    fields = sorted(previous_values.keys() | current_values.keys())
    rows = [
        {
            "Campo": field,
            "Valor anterior": previous_values.get(field),
            "Valor posterior": current_values.get(field),
        }
        for field in fields
        if previous_values.get(field) != current_values.get(field)
    ]
    return pd.DataFrame(
        rows,
        columns=["Campo", "Valor anterior", "Valor posterior"],
    )


def prepare_iteration_table(records: list[IterationRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_f1: float | None = None
    for record in sorted(records, key=lambda item: item.iteration):
        metrics = record.metrics
        f1 = metrics.f1_score_attack
        variation = None if f1 is None or previous_f1 is None else f1 - previous_f1
        rows.append(
            {
                "Iteração": record.iteration,
                "F1": f1,
                "Precisão": metrics.precision_attack,
                "Recall": metrics.recall_attack,
                "Acurácia": metrics.accuracy,
                "Variação F1": variation,
                "Configuração alterada": metrics.config_changed,
                "Variante degenerada": metrics.degenerate_variant,
                "Analista disponível": record.analyst_output is not None,
            }
        )
        if f1 is not None:
            previous_f1 = f1
    return pd.DataFrame(rows)


def render_summary(records: list[IterationRecord]) -> None:
    st.subheader("Resumo da demonstração")
    f1_data = prepare_f1_data(records)
    columns = st.columns(5)
    columns[0].metric("Registros", len(records))

    if f1_data.empty:
        for column, label in zip(
            columns[1:], ["F1 inicial", "F1 final", "Variação", "Menor F1"]
        ):
            column.metric(label, "Indisponível")
        return

    initial = float(f1_data.iloc[0]["F1"])
    final = float(f1_data.iloc[-1]["F1"])
    lowest_row = f1_data.loc[f1_data["F1"].idxmin()]
    columns[1].metric("F1 inicial", f"{initial:.4f}")
    columns[2].metric("F1 final", f"{final:.4f}")
    columns[3].metric("Variação", f"{final - initial:+.4f}")
    columns[4].metric(
        "Menor F1",
        f"{float(lowest_row['F1']):.4f}",
        help=f"Iteração {int(lowest_row['Iteração'])}",
    )


def render_f1(records: list[IterationRecord]) -> None:
    st.subheader("Evolução do F1")
    data = prepare_f1_data(records)
    if data.empty:
        st.info("Nenhuma métrica F1 está disponível no histórico.")
        return
    st.line_chart(data.set_index("Iteração"), y="F1")
    st.dataframe(data, hide_index=True, width="stretch")


def render_confusion_matrix(records: list[IterationRecord]) -> None:
    st.subheader("Matriz de confusão")
    by_iteration = {record.iteration: record for record in records}
    selected = st.selectbox(
        "Iteração da matriz",
        options=sorted(by_iteration),
        key="confusion_iteration",
    )
    matrix = prepare_confusion_matrix(by_iteration[selected])
    if matrix is None:
        st.info("A matriz de confusão binária não está disponível nesta iteração.")
        return
    st.caption("Linhas: classe real · Colunas: classe prevista")
    st.dataframe(matrix, width="stretch")


def render_config_diff(records: list[IterationRecord]) -> None:
    st.subheader("Diferenças de configuração")
    by_iteration = {record.iteration: record for record in records}
    iterations = sorted(by_iteration)
    left, right = st.columns(2)
    previous_iteration = left.selectbox(
        "Iteração anterior", iterations, index=0, key="diff_previous"
    )
    current_iteration = right.selectbox(
        "Iteração posterior",
        iterations,
        index=len(iterations) - 1,
        key="diff_current",
    )
    differences = compare_configurations(
        by_iteration[previous_iteration].attack_config,
        by_iteration[current_iteration].attack_config,
    )
    if differences.empty:
        st.info("As configurações selecionadas são iguais.")
        return
    st.dataframe(differences, hide_index=True, width="stretch")


def render_analyst_output(records: list[IterationRecord]) -> None:
    st.subheader("Resultado do Analista")
    by_iteration = {record.iteration: record for record in records}
    selected = st.selectbox(
        "Iteração analisada",
        options=sorted(by_iteration),
        key="analyst_iteration",
    )
    output = by_iteration[selected].analyst_output
    if not output:
        st.info("Não há resultado do Analista para esta iteração.")
        return

    output_data = output.model_dump(mode="json")
    severity = output_data.get("severity")
    diagnosis = output_data.get("diagnosis")
    if severity:
        st.metric("Severidade", str(severity).upper())
    if diagnosis:
        st.markdown("**Diagnóstico**")
        st.write(diagnosis)

    deceptive_features = output_data.get("deceptive_features")
    if isinstance(deceptive_features, list) and deceptive_features:
        st.markdown("**Features relevantes**")
        st.dataframe(deceptive_features, hide_index=True, width="stretch")

    mitigations = output_data.get("mitigations")
    if isinstance(mitigations, list) and mitigations:
        st.markdown("**Mitigações recomendadas**")
        st.dataframe(mitigations, hide_index=True, width="stretch")

    known_fields = {"iteration", "severity", "diagnosis", "deceptive_features", "mitigations"}
    extra_fields = {
        key: value
        for key, value in output_data.items()
        if key not in known_fields
    }
    if extra_fields:
        with st.expander("Outros dados do Analista"):
            st.json(extra_fields)


def render_iteration_table(records: list[IterationRecord]) -> None:
    st.subheader("Dados por iteração")
    st.dataframe(
        prepare_iteration_table(records),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="Laboratório Inteligente · Dashboard",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("ERENO AI - Dashboard")
    st.write(
        "Demonstração do loop adversarial Red Team × Blue Team para avaliação "
        "de robustez de um IDS em Smart Grids."
    )
    st.session_state.setdefault("cached_records", None)
    st.session_state.setdefault("dashboard_error", None)
    st.session_state.setdefault("records_source", None)
    st.session_state.setdefault("auto_loaded", False)

    latest_exists = ITERATION_HISTORY_PATH.exists()

    # Ao abrir, carrega automaticamente a última execução salva pela CLI/loop —
    # assim o dashboard já mostra o panorama dos resultados sem exigir cliques.
    if (
        not st.session_state.auto_loaded
        and st.session_state.cached_records is None
        and latest_exists
    ):
        st.session_state.auto_loaded = True
        with st.spinner("Carregando a última execução salva..."):
            _run_into_session(CachedHistoryRunner(ITERATION_HISTORY_PATH))
        st.session_state.records_source = f"Última execução · {ITERATION_HISTORY_PATH.name}"

    latest_col, demo_col, live_col = st.columns(3)

    with latest_col:
        st.caption("Última execução salva pela CLI (`outputs/`).")
        if st.button(
            "Carregar última execução",
            type="primary",
            disabled=not latest_exists,
        ):
            st.session_state.dashboard_error = None
            with st.spinner("Carregando a última execução..."):
                _run_into_session(CachedHistoryRunner(ITERATION_HISTORY_PATH))
            st.session_state.records_source = f"Última execução · {ITERATION_HISTORY_PATH.name}"

    with demo_col:
        st.caption("Histórico golden versionado (`data/`), sem Groq/Java.")
        if st.button("Demo cacheada (golden)"):
            st.session_state.dashboard_error = None
            with st.spinner("Carregando o histórico golden..."):
                _run_into_session(create_default_runner("demo"))
            st.session_state.records_source = "Demo golden cacheada"

    with live_col:
        st.caption("Loop real (Groq). Exige GROQ_API_KEY; jar exige o JAR.")
        live_iterations = st.number_input(
            "Iterações", min_value=1, max_value=TOTAL_ITERATIONS, value=3, step=1
        )
        live_mode = st.selectbox("Modo do gerador", ("cached", "jar"))
        if st.button("Executar ao vivo (Groq)"):
            st.session_state.dashboard_error = None
            with st.spinner("Executando o loop adversarial ao vivo..."):
                _run_into_session(
                    create_default_runner("live"),
                    iterations=int(live_iterations),
                    generator_mode=live_mode,
                )
            st.session_state.records_source = "Execução ao vivo (Groq)"

    error = st.session_state.dashboard_error
    if error:
        st.error(f"Não foi possível carregar a demonstração: {error['message']}")
        with st.expander("Detalhes técnicos"):
            st.code(error["traceback"])
        return

    records = st.session_state.cached_records
    if records is None:
        st.info(
            "Nenhum resultado carregado ainda. Rode um experimento pela CLI "
            "(`adversarial-ids --engine live ...`) e recarregue, ou use um dos "
            "botões acima."
        )
        return
    if not records:
        st.warning("A execução terminou sem registros para exibir.")
        return

    source = st.session_state.records_source or "resultados"
    st.success(f"Exibindo: {source} — {len(records)} registro(s).")
    render_summary(records)
    render_f1(records)
    render_confusion_matrix(records)
    render_config_diff(records)
    render_iteration_table(records)
    render_analyst_output(records)


if __name__ == "__main__":
    main()
