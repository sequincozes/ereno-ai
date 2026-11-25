"""Estado de sessão compartilhado entre as páginas do Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from adversarial_ids.core.experiment_memory import ExperimentMemory
from adversarial_ids.domain import IterationRecord

from .bridge import ExperimentConfig


def init() -> None:
    ss = st.session_state
    ss.setdefault("config", ExperimentConfig())
    ss.setdefault("job", None)               # ExperimentJob | None
    ss.setdefault("records", None)           # list[IterationRecord] | None
    ss.setdefault("records_source", None)    # str | None


def get_config() -> ExperimentConfig:
    init()
    return st.session_state["config"]


def set_records(records: list[IterationRecord], source: str) -> None:
    st.session_state["records"] = records
    st.session_state["records_source"] = source


def get_records() -> list[IterationRecord] | None:
    return st.session_state.get("records")


def load_history_file(path: Path, source: str) -> list[IterationRecord]:
    """Carrega um histórico salvo em disco (outputs/ ou data/)."""
    memory = ExperimentMemory.load(path)
    records = memory.get_records()
    set_records(records, source)
    return records
