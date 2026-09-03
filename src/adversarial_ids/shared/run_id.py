"""Identificador de execução para telemetria (ação 72h #5 do plano)."""

from __future__ import annotations

import uuid


def new_run_id() -> str:
    """Gera um identificador único de execução, para correlacionar logs/artefatos."""
    return uuid.uuid4().hex
