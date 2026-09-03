"""loop_record_store — persistência append-only de LoopRecord (épico E3).

``LoopRecord`` é congelado (``ConfigDict(frozen=True)``) e representa uma
execução completa do pipeline intent-driven. Diferente de
``ExperimentMemory`` (que reescreve o arquivo inteiro a cada iteração de uma
mesma execução), aqui cada chamada de ``append_loop_record`` é uma execução
inteira já concluída — o arquivo só cresce, registros antigos nunca são
reescritos ou removidos.
"""

from __future__ import annotations

from pathlib import Path

from adversarial_ids.domain.loop_record import LoopRecord
from adversarial_ids.shared.json_io import load_json, save_json


def load_loop_records(path: str | Path) -> list[LoopRecord]:
    """Lê todos os ``LoopRecord`` já persistidos (lista vazia se não existir)."""

    history_path = Path(path)
    if not history_path.exists():
        return []

    data = load_json(history_path)
    raw_records = data.get("loop_records", []) if isinstance(data, dict) else []
    return [LoopRecord.model_validate(raw) for raw in raw_records]


def append_loop_record(path: str | Path, record: LoopRecord) -> None:
    """Acrescenta ``record`` ao histórico persistido em ``path``.

    Nunca reescreve nem remove registros existentes — apenas relê o arquivo,
    anexa e grava de volta a lista completa (formato ``{"loop_records": [...]}``).
    """

    records = load_loop_records(path)
    records.append(record)
    save_json(
        path,
        {"loop_records": [item.model_dump(mode="json") for item in records]},
    )
