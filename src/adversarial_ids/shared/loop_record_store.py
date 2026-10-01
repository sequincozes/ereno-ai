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

# Versão do contrato que este store sabe ler. Derivada do próprio modelo para
# não haver um número a atualizar em dois lugares no próximo bump.
_CURRENT_SCHEMA_VERSION: int = LoopRecord.model_fields["schema_version"].default


def load_loop_records(path: str | Path) -> list[LoopRecord]:
    """Lê todos os ``LoopRecord`` já persistidos (lista vazia se não existir).

    Um ledger da versão 1 é recusado com a razão, e não com o erro de
    validação cru do Pydantic: a v2 renomeou dois estágios (``generator`` →
    ``compiler``, ``ereno`` → ``generator``), então um registro antigo falha
    por um valor de enum que não existe mais — mensagem que não ajuda ninguém
    a entender que o arquivo é de outra versão do contrato.
    """

    history_path = Path(path)
    if not history_path.exists():
        return []

    data = load_json(history_path)
    raw_records = data.get("loop_records", []) if isinstance(data, dict) else []
    _reject_superseded_schema(raw_records, history_path)
    return [LoopRecord.model_validate(raw) for raw in raw_records]


def _reject_superseded_schema(raw_records: list, history_path: Path) -> None:
    stale = sorted(
        {
            version
            for raw in raw_records
            if isinstance(raw, dict)
            and isinstance(version := raw.get("schema_version"), int)
            and version < _CURRENT_SCHEMA_VERSION
        }
    )
    if not stale:
        return
    raise ValueError(
        f"{history_path} tem LoopRecord de schema_version {stale!r}, e o "
        f"contrato atual é {_CURRENT_SCHEMA_VERSION}. A v2 renomeou os "
        "estágios 'generator' (compilação) para 'compiler' e 'ereno' para "
        "'generator'. Ledgers vivem em outputs/, que é descartável: apague o "
        "arquivo ou rode `uv run python scripts/init_state.py`."
    )


def append_loop_record(path: str | Path, record: LoopRecord) -> None:
    """Acrescenta ``record`` ao histórico persistido em ``path``.

    Nunca reescreve nem remove registros existentes — apenas relê o arquivo,
    anexa e grava de volta a lista completa (formato ``{"loop_records": [...]}``).

    Recusa um ``run_id`` que já está no ledger. É a última linha de defesa da
    retomada (``IntentLoopOrchestrator.resume_campaign``): uma rodada retomada
    sempre nasce com ``run_id`` novo, então um repetido só pode ser o mesmo
    registro gravado duas vezes.
    """

    records = load_loop_records(path)
    if any(existing.run_id == record.run_id for existing in records):
        raise ValueError(
            f"run_id {record.run_id!r} já está no ledger {path}; um LoopRecord "
            "é gravado uma única vez."
        )
    records.append(record)
    save_json(
        path,
        {"loop_records": [item.model_dump(mode="json") for item in records]},
    )
