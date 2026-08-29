"""core/experiment_memory.py — histórico do experimento como IterationRecord (#17).

Persiste cada passo do loop adversarial como um ``IterationRecord`` (#2), a
unidade de histórico que costura ``AttackConfig`` + ``StrategistOutput`` +
``Metrics`` + ``AnalystOutput``. É o que o Orquestrador (#16) alimenta e o que o
Dashboard (#19) e a memória do Estrategista (#8) leem de volta.

Formato em disco (idêntico ao golden ``data/iteration_history.json`` da #3)::

    {"history": [ <IterationRecord.model_dump(mode="json")>, ... ]}

Dois caminhos de escrita:

- ``add_record`` — recebe um ``IterationRecord`` já tipado (caminho do
  Orquestrador, que monta o registro a partir das saídas dos agentes).
- ``add_iteration`` — adaptador retrocompatível a partir de peças frouxas
  (``attack_json``/``metrics`` como ``dict``), usado por ``interfaces/cli.py``.
  Valida contra os schemas de ``domain/`` antes de guardar, então o histórico
  persistido é sempre um contrato válido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adversarial_ids.domain import (
    IterationRecord,
    Metrics,
    StrategistOutput,
)
from adversarial_ids.shared.json_io import load_json, save_json


class ExperimentMemory:
    """Histórico em memória de ``IterationRecord``, persistível em JSON."""

    def __init__(self, records: list[IterationRecord] | None = None) -> None:
        self.records: list[IterationRecord] = list(records) if records else []

    # ------------------------------------------------------------------ #
    # Escrita                                                             #
    # ------------------------------------------------------------------ #
    def add_record(self, record: IterationRecord) -> IterationRecord:
        """Anexa um ``IterationRecord`` já tipado (caminho do Orquestrador #16)."""

        self.records.append(record)
        return record

    def add_iteration(
        self,
        iteration: int,
        attack_json: dict[str, Any],
        metrics: dict[str, Any] | Metrics,
        llm_response: str = "",
        strategist_output: dict[str, Any] | StrategistOutput | None = None,
        analyst_output: dict[str, Any] | None = None,
    ) -> IterationRecord:
        """Monta e anexa um ``IterationRecord`` a partir de peças frouxas.

        Mantém a assinatura que ``interfaces/cli.py`` já chama. Valida
        ``attack_json`` → ``AttackConfig`` e ``metrics`` → ``Metrics``.
        ``strategist_output`` aceita ``StrategistOutput`` (tipado) ou
        ``dict`` (retrocompatível, validado via ``StrategistOutput.model_validate``).
        Se apenas ``llm_response`` for informado, cria um ``StrategistOutput``
        mínimo com o texto cru em ``reasoning``.
        """

        strategist_output_typed = _as_strategist_output(
            strategist_output, llm_response
        )

        record = IterationRecord(
            iteration=iteration,
            attack_config=attack_json,
            strategist_output=strategist_output_typed,
            metrics=_as_metrics(metrics),
            analyst_output=analyst_output,
        )
        return self.add_record(record)

    # ------------------------------------------------------------------ #
    # Leitura                                                             #
    # ------------------------------------------------------------------ #
    def get_records(self) -> list[IterationRecord]:
        """Cópia da lista de registros tipados."""

        return list(self.records)

    def get_history(self) -> list[dict[str, Any]]:
        """Histórico serializável (dicts) — o que os consumidores leem.

        Cada item tem as chaves ``iteration`` e ``metrics``, então continua
        compatível com ``StrategistAgent._compact_history``.
        """

        return [record.model_dump(mode="json") for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    # Persistência                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Grava ``{"history": [...]}`` no formato do golden."""

        save_json(path, {"history": self.get_history()})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentMemory":
        """Reconstrói a memória a partir de um histórico salvo.

        Se o arquivo não existir, devolve uma memória vazia — assim o loop
        (e o ``scripts/init_state.py``) pode partir de um estado limpo sem o
        chamador tratar exceção.
        """

        history_path = Path(path)
        if not history_path.exists():
            return cls()

        data = load_json(history_path)
        raw_history = data.get("history", []) if isinstance(data, dict) else []
        records = [IterationRecord.model_validate(raw) for raw in raw_history]
        return cls(records)


def _as_metrics(value: dict[str, Any] | Metrics) -> Metrics:
    if isinstance(value, Metrics):
        return value
    return Metrics.model_validate(value)


def _as_strategist_output(
    value: dict[str, Any] | StrategistOutput | None,
    llm_response: str,
) -> StrategistOutput | None:
    """Converte ``dict`` ou ``StrategistOutput`` em ``StrategistOutput`` tipado.

    Retrocompatível: se ``value`` for ``None`` mas ``llm_response`` estiver
    preenchido, cria um ``StrategistOutput`` mínimo com o texto em ``reasoning``.
    """
    if isinstance(value, StrategistOutput):
        return value
    if isinstance(value, dict):
        return StrategistOutput.model_validate(value)
    if llm_response:
        return StrategistOutput(
            reasoning=llm_response,
            persona="conservative",
            changes=[],
        )
    return None
