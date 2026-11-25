"""Extração do consumo declarado por uma resposta de agente (épico E11).

O ``agno`` devolve um ``RunOutput`` com ``.metrics`` (``input_tokens``,
``output_tokens``, ``cost``, …). Este módulo lê aquilo por ``getattr`` em vez de
importar o tipo: assim ele funciona igual com o agente real, com um stub de
teste e com uma versão do agno em que o objeto de métricas mudou de nome — e
continua fora do conjunto de módulos que arrastam ``agno`` para o import.

Nunca levanta. Contabilidade que derruba a execução que ela apenas mede é pior
do que contabilidade ausente; quando não dá para ler, devolve ``None``, e quem
chama mostra "—" em vez de zero. A distinção importa: zero token é uma
afirmação sobre a chamada, ``None`` é a confissão de que não se sabe.
"""

from __future__ import annotations

from typing import Any

from adversarial_ids.domain.run_usage import AgentUsage

# Nomes já vistos para o mesmo número, do mais específico para o mais genérico.
_INPUT_FIELDS: tuple[str, ...] = ("input_tokens", "prompt_tokens")
_OUTPUT_FIELDS: tuple[str, ...] = ("output_tokens", "completion_tokens")
_COST_FIELDS: tuple[str, ...] = ("cost", "cost_usd")


def _first_number(source: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = getattr(source, name, None)
        if isinstance(value, bool):  # bool é int em Python; aqui não serve
            continue
        if isinstance(value, (int, float)):
            return value

    return None


def usage_from_response(response: Any) -> AgentUsage | None:
    """Lê o consumo de uma resposta de agente, ou ``None`` se ela não declara."""

    metrics = getattr(response, "metrics", None)
    if metrics is None:
        return None

    input_tokens = _first_number(metrics, _INPUT_FIELDS)
    output_tokens = _first_number(metrics, _OUTPUT_FIELDS)
    cost = _first_number(metrics, _COST_FIELDS)

    if input_tokens is None and output_tokens is None and cost is None:
        return None

    try:
        return AgentUsage(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cost_usd=float(cost) if cost is not None else None,
        )
    except Exception:  # métrica incoerente não derruba a execução
        return None
