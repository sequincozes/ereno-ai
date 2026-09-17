"""loop_event_store — persistência e distribuição de ``LoopEvent`` (épico E11).

## Por que JSONL, e não o formato dos outros artefatos

Todo o resto do repo grava JSON bem formado via ``json_io.save_json``, e aqui
isso não serve: um array JSON só é legível depois da vírgula final e do
colchete de fechamento, ou seja, **depois que a execução termina** — exatamente
quando o arquivo deixa de ser interessante para observabilidade. JSONL é válido
linha a linha, então a UI consegue ler a timeline de uma execução que ainda está
rodando, e uma execução que morreu no meio deixa em disco tudo que chegou a
acontecer até o estouro.

O efeito prático é o que o E11 cobra: uma falha tem estágio e causa mesmo quando
o processo não chegou a persistir o ``LoopRecord``.

## Sink é um callable, não uma classe abstrata

O orquestrador só precisa de ``(LoopEvent) -> None``. Um protocolo formal
custaria mais do que entrega: o sink de arquivo é uma classe porque tem um
descritor para manter aberto, o sink da UI é um ``list.append``, e um teste
passa uma lambda. Os três já satisfazem o tipo.

``fanout`` compõe vários. É como o orquestrador escreve em disco *e* alimenta a
UI ao vivo sem que nenhum dos dois lados saiba do outro.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from adversarial_ids.domain.loop_event import LoopEvent

LoopEventSink = Callable[[LoopEvent], None]


class JsonlEventSink:
    """Grava cada evento como uma linha JSON, criando o diretório se preciso.

    Abre e fecha o arquivo a cada evento em vez de manter um descritor: uma
    execução emite dezenas de eventos, não milhares, e o custo de abrir é
    irrelevante perto da garantia que compra — o evento está em disco no
    instante em que foi emitido, mesmo que o processo morra no estágio seguinte.
    Um sink que bufferizasse perderia justamente os eventos que explicam a morte.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __call__(self, event: LoopEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MemoryEventSink:
    """Acumula eventos em memória — o que a UI registra para a timeline ao vivo.

    A lista é reposta por uma cópia a cada leitura (``snapshot``) porque quem
    emite é a thread da execução e quem lê é a thread do Streamlit; devolver a
    lista viva deixaria o leitor iterando sobre algo que cresce embaixo dele.
    """

    def __init__(self) -> None:
        self._events: list[LoopEvent] = []

    def __call__(self, event: LoopEvent) -> None:
        self._events.append(event)

    def snapshot(self) -> tuple[LoopEvent, ...]:
        return tuple(self._events)


def fanout(*sinks: LoopEventSink | None) -> LoopEventSink:
    """Compõe sinks num só, ignorando os ausentes.

    Um sink que levanta não pode derrubar a execução que ele apenas observa:
    observabilidade quebrada é um problema menor que um pipeline interrompido
    por causa dela. A exceção é engolida por sink, e os demais ainda recebem o
    evento.
    """

    active = [sink for sink in sinks if sink is not None]

    def _emit(event: LoopEvent) -> None:
        for sink in active:
            try:
                sink(event)
            except Exception:  # fronteira de observabilidade
                continue

    return _emit


def load_loop_events(path: str | Path) -> list[LoopEvent]:
    """Lê a timeline persistida, em ordem de ``sequence`` (vazia se não existir).

    Linhas ilegíveis são puladas em vez de derrubar a leitura: o arquivo de uma
    execução morta no meio pode terminar numa linha truncada, e é dessa execução
    que mais se quer ver a timeline.
    """

    events_path = Path(path)
    if not events_path.exists():
        return []

    events: list[LoopEvent] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(LoopEvent.model_validate(json.loads(stripped)))
        except Exception:  # linha truncada ou de um schema futuro
            continue

    return sorted(events, key=lambda event: (event.round, event.sequence))


def stage_timeline(events: Iterable[LoopEvent]) -> dict[str, LoopEvent]:
    """Último evento de cada estágio, para a UI desenhar o estado corrente.

    "Último" pela ordem de emissão: um ``stage_finished`` sobrescreve o
    ``stage_started`` do mesmo estágio, que é precisamente a transição de
    "rodando" para o resultado.
    """

    latest: dict[str, LoopEvent] = {}
    for event in sorted(events, key=lambda item: (item.round, item.sequence)):
        if event.stage is not None:
            latest[event.stage] = event

    return latest
