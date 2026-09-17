"""Testes do LoopEvent e da sua persistência/distribuição (épico E11)."""

from __future__ import annotations

import json

import pytest

from adversarial_ids.domain.loop_event import LOOP_EVENT_KINDS, LoopEvent
from adversarial_ids.domain.loop_record import LOOP_STAGE_NAMES, LoopStageStatus
from adversarial_ids.shared.loop_event_store import (
    JsonlEventSink,
    MemoryEventSink,
    fanout,
    load_loop_events,
    stage_timeline,
)


def event(**overrides) -> LoopEvent:
    payload = {
        "run_id": "run-1",
        "sequence": 0,
        "kind": "stage_started",
        "stage": "intent",
    }
    payload.update(overrides)
    return LoopEvent.model_validate(payload)


# --------------------------------------------------------------------------- #
# Contrato                                                                    #
# --------------------------------------------------------------------------- #
def test_stage_event_must_name_the_stage_it_came_from():
    with pytest.raises(ValueError, match="exige o estágio"):
        LoopEvent.model_validate(
            {"run_id": "r", "sequence": 0, "kind": "stage_started"}
        )


def test_run_event_may_not_name_a_stage():
    """`run_started` com estágio esconderia uma transição que não aconteceu."""

    with pytest.raises(ValueError, match="evento de execução"):
        LoopEvent.model_validate(
            {
                "run_id": "r",
                "sequence": 0,
                "kind": "run_started",
                "stage": "intent",
            }
        )


def test_finished_event_requires_the_resulting_status():
    with pytest.raises(ValueError, match="exige o status"):
        LoopEvent.model_validate(
            {
                "run_id": "r",
                "sequence": 0,
                "kind": "stage_finished",
                "stage": "detector",
            }
        )


@pytest.mark.parametrize(
    "field, value",
    [("status", "succeeded"), ("duration_seconds", 1.5)],
)
def test_a_starting_event_cannot_report_an_outcome(field: str, value):
    """Começar não produz resultado; declarar um seria mentira verificável."""

    with pytest.raises(ValueError, match="não pode declarar"):
        event(**{field: value})


def test_only_the_run_finished_event_is_terminal():
    assert event(
        kind="run_finished", stage=None, status=LoopStageStatus.SUCCEEDED
    ).is_terminal
    assert not event().is_terminal


def test_event_vocabulary_matches_the_record():
    """O evento e o LoopRecord falam dos mesmos sete estágios, por construção."""

    for stage in LOOP_STAGE_NAMES:
        assert event(stage=stage).stage == stage

    assert set(LOOP_EVENT_KINDS) == {
        "run_started",
        "stage_started",
        "stage_finished",
        "run_finished",
    }


# --------------------------------------------------------------------------- #
# Sinks                                                                       #
# --------------------------------------------------------------------------- #
def test_jsonl_sink_writes_one_readable_line_per_event(tmp_path):
    """Uma linha por evento é o que torna a timeline legível durante a execução.

    Um array JSON só valida depois do colchete final — ou seja, depois que a
    execução acabou, quando o arquivo já não serve para observar nada.
    """

    path = tmp_path / "nested" / "events.jsonl"
    sink = JsonlEventSink(path)

    sink(event(sequence=0))
    sink(event(sequence=1, kind="stage_finished", status=LoopStageStatus.SUCCEEDED))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # cada linha vale sozinha


def test_memory_sink_snapshot_does_not_alias_the_live_list(tmp_path):
    """Quem emite é a thread da execução; quem lê é a da UI."""

    sink = MemoryEventSink()
    sink(event(sequence=0))

    snapshot = sink.snapshot()
    sink(event(sequence=1))

    assert len(snapshot) == 1
    assert len(sink.snapshot()) == 2


def test_fanout_feeds_every_sink_and_ignores_the_absent_ones():
    seen_a: list[LoopEvent] = []
    seen_b: list[LoopEvent] = []

    fanout(seen_a.append, None, seen_b.append)(event())

    assert len(seen_a) == len(seen_b) == 1


def test_a_broken_sink_never_takes_the_run_down_with_it():
    """Observabilidade quebrada é problema menor que pipeline interrompido."""

    seen: list[LoopEvent] = []

    def explode(_: LoopEvent) -> None:
        raise RuntimeError("disco cheio")

    fanout(explode, seen.append)(event())

    assert len(seen) == 1


# --------------------------------------------------------------------------- #
# Leitura                                                                     #
# --------------------------------------------------------------------------- #
def test_loading_a_missing_timeline_is_empty_not_an_error(tmp_path):
    assert load_loop_events(tmp_path / "ausente.jsonl") == []


def test_events_are_ordered_by_sequence_not_by_timestamp(tmp_path):
    """Dois estágios rápidos cabem no mesmo timestamp; `sequence` não empata."""

    path = tmp_path / "events.jsonl"
    same_instant = "2026-09-17T12:00:00+00:00"
    sink = JsonlEventSink(path)
    for seq in (2, 0, 1):
        sink(event(sequence=seq, created_at=same_instant))

    assert [e.sequence for e in load_loop_events(path)] == [0, 1, 2]


def test_a_truncated_last_line_does_not_hide_the_rest(tmp_path):
    """É justamente da execução que morreu no meio que se quer ver a timeline."""

    path = tmp_path / "events.jsonl"
    JsonlEventSink(path)(event(sequence=0))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "run-1", "sequ')

    events = load_loop_events(path)

    assert [e.sequence for e in events] == [0]


def test_campaign_rounds_stay_in_order_across_the_timeline(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlEventSink(path)
    sink(event(round=2, sequence=0))
    sink(event(round=1, sequence=1))

    assert [(e.round, e.sequence) for e in load_loop_events(path)] == [(1, 1), (2, 0)]


def test_stage_timeline_keeps_the_latest_state_of_each_stage():
    """O `stage_finished` sobrescreve o `stage_started`: é a transição."""

    timeline = stage_timeline(
        [
            event(sequence=0, stage="intent"),
            event(
                sequence=1,
                stage="intent",
                kind="stage_finished",
                status=LoopStageStatus.SUCCEEDED,
            ),
            event(sequence=2, stage="generator"),
        ]
    )

    assert set(timeline) == {"intent", "generator"}
    assert timeline["intent"].kind == "stage_finished"
    assert timeline["generator"].kind == "stage_started"


def test_stage_timeline_ignores_the_run_level_events():
    timeline = stage_timeline(
        [
            event(sequence=0, kind="run_started", stage=None),
            event(sequence=1, stage="intent"),
        ]
    )

    assert set(timeline) == {"intent"}
