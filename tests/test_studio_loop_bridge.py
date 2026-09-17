"""Testes da ponte do loop intent-driven para a UI (épico E11).

Lógica pura: nada aqui sobe Streamlit, chama a Groq ou roda o ERENO.
"""

from __future__ import annotations

import time

import pytest

from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.interfaces.dashboard.studio.loop_bridge import (
    IntentLoopConfig,
    IntentLoopJob,
    outcome_for,
    run_directory,
    saved_records,
    timeline_for,
)
from adversarial_ids.shared.json_io import save_json
from adversarial_ids.shared.loop_event_store import JsonlEventSink
from adversarial_ids.shared.loop_record_store import append_loop_record
from adversarial_ids.domain.loop_event import LoopEvent


def _record(run_dir=None, *, run_id: str = "run-1", failed: bool = False) -> LoopRecord:
    stages = [
        LoopStage(
            name="intent",
            status=LoopStageStatus.SUCCEEDED,
            duration_seconds=0.5,
            artifact_ref=str(run_dir / "intent.json") if run_dir else None,
        )
    ]
    if failed:
        stages.append(
            LoopStage(
                name="ereno",
                status=LoopStageStatus.FAILED,
                duration_seconds=0.1,
                error="jar ausente",
            )
        )
    return LoopRecord(
        run_id=run_id,
        source_prompt="Reduza o recall.",
        seed=42,
        stages=tuple(stages),
    )


def _report_payload() -> dict:
    return {
        "model_name": "random_forest",
        "split": "train_test_80_20_seed42",
        "accuracy": 0.75,
        "precision": 0.70,
        "recall": 0.60,
        "f1": 0.65,
        "confusion_matrix": {"tp": 12, "fp": 4, "fn": 8, "tn": 20},
        "top_features": [{"feature": "isbATrapAreaSum", "importance": 0.42}],
        "latency_ms": 3.5,
    }


# --------------------------------------------------------------------------- #
# Localizar os artefatos de uma execução                                      #
# --------------------------------------------------------------------------- #
def test_run_directory_is_derived_from_the_artifacts_not_from_settings(tmp_path):
    """Recompor o caminho a partir de settings ignoraria um output_dir custom."""

    run_dir = tmp_path / "artifacts" / "run-1"
    assert run_directory(_record(run_dir)) == run_dir


def test_run_directory_is_none_when_no_stage_left_an_artifact():
    assert run_directory(_record()) is None


def test_outcome_reads_the_three_result_artifacts(tmp_path):
    run_dir = tmp_path / "run-1"
    save_json(run_dir / "detection_report.json", _report_payload())
    save_json(
        run_dir / "defense_plan.json",
        {
            "priority": "medium",
            "detection_actions": [
                {
                    "description": "Revisar o limiar.",
                    "technique": "detector_threshold_tuning",
                    "evidence": [{"metric_or_feature": "recall", "value": 0.60}],
                    "validation_test": {
                        "metric": "recall",
                        "direction": "increase",
                        "target": 0.80,
                        "procedure": "Reavaliar.",
                    },
                }
            ],
        },
    )
    save_json(
        run_dir / "defense_rules.json",
        {"total_actions": 1, "grounded_actions": 1, "findings": []},
    )

    outcome = outcome_for(_record(run_dir))

    assert outcome.has_result
    assert outcome.detection_report is not None
    assert outcome.detection_report.f1 == 0.65
    assert outcome.defense_plan is not None
    assert outcome.defense_rules is not None
    assert outcome.defense_rules.is_grounded


def test_a_run_that_died_early_shows_what_it_has_instead_of_blowing_up(tmp_path):
    """Uma execução que parou no ERENO não tem relatório — e isso é informação."""

    run_dir = tmp_path / "run-1"
    save_json(run_dir / "intent.json", {"qualquer": "coisa"})

    outcome = outcome_for(_record(run_dir, failed=True))

    assert not outcome.has_result
    assert outcome.run_dir == run_dir
    assert outcome.defense_plan is None


def test_a_corrupt_artifact_reads_as_absent_not_as_a_crash(tmp_path):
    run_dir = tmp_path / "run-1"
    save_json(run_dir / "intent.json", {})
    (run_dir / "detection_report.json").write_text("{isso não é json", encoding="utf-8")

    assert outcome_for(_record(run_dir)).detection_report is None


# --------------------------------------------------------------------------- #
# Histórico e timeline                                                        #
# --------------------------------------------------------------------------- #
def test_saved_records_come_back_newest_first(tmp_path):
    path = tmp_path / "loop_records.json"
    append_loop_record(path, _record(run_id="run-antigo"))
    append_loop_record(path, _record(run_id="run-novo"))

    assert [record.run_id for record in saved_records(path)] == [
        "run-novo",
        "run-antigo",
    ]


def test_saved_records_is_empty_before_the_first_run(tmp_path):
    assert saved_records(tmp_path / "ausente.json") == []


def test_timeline_for_reads_the_events_of_that_run(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    save_json(run_dir / "intent.json", {})
    sink = JsonlEventSink(run_dir / "events.jsonl")
    sink(LoopEvent(run_id="run-1", sequence=0, kind="stage_started", stage="intent"))

    events = timeline_for(_record(run_dir))

    assert [event.stage for event in events] == ["intent"]


def test_timeline_is_empty_when_the_run_left_no_artifacts():
    assert timeline_for(_record()) == []


# --------------------------------------------------------------------------- #
# O job em background                                                         #
# --------------------------------------------------------------------------- #
def _finished_job(runner) -> IntentLoopJob:
    job = IntentLoopJob(IntentLoopConfig(prompt="Reduza o recall."), runner=runner).start()
    deadline = time.time() + 5
    while job.is_running() and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_job_passes_the_form_through_and_registers_its_own_sink():
    seen: dict = {}

    def fake_runner(**kwargs):
        seen.update(kwargs)
        kwargs["event_sink"](
            LoopEvent(run_id="r", sequence=0, kind="stage_started", stage="intent")
        )
        return (_record(),)

    job = _finished_job(fake_runner)

    assert seen["prompt"] == "Reduza o recall."
    assert seen["rounds"] == 1
    assert job.records is not None and len(job.records) == 1
    assert [event.stage for event in job.events()] == ["intent"]


def test_progress_counts_finished_stages_not_substrings_of_a_log():
    """O motivo do E11: a ponte legada deduzia a fase procurando texto no stdout."""

    def fake_runner(**kwargs):
        sink = kwargs["event_sink"]
        for index, stage in enumerate(("intent", "generator", "ereno")):
            sink(
                LoopEvent(
                    run_id="r",
                    sequence=index,
                    kind="stage_finished",
                    stage=stage,
                    status=LoopStageStatus.SUCCEEDED,
                    duration_seconds=0.1,
                )
            )
        return (_record(),)

    job = _finished_job(fake_runner)

    assert job.progress() == 3 / 7


def test_current_stage_is_the_one_started_and_not_yet_finished():
    def fake_runner(**kwargs):
        sink = kwargs["event_sink"]
        sink(LoopEvent(run_id="r", sequence=0, kind="stage_started", stage="intent"))
        sink(
            LoopEvent(
                run_id="r",
                sequence=1,
                kind="stage_finished",
                stage="intent",
                status=LoopStageStatus.SUCCEEDED,
            )
        )
        sink(LoopEvent(run_id="r", sequence=2, kind="stage_started", stage="ereno"))
        return (_record(),)

    assert _finished_job(fake_runner).current_stage() == "ereno"


def test_failed_stage_carries_the_cause_to_the_screen():
    """Critério da linha Operação: a falha tem estágio *e* causa."""

    def fake_runner(**kwargs):
        kwargs["event_sink"](
            LoopEvent(
                run_id="r",
                sequence=0,
                kind="stage_finished",
                stage="ereno",
                status=LoopStageStatus.FAILED,
                duration_seconds=0.2,
                message="jar ausente",
            )
        )
        return (_record(failed=True),)

    failed = _finished_job(fake_runner).failed_stage()

    assert failed is not None
    assert failed.stage == "ereno"
    assert failed.message == "jar ausente"


def test_a_runner_that_raises_becomes_a_visible_error_not_a_dead_page():
    def explode(**_kwargs):
        raise RuntimeError("GROQ_API_KEY ausente")

    job = _finished_job(explode)

    assert job.records is None
    assert job.error == "GROQ_API_KEY ausente"
    assert not job.is_running()


# --------------------------------------------------------------------------- #
# A página                                                                    #
# --------------------------------------------------------------------------- #
# `AppTest.from_function` reexecuta o *fonte* da função num namespace novo, e
# ali `ui`/`st` do módulo não existem. O script mínimo abaixo importa a página e
# a chama — é o mesmo caminho que o Streamlit percorre de verdade.
_PAGE_SCRIPT = """
from adversarial_ids.interfaces.dashboard.studio import view_loop

view_loop.render()
"""


def _page(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "pagina_loop.py"
    script.write_text(_PAGE_SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(script))


def test_the_loop_page_renders_without_a_run_history(tmp_path, monkeypatch):
    """Primeira abertura, `outputs/` vazio: informa, não quebra."""

    from adversarial_ids.interfaces.dashboard.studio import view_loop

    monkeypatch.setattr(view_loop, "LOOP_RECORDS_PATH", tmp_path / "ausente.json")

    app = _page(tmp_path).run(timeout=20)

    assert not app.exception
    assert app.button, "a página precisa oferecer o disparo da campanha"
    assert any("Nenhuma execução salva" in str(info.value) for info in app.info)


def test_the_loop_page_shows_stage_duration_and_artifact_of_a_saved_run(
    tmp_path, monkeypatch
):
    """O critério de pronto do E11, verificado na tela e não só no contrato."""

    from adversarial_ids.interfaces.dashboard.studio import view_loop

    run_dir = tmp_path / "run-1"
    save_json(run_dir / "detection_report.json", _report_payload())
    records_path = tmp_path / "loop_records.json"
    append_loop_record(records_path, _record(run_dir, failed=True))

    monkeypatch.setattr(view_loop, "LOOP_RECORDS_PATH", records_path)

    app = _page(tmp_path).run(timeout=20)

    assert not app.exception
    frames = [df.value for df in app.dataframe]
    stage_table = next(frame for frame in frames if "Estágio" in frame.columns)
    stages = stage_table.set_index("Estágio")
    assert stages.loc["Intenção"]["Status"] == "concluído"
    assert stages.loc["Intenção"]["Duração"] == "0.50s"
    assert stages.loc["Intenção"]["Artefato"] == "intent.json"
    # A causa da falha aparece; os estágios que nunca rodaram ficam pendentes.
    assert any("jar ausente" in str(error.value) for error in app.error)
    assert stages.loc["Defesa"]["Status"] == "pendente"


def test_the_loop_page_shows_the_result_of_a_run_that_reached_detection(
    tmp_path, monkeypatch
):
    from adversarial_ids.interfaces.dashboard.studio import view_loop

    run_dir = tmp_path / "run-1"
    save_json(run_dir / "detection_report.json", _report_payload())
    records_path = tmp_path / "loop_records.json"
    append_loop_record(records_path, _record(run_dir))

    monkeypatch.setattr(view_loop, "LOOP_RECORDS_PATH", records_path)

    app = _page(tmp_path).run(timeout=20)

    assert not app.exception
    rendered = " ".join(str(block.value) for block in app.markdown)
    assert "0.6500" in rendered  # F1 do relatório persistido
    assert "random_forest" in rendered
