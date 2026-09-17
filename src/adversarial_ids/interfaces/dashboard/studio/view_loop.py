"""Página · Loop intent-driven — dispara a campanha e mostra a timeline (E11).

Esta é a tela do critério de pronto do E11: "usuário vê estágio, duração,
artefatos, erro e resultado". As cinco coisas saem de fontes diferentes e é
bom não confundi-las:

- **estágio e erro**, ao vivo, do ``LoopEvent`` que o núcleo emite;
- **duração e artefatos**, do ``LoopStage`` do ``LoopRecord`` persistido;
- **resultado**, dos artefatos JSON que a execução deixou no diretório dela
  (``loop_bridge.outcome_for``), porque o registro guarda o caminho e não o
  conteúdo.

Até aqui o motor intent-driven não era alcançável pela UI — o studio só rodava
``demo``/``live``, o que contradizia o "substitui o uso via terminal" do próprio
``studio_app``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from adversarial_ids.config.settings import LOOP_RECORDS_PATH
from adversarial_ids.domain.loop_record import LoopRecord, LoopStageStatus

from . import components as ui
from .loop_bridge import (
    IntentLoopConfig,
    IntentLoopJob,
    outcome_for,
    run_directory,
    saved_records,
    timeline_for,
)

_POLL_SECONDS = 0.6

# Os sete estágios na ordem do pipeline, com o rótulo que a tela mostra. Ordem
# fixa de propósito: uma execução que parou no terceiro estágio precisa mostrar
# os quatro que não rodaram como pendentes, e não simplesmente omiti-los.
_STAGES: tuple[tuple[str, str], ...] = (
    ("intent", "Intenção"),
    ("generator", "Compilação"),
    ("ereno", "Geração ERENO"),
    ("preprocess", "Dataset"),
    ("detector", "Detecção"),
    ("defender", "Defesa"),
    ("feedback", "Feedback"),
)

_STATUS_BADGE: dict[LoopStageStatus, tuple[str, str]] = {
    LoopStageStatus.SUCCEEDED: ("concluído", "good"),
    LoopStageStatus.FAILED: ("falhou", "bad"),
    LoopStageStatus.SKIPPED: ("pulado", "warn"),
    LoopStageStatus.RUNNING: ("rodando", "warn"),
    LoopStageStatus.PENDING: ("pendente", ""),
}


def _duration(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds:.2f}s"


# --------------------------------------------------------------------------- #
# Timeline ao vivo                                                            #
# --------------------------------------------------------------------------- #
def _render_live(job: IntentLoopJob) -> None:
    running_stage = job.current_stage()
    failed = job.failed_stage()

    if failed is not None:
        state = ("Falhou", "bad")
    elif job.is_running():
        state = ("Executando", "warn")
    elif job.error:
        state = ("Erro", "bad")
    else:
        state = ("Concluído", "good")

    left, right = st.columns([3, 1])
    with left:
        label = next(
            (title for key, title in _STAGES if key == running_stage),
            "Aguardando…" if job.is_running() else "Encerrado",
        )
        st.markdown(
            f'<div class="eren-mono-label">Estágio atual</div>'
            f'<div style="font-family:Chakra Petch,sans-serif;font-size:1.15rem;'
            f'color:#e6edf3;margin-bottom:6px">{label}</div>',
            unsafe_allow_html=True,
        )
        st.progress(job.progress())
    with right:
        st.markdown(ui.badge_html(*state), unsafe_allow_html=True)
        st.markdown(
            f'<div class="eren-mono-label" style="margin-top:8px">Tempo</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:#8b98a9">'
            f"{job.elapsed():.1f}s</div>",
            unsafe_allow_html=True,
        )

    if failed is not None:
        st.error(f"Estágio **{failed.stage}** falhou: {failed.message}")

    events = job.events()
    if events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": event.sequence,
                        "Rodada": event.round,
                        "Evento": event.kind,
                        "Estágio": event.stage or "—",
                        "Status": event.status.value if event.status else "—",
                        "Duração": _duration(event.duration_seconds),
                        "Detalhe": event.message or event.artifact_ref or "",
                    }
                    for event in events
                ]
            ),
            hide_index=True,
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Execução salva                                                              #
# --------------------------------------------------------------------------- #
def _render_stages(record: LoopRecord) -> None:
    by_name = {stage.name: stage for stage in record.stages}

    rows = []
    for key, title in _STAGES:
        stage = by_name.get(key)
        status = stage.status if stage else LoopStageStatus.PENDING
        text, _kind = _STATUS_BADGE[status]
        artifact = stage.artifact_ref if stage else None
        rows.append(
            {
                "Estágio": title,
                "Status": text,
                "Duração": _duration(stage.duration_seconds if stage else None),
                "Artefato": Path(artifact).name if artifact else "—",
                "Erro": (stage.error if stage and stage.error else ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    failed = [stage for stage in record.stages if stage.status is LoopStageStatus.FAILED]
    for stage in failed:
        st.error(f"Estágio **{stage.name}** falhou: {stage.error}")


def _render_outcome(record: LoopRecord) -> None:
    outcome = outcome_for(record)

    if not outcome.has_result:
        st.info(
            "Esta execução não chegou a produzir um DetectionReport — o "
            "resultado só existe a partir do estágio de detecção."
        )
        return

    report = outcome.detection_report
    assert report is not None  # garantido por has_result
    cards = [
        {"label": "Detector", "value": report.model_name},
        {"label": "F1", "value": f"{report.f1:.4f}", "accent": "#35c07a"},
        {"label": "Recall", "value": f"{report.recall:.4f}", "accent": "#3987e5"},
        {"label": "Precisão", "value": f"{report.precision:.4f}"},
        {"label": "Latência", "value": f"{report.latency_ms:.1f}ms", "accent": "#f5a524"},
    ]
    ui.kpi_row(cards)

    matrix = report.confusion_matrix
    ui.confusion_matrix(matrix.tn, matrix.fp, matrix.fn, matrix.tp)

    if report.top_features:
        st.markdown("**Features mais influentes**")
        ui.feature_bars(
            [feature.model_dump() for feature in report.top_features[:8]]
        )

    plan = outcome.defense_plan
    rules = outcome.defense_rules
    if plan is None:
        st.info("Esta execução não chegou a produzir um DefensePlan.")
        return

    st.markdown("**Plano defensivo**")
    grounded = (
        f"{rules.grounded_actions}/{rules.total_actions} ações com lastro"
        if rules is not None
        else "avaliação por regras indisponível"
    )
    ui.badges(
        [
            (f"prioridade: {plan.priority}", "warn"),
            (grounded, "good" if rules is None or rules.is_grounded else "bad"),
        ]
    )

    actions = [
        {
            "Balde": bucket,
            "Técnica": action.technique,
            "Ação": action.description,
            "Evidência": ", ".join(
                item.metric_or_feature for item in action.evidence
            ),
            "Teste": f"{action.validation_test.metric} "
            f"{action.validation_test.direction} {action.validation_test.target}",
        }
        for bucket, bucket_actions in (
            ("detecção", plan.detection_actions),
            ("contenção", plan.containment_actions),
            ("hardening", plan.hardening_actions),
        )
        for action in bucket_actions
    ]
    st.dataframe(pd.DataFrame(actions), hide_index=True, width="stretch")

    if rules is not None and rules.findings:
        with st.expander(f"Achados da avaliação por regras ({len(rules.findings)})"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Regra": finding.rule,
                            "Severidade": finding.severity,
                            "Estágio": finding.bucket or "plano",
                            "Mensagem": finding.message,
                        }
                        for finding in rules.findings
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def _render_saved() -> None:
    records = saved_records(LOOP_RECORDS_PATH)

    if not records:
        st.info(
            f"Nenhuma execução salva em `{LOOP_RECORDS_PATH.name}`. Dispare uma "
            "acima, ou rode `adversarial-ids --engine intent --prompt \"...\"`."
        )
        return

    def _label(record: LoopRecord) -> str:
        rodada = f" · rodada {record.round}" if record.round > 1 else ""
        return f"{record.created_at[:19]} · {record.run_id}{rodada}"

    chosen = st.selectbox(
        "Execução",
        options=records,
        format_func=_label,
        key="loop_selected_run",
    )

    ui.kpi_row(
        [
            {"label": "run_id", "value": chosen.run_id},
            {"label": "Rodada", "value": chosen.round},
            {"label": "Seed", "value": chosen.seed},
            {
                "label": "Duração total",
                "value": _duration(chosen.total_duration_seconds),
                "accent": "#f5a524",
            },
            {
                "label": "Tokens",
                "value": (
                    "—" if chosen.total_tokens is None else f"{chosen.total_tokens:,}"
                ),
                "hint": (
                    "não informado pelo provedor"
                    if chosen.total_tokens is None
                    else (
                        "custo não informado"
                        if chosen.cost_usd is None
                        else f"US$ {chosen.cost_usd:.4f}"
                    )
                ),
            },
        ]
    )
    st.markdown(
        f'<div class="eren-mono-label">Intenção de origem</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:#e6edf3;'
        f'font-size:0.86rem">{chosen.source_prompt}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    _render_stages(chosen)

    run_dir = run_directory(chosen)
    if run_dir is not None:
        st.caption(f"Artefatos em `{run_dir}`")

    events = timeline_for(chosen)
    if events:
        with st.expander(f"Timeline gravada ({len(events)} eventos)"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "#": event.sequence,
                            "Evento": event.kind,
                            "Estágio": event.stage or "—",
                            "Status": event.status.value if event.status else "—",
                            "Duração": _duration(event.duration_seconds),
                            "Quando": event.created_at[:19],
                        }
                        for event in events
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    _render_outcome(chosen)


# --------------------------------------------------------------------------- #
# Página                                                                      #
# --------------------------------------------------------------------------- #
def _has_groq() -> bool:
    import os

    return bool(os.getenv("GROQ_API_KEY", "").strip())


def render() -> None:
    ui.hero(
        kicker="Pipeline intent-driven · E3 → E10",
        title_html='Intenção → <span class="accent">defesa</span>',
        subtitle="Descreva em português o que o Red Team deve tentar. O loop "
        "compila a intenção, gera o tráfego, avalia o IDS e devolve um plano "
        "defensivo — com cada estágio, duração, artefato e erro à vista.",
    )

    ui.section("Disparar", "Uma intenção, uma campanha")

    prompt = st.text_area(
        "Intenção em linguagem natural",
        placeholder="reduza o recall do detector variando a temporização da falha",
        key="loop_prompt",
    )
    col_rounds, col_gen, col_det = st.columns(3)
    rounds = col_rounds.number_input(
        "Rodadas", min_value=1, max_value=5, value=1, step=1, key="loop_rounds"
    )
    generator_mode = col_gen.selectbox(
        "Modo do gerador", ("cached", "jar"), key="loop_generator"
    )
    detector = col_det.selectbox(
        "Detector",
        ("random_forest", "decision_tree", "svm_linear", "svm_rbf"),
        key="loop_detector",
    )

    job: IntentLoopJob | None = st.session_state.get("loop_job")
    busy = job is not None and job.is_running()

    c1, c2 = st.columns([1, 1])
    start = c1.button(
        "⏵ Rodar campanha", type="primary", disabled=busy, width="stretch"
    )
    if c2.button("↺ Limpar", disabled=busy, width="stretch"):
        st.session_state["loop_job"] = None
        st.rerun()

    if start:
        if not prompt.strip():
            st.error("Descreva a intenção antes de disparar a campanha.")
        elif not _has_groq():
            st.error(
                "`GROQ_API_KEY` não detectada — o loop intent-driven chama a "
                "Groq para interpretar a intenção. Configure a chave no `.env`."
            )
        else:
            job = IntentLoopJob(
                IntentLoopConfig(
                    prompt=prompt.strip(),
                    rounds=int(rounds),
                    generator_mode=generator_mode,
                    detector=detector,
                )
            ).start()
            st.session_state["loop_job"] = job

    if job is not None:
        ui.section("Telemetria", "Estágio, duração e causa — ao vivo")
        placeholder = st.empty()
        while job.is_running():
            with placeholder.container():
                _render_live(job)
            time.sleep(_POLL_SECONDS)
        with placeholder.container():
            _render_live(job)

        if job.error:
            st.error(f"A campanha não pôde ser executada: {job.error}")
        elif job.records:
            st.success(
                f"Campanha concluída — {len(job.records)} rodada(s). "
                "O resultado está abaixo, junto das execuções salvas."
            )

    ui.section("Execuções", "Estágio, duração, artefatos, erro e resultado")
    _render_saved()
