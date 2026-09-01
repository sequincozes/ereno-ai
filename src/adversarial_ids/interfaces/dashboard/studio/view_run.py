"""Página · Execução — dispara o loop e transmite os logs ao vivo."""

from __future__ import annotations

import time

import streamlit as st

from . import components as ui
from .bridge import ExperimentConfig, ExperimentJob
from .state import get_config, set_records

_POLL_SECONDS = 0.6


def _render_progress(job: ExperimentJob) -> None:
    pct, phase = job.progress()
    running = job.is_running()
    benign_status = job.benign_generation_status()
    if benign_status == "generating":
        st.warning(
            "O dataset benigno não foi encontrado. Ele está sendo gerado "
            "automaticamente a partir do seed local antes da execução do ataque."
        )
    elif benign_status == "created":
        st.success(
            "Dataset benigno gerado com sucesso. A execução do ataque pode continuar."
        )
    state_badge = ("Executando", "warn") if running else (
        ("Falhou", "bad") if job.error else ("Concluído", "good"))
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f'<div class="eren-mono-label">Fase atual</div>'
            f'<div style="font-family:Chakra Petch,sans-serif;font-size:1.15rem;'
            f'color:#e6edf3;margin-bottom:6px">{phase}</div>',
            unsafe_allow_html=True,
        )
        st.progress(pct / 100.0)
    with right:
        st.markdown(ui.badge_html(*state_badge), unsafe_allow_html=True)
        st.markdown(
            f'<div class="eren-mono-label" style="margin-top:8px">Tempo</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:#8b98a9">'
            f'{job.elapsed():.1f}s</div>',
            unsafe_allow_html=True,
        )


def render() -> None:
    cfg: ExperimentConfig = get_config()

    ui.hero(
        kicker="Passo 2 de 3 · Execução monitorada",
        title_html='Executar & <span class="accent">observar</span>',
        subtitle="Dispare o loop adversarial e acompanhe cada fase pelos logs "
                 "estruturados — baseline, propostas do Red Team, validação e avaliação do IDS.",
    )

    # resumo compacto da config
    cols = st.columns(len(cfg.as_summary()))
    for col, (k, v) in zip(cols, cfg.as_summary().items()):
        col.markdown(
            f'<div class="eren-mono-label">{k}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:#e6edf3;font-size:0.86rem">{v}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    job: ExperimentJob | None = st.session_state.get("job")
    busy = job is not None and job.is_running()

    c1, c2 = st.columns([1, 1])
    with c1:
        start = st.button("⏵ Iniciar execução", type="primary",
                          disabled=busy, width="stretch")
    with c2:
        reset = st.button("↺ Nova execução", disabled=busy, width="stretch")

    if reset:
        st.session_state["job"] = None
        st.rerun()

    if start:
        if cfg.engine == "live" and not _has_groq():
            st.error("`GROQ_API_KEY` não detectada — o motor ao vivo não pode iniciar. "
                     "Use o motor de demonstração ou configure a chave no `.env`.")
            return
        job = ExperimentJob(cfg).start()
        st.session_state["job"] = job

    if job is None:
        st.info("Nenhuma execução iniciada. Clique em **Iniciar execução** para começar. "
                "No motor de demonstração o resultado é imediato e reprodutível.")
        return

    ui.section("Telemetria", "Progresso e logs em tempo real")

    placeholder = st.empty()

    # loop de polling: atualiza o placeholder enquanto a thread vive
    while job.is_running():
        with placeholder.container():
            _render_progress(job)
            ui.terminal(job.logs())
        time.sleep(_POLL_SECONDS)

    # render final (já concluído)
    with placeholder.container():
        _render_progress(job)
        ui.terminal(job.logs())

    # persiste resultados e conclui
    if job.error:
        st.error(f"A execução falhou: {job.error}")
    elif job.records is not None:
        source = ("Demo golden" if cfg.engine == "demo"
                  else f"Ao vivo · {cfg.attack}")
        set_records(job.records, source)
        st.success(f"Execução concluída — {len(job.records)} registro(s) capturado(s). "
                   "Abra **Resultados** no menu para o dashboard completo.")
        final = job.records[-1]
        f1 = final.metrics.f1_score_attack
        f1_text = "indisponível" if f1 is None else f"{f1:.4f}"
        ui.kpi_row([
            {"label": "Registros", "value": len(job.records)},
            {"label": "Última iteração", "value": final.iteration, "accent": "#3987e5"},
            {"label": "F1 final (ataque)", "value": f1_text, "accent": "#35c07a"},
            {"label": "Duração", "value": f"{job.elapsed():.1f}s", "accent": "#f5a524"},
        ])


def _has_groq() -> bool:
    import os
    return bool(os.getenv("GROQ_API_KEY", "").strip())
