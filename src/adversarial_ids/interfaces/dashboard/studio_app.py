"""ERENO AI — entrypoint do frontend Streamlit.

Execute com:

    uv run --extra dashboard streamlit run \
        src/adversarial_ids/interfaces/dashboard/studio_app.py

Frontend completo do loop adversarial: Visão Geral → Configuração → Execução
(logs ao vivo) → Resultados, mais a página do pipeline intent-driven (E11), onde
uma intenção em português vira campanha e cada estágio se mostra ao vivo.
Substitui o uso via terminal — tudo pela UI.
"""

from __future__ import annotations

import streamlit as st

from adversarial_ids.interfaces.dashboard.studio import state, theme
from adversarial_ids.interfaces.dashboard.studio import (
    view_configure,
    view_home,
    view_loop,
    view_results,
    view_run,
)


def _sidebar_brand() -> None:
    st.markdown(
        """
        <div class="eren-brand">
          <div class="logo">🛡️</div>
          <div>
            <div class="bt">ERENO AI</div>
            <div class="bs">Adversarial IDS Lab</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eren-mono-label" style="margin:14px 0 2px">Smart Grid · IEC-61850</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ERENO AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject()
    state.init()

    pages = [
        st.Page(view_home.render, title="Visão Geral", icon="🛰️", default=True, url_path="visao"),
        st.Page(view_configure.render, title="Configuração", icon="🎛️", url_path="config"),
        st.Page(view_run.render, title="Execução", icon="▶️", url_path="run"),
        st.Page(view_results.render, title="Resultados", icon="📊", url_path="resultados"),
        st.Page(view_loop.render, title="Loop intent-driven", icon="🧭", url_path="loop"),
    ]

    with st.sidebar:
        _sidebar_brand()

    nav = st.navigation(pages, position="sidebar")

    with st.sidebar:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        records = state.get_records()
        job = st.session_state.get("job")
        status = "sem dados"
        kind = ""
        if job is not None and job.is_running():
            status, kind = "executando…", "warn"
        elif records:
            status, kind = f"{len(records)} registro(s)", "good"
        from adversarial_ids.interfaces.dashboard.studio import components as ui
        st.markdown(
            '<div class="eren-mono-label" style="margin-top:6px">Sessão</div>',
            unsafe_allow_html=True,
        )
        st.markdown(ui.badge_html(status, kind), unsafe_allow_html=True)
        st.markdown(
            '<div style="position:fixed;bottom:14px;left:16px;font-family:JetBrains Mono,'
            'monospace;font-size:0.64rem;color:#5c6b7d">Red Team × Blue Team · Agno</div>',
            unsafe_allow_html=True,
        )

    nav.run()


if __name__ == "__main__":
    main()
