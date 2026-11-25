"""Componentes de UI do ERENO AI.

Peças de apresentação renderizadas como HTML (via ``st.markdown``) para termos
controle estético total, independente dos internals do Streamlit. Todas as
funções escrevem direto na página; helpers ``*_html`` devolvem string.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

from . import PALETTE


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Hero e cabeçalhos                                                            #
# --------------------------------------------------------------------------- #
def hero(
    *,
    kicker: str,
    title_html: str,
    subtitle: str,
    show_teams: bool = False,
) -> None:
    teams = ""
    if show_teams:
        teams = f"""
        <div class="eren-vs">
          <div class="eren-team red">
            <div class="tag">▲ Red Team · Estrategista</div>
            <div class="name">Propõe a variante do ataque</div>
            <div class="desc">A cada iteração altera parâmetros do ataque sintético
              (temporização, campos GOOSE, intensidade) buscando derrubar a detecção do IDS.</div>
          </div>
          <div class="eren-team blue">
            <div class="tag">◆ Blue Team · Analista</div>
            <div class="name">Explica e recomenda mitigação</div>
            <div class="desc">Interpreta as métricas, aponta quais features enganaram o
              detector e sugere defesas — com a saída validada contra as próprias métricas.</div>
          </div>
        </div>"""
    st.markdown(
        f"""
        <div class="eren-hero">
          <div class="eren-kicker"><span class="dot"></span>{_esc(kicker)}</div>
          <h1>{title_html}</h1>
          <p>{_esc(subtitle)}</p>
          {teams}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(eyebrow: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="sub">{_esc(sub)}</p>' if sub else ""
    st.markdown(
        f"""
        <div class="eren-section">
          <div class="eyebrow">{_esc(eyebrow)}</div>
          <h2>{_esc(title)}</h2>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# KPIs                                                                          #
# --------------------------------------------------------------------------- #
def kpi_row(cards: Sequence[dict[str, Any]]) -> None:
    """Renderiza uma grade de KPIs.

    Cada card: ``{label, value, delta?, delta_dir?('up'|'down'|'flat'),
    hint?, accent?}``.
    """
    items = []
    for c in cards:
        accent = c.get("accent", PALETTE["cyan"])
        delta_html = ""
        if c.get("delta") is not None:
            direction = c.get("delta_dir", "flat")
            arrow = {"up": "▲", "down": "▼", "flat": "→"}.get(direction, "→")
            delta_html = f'<div class="k-delta {direction}">{arrow} {_esc(c["delta"])}</div>'
        hint_html = f'<div class="k-hint">{_esc(c["hint"])}</div>' if c.get("hint") else ""
        items.append(
            f"""
            <div class="eren-kpi" style="--accent:{accent}">
              <div class="k-label">{_esc(c['label'])}</div>
              <div class="k-value">{_esc(c['value'])}</div>
              {delta_html}
              {hint_html}
            </div>"""
        )
    st.markdown(f'<div class="eren-kpis">{"".join(items)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Badges                                                                        #
# --------------------------------------------------------------------------- #
def badge_html(text: str, kind: str = "") -> str:
    return f'<span class="eren-badge {_esc(kind)}">{_esc(text)}</span>'


def badges(items: Iterable[tuple[str, str]]) -> None:
    html_parts = " ".join(badge_html(t, k) for t, k in items)
    st.markdown(f'<div style="display:flex;gap:8px;flex-wrap:wrap">{html_parts}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Passos numerados (fluxo)                                                      #
# --------------------------------------------------------------------------- #
def steps(items: Sequence[dict[str, str]]) -> None:
    rows = []
    for i, it in enumerate(items, start=1):
        cls = "red" if it.get("team") == "red" else ""
        rows.append(
            f"""
            <div class="eren-step {cls}">
              <div class="num">{it.get('num', i)}</div>
              <div>
                <div class="st-title">{_esc(it['title'])}</div>
                <div class="st-body">{_esc(it['body'])}</div>
              </div>
            </div>"""
        )
    st.markdown(f'<div class="eren-steps">{"".join(rows)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Catálogo de ataques                                                          #
# --------------------------------------------------------------------------- #
def attack_card(code: str, name: str, desc: str) -> str:
    return f"""
    <div class="eren-attack">
      <div class="code">{_esc(code)}</div>
      <div class="an">{_esc(name)}</div>
      <div class="ad">{_esc(desc)}</div>
    </div>"""


# --------------------------------------------------------------------------- #
# Matriz de confusão                                                           #
# --------------------------------------------------------------------------- #
def confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> None:
    def cell(value: int, tag: str, ok: bool) -> str:
        cls = "correct" if ok else "error"
        return f'<div class="cell {cls}"><div class="v">{value:,}</div><div class="t">{tag}</div></div>'.replace(",", ".")

    st.markdown(
        f"""
        <div class="eren-cm">
          <div class="lbl"></div>
          <div class="lbl">Previsto<br>normal</div>
          <div class="lbl">Previsto<br>ataque</div>

          <div class="lbl">Real<br>normal</div>
          {cell(tn, 'TN', True)}
          {cell(fp, 'FP', False)}

          <div class="lbl">Real<br>ataque</div>
          {cell(fn, 'FN', False)}
          {cell(tp, 'TP', True)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Barras de importância de features                                           #
# --------------------------------------------------------------------------- #
def feature_bars(features: Sequence[dict[str, Any]]) -> None:
    if not features:
        st.markdown('<div class="eren-mono-label">Sem importâncias registradas.</div>',
                    unsafe_allow_html=True)
        return
    top = max((float(f.get("importance", 0)) for f in features), default=1.0) or 1.0
    rows = []
    for f in features:
        imp = float(f.get("importance", 0))
        width = max(3.0, imp / top * 100.0)
        rows.append(
            f"""
            <div class="row">
              <div class="name" title="{_esc(f.get('feature',''))}">{_esc(f.get('feature',''))}</div>
              <div class="track"><div class="fill" style="width:{width:.1f}%"></div></div>
              <div class="pct">{imp:.3f}</div>
            </div>"""
        )
    st.markdown(f'<div class="eren-feat">{"".join(rows)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Terminal de logs                                                             #
# --------------------------------------------------------------------------- #
_CHANNELS = {
    "[ORCH]": ("orch", "ORCH"),
    "[VALIDATOR]": ("valid", "VALIDATOR"),
    "[DEMO]": ("demo", "DEMO"),
    "[ERRO]": ("err", "ERRO"),
    "[ERROR]": ("err", "ERROR"),
}


def _classify(line: str) -> tuple[str, str, str]:
    stripped = line.strip()
    for prefix, (cls, label) in _CHANNELS.items():
        if stripped.startswith(prefix):
            return cls, label, stripped[len(prefix):].strip()
    low = stripped.lower()
    if "traceback" in low or "error" in low or "exception" in low or "falh" in low:
        return "err", "ERRO", stripped
    return "sys", "•", stripped


def terminal(lines: Sequence[str], *, title: str = "ereno://execução — stdout") -> None:
    body = []
    for i, raw in enumerate(lines, start=1):
        if raw == "":
            continue
        cls, label, msg = _classify(raw)
        body.append(
            f'<div class="ln {cls}"><span class="gutter">{i:03d}</span>'
            f'<span class="chan">{_esc(label)}</span>'
            f'<span class="msg">{_esc(msg)}</span></div>'
        )
    if not body:
        body.append('<div class="ln sys"><span class="gutter">···</span>'
                     '<span class="chan">•</span><span class="msg">aguardando saída…</span></div>')
    st.markdown(
        f"""
        <div class="eren-term">
          <div class="bar">
            <span class="lamp r"></span><span class="lamp y"></span><span class="lamp g"></span>
            <span class="title">{_esc(title)}</span>
          </div>
          <div class="body" id="eren-term-body">{"".join(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str = "") -> None:
    t = f'<div class="eren-mono-label" style="margin-bottom:10px">{_esc(title)}</div>' if title else ""
    st.markdown(f'<div class="eren-card">{t}', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
