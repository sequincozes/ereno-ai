"""Gráficos do ERENO AI Studio (Altair sobre superfície escura).

Paleta de séries validada pela skill de dataviz (blue/orange/aqua, ``--mode
dark``). Um eixo só, legenda quando ≥ 2 séries, hover com crosshair e tooltip,
marcas finas — conforme as não-negociáveis da skill.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from . import PALETTE

_SURFACE = PALETTE["surface"]
_GRID = "rgba(120,160,200,0.12)"
_AXIS = PALETTE["text_dim"]
_TEXT = PALETTE["text_muted"]
_MONO = "JetBrains Mono, monospace"

# blue, orange, aqua — validadas para dark
_SERIES_ORDER = ["F1 (ataque)", "Precisão", "Recall"]
_SERIES_COLORS = [PALETTE["series_1"], PALETTE["series_2"], PALETTE["series_3"]]


def _base_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(
        strokeWidth=0, fill=_SURFACE
    ).configure_axis(
        gridColor=_GRID,
        domainColor=_AXIS,
        tickColor=_AXIS,
        labelColor=_TEXT,
        titleColor=_TEXT,
        labelFont=_MONO,
        titleFont=_MONO,
        labelFontSize=11,
        titleFontSize=11,
    ).configure_legend(
        labelColor=PALETTE["text"],
        titleColor=_TEXT,
        labelFont="IBM Plex Sans, sans-serif",
        titleFont=_MONO,
        symbolType="stroke",
        symbolStrokeWidth=3,
    ).configure(background="rgba(0,0,0,0)")


def metrics_evolution(records) -> alt.LayerChart | None:
    """Evolução de F1/Precisão/Recall da classe de ataque por iteração."""
    rows = []
    for r in records:
        m = r.metrics
        for label, value in (
            ("F1 (ataque)", m.f1_score_attack),
            ("Precisão", m.precision_attack),
            ("Recall", m.recall_attack),
        ):
            if value is not None:
                rows.append({"Iteração": r.iteration, "Métrica": label, "Valor": float(value)})
    if not rows:
        return None
    df = pd.DataFrame(rows)

    x = alt.X("Iteração:O", title="Iteração", axis=alt.Axis(labelAngle=0))
    color = alt.Color(
        "Métrica:N",
        scale=alt.Scale(domain=_SERIES_ORDER, range=_SERIES_COLORS),
        legend=alt.Legend(title="Série", orient="top"),
    )

    hover = alt.selection_point(fields=["Iteração"], nearest=True, on="mouseover", empty=False)

    line = alt.Chart(df).mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=x,
        y=alt.Y("Valor:Q", title="Valor", scale=alt.Scale(domain=[0, 1.02])),
        color=color,
    )
    points = alt.Chart(df).mark_point(size=64, filled=True, opacity=1).encode(
        x=x, y="Valor:Q", color=color,
    )
    rule = alt.Chart(df).mark_rule(color="rgba(120,160,200,0.4)").encode(
        x=x,
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[
            alt.Tooltip("Iteração:O"),
            alt.Tooltip("Métrica:N"),
            alt.Tooltip("Valor:Q", format=".4f"),
        ],
    ).add_params(hover)

    layered = (line + points + rule).properties(height=300)
    return _base_theme(layered)


def attack_ratio_bars(records) -> alt.Chart | None:
    """Razão de contagem de ataque vs. baseline por iteração (sequência)."""
    rows = []
    for r in records:
        ratio = r.metrics.attack_count_ratio_vs_baseline
        if ratio is not None:
            rows.append({"Iteração": r.iteration, "Razão": float(ratio)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(size=26, cornerRadiusEnd=4, color=PALETTE["series_1"]).encode(
        x=alt.X("Iteração:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Razão:Q", title="Ataque / baseline"),
        tooltip=[alt.Tooltip("Iteração:O"), alt.Tooltip("Razão:Q", format=".2%")],
    ).properties(height=220)
    return _base_theme(bars)
