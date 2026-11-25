"""Injeção do design system do ERENO AI.

Toda a estética vive aqui: fontes (Chakra Petch / IBM Plex Sans / JetBrains
Mono), a paleta de sala de controle de subestação, a atmosfera de fundo (grid +
glow) e o *override* dos widgets nativos do Streamlit. Chame :func:`inject`
uma vez no topo de cada página.
"""

from __future__ import annotations

import streamlit as st

from . import PALETTE

_FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Chakra+Petch:wght@400;500;600;700&"
    "family=IBM+Plex+Sans:wght@300;400;500;600&"
    "family=JetBrains+Mono:wght@400;500;700&display=swap');"
)


def _css() -> str:
    p = PALETTE
    return f"""
{_FONTS}

:root {{
    --bg: {p['bg']};
    --surface: {p['surface']};
    --surface-2: {p['surface_2']};
    --surface-3: {p['surface_3']};
    --border: {p['border']};
    --border-strong: {p['border_strong']};
    --text: {p['text']};
    --muted: {p['text_muted']};
    --dim: {p['text_dim']};
    --red: {p['red']};
    --cyan: {p['cyan']};
    --amber: {p['amber']};
    --good: {p['good']};
    --font-display: 'Chakra Petch', sans-serif;
    --font-body: 'IBM Plex Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}}

/* ---------- base ---------- */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
    font-family: var(--font-body);
    color: var(--text);
}}

.stApp {{
    background: var(--bg);
    background-image:
        radial-gradient(1100px 520px at 82% -8%, rgba(52,211,238,0.10), transparent 60%),
        radial-gradient(900px 480px at 5% 4%, rgba(255,70,85,0.07), transparent 55%),
        linear-gradient(rgba(120,160,200,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120,160,200,0.045) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 46px 46px, 46px 46px;
    background-attachment: fixed;
}}

/* topo/hamburguer discretos */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}

.block-container {{
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}}

/* ---------- tipografia ---------- */
h1, h2, h3, h4 {{
    font-family: var(--font-display) !important;
    letter-spacing: 0.2px;
    color: var(--text);
}}
h1 {{ font-weight: 700; }}
p, li, span, label, div {{ font-family: var(--font-body); }}

a, a:visited {{ color: var(--cyan); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

code, pre, .stCode {{ font-family: var(--font-mono) !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0c121b 0%, #0a0e14 100%);
    border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

/* navegação (st.navigation) — estilizada, não escondida */
[data-testid="stSidebarNav"] ul {{ gap: 2px; }}
[data-testid="stSidebarNav"] a {{
    border-radius: 9px;
    font-family: var(--font-display);
    letter-spacing: 0.3px;
}}
[data-testid="stSidebarNav"] a:hover {{
    background: var(--surface-2);
}}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: linear-gradient(90deg, rgba(52,211,238,0.16), transparent);
    border-left: 2px solid var(--cyan);
}}

/* ---------- botões ---------- */
.stButton > button, .stDownloadButton > button {{
    font-family: var(--font-display);
    font-weight: 600;
    letter-spacing: 0.4px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--surface-2);
    color: var(--text);
    transition: all .16s ease;
    padding: 0.5rem 1.05rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: var(--cyan);
    color: #ffffff;
    box-shadow: 0 0 0 1px rgba(52,211,238,0.35), 0 8px 26px -14px rgba(52,211,238,0.8);
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, rgba(52,211,238,0.22), rgba(57,135,229,0.22));
    border-color: rgba(52,211,238,0.55);
    color: #eafcff;
}}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, rgba(52,211,238,0.34), rgba(57,135,229,0.30));
    box-shadow: 0 10px 30px -12px rgba(52,211,238,0.9);
}}

/* ---------- inputs / selects ---------- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 9px !important;
    font-family: var(--font-body);
}}
[data-baseweb="select"] > div:focus-within {{
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px rgba(52,211,238,0.4) !important;
}}
.stRadio label, .stCheckbox label, .stSelectbox label, .stNumberInput label,
.stSlider label, .stTextInput label {{
    font-family: var(--font-display) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--muted) !important;
}}

/* slider na cor do tema */
[data-baseweb="slider"] [role="slider"] {{ background: var(--cyan) !important; }}

/* ---------- métricas nativas (fallback) ---------- */
[data-testid="stMetric"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
}}
[data-testid="stMetricValue"] {{ font-family: var(--font-display); }}

/* ---------- dataframes ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border);
    border-radius: 12px;
}}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    font-family: var(--font-display);
    letter-spacing: 0.4px;
    color: var(--muted);
    background: transparent;
    border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan);
}}

/* ---------- expander ---------- */
[data-testid="stExpander"] {{
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
}}

/* ---------- alerts (info/success/warn/error) refinados ---------- */
[data-testid="stAlert"] {{ border-radius: 12px; font-family: var(--font-body); }}

/* ---------- divisor ---------- */
hr {{ border-color: var(--border); }}

/* animação de entrada */
@keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* =======================================================================
   Componentes próprios (render via st.markdown unsafe_allow_html)
   ======================================================================= */

.eren-hero {{
    position: relative;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 34px 34px 30px;
    background:
        radial-gradient(120% 140% at 100% 0%, rgba(52,211,238,0.10), transparent 55%),
        radial-gradient(120% 140% at 0% 100%, rgba(255,70,85,0.08), transparent 55%),
        var(--surface);
    overflow: hidden;
    animation: riseIn .5s ease both;
}}
.eren-hero::before {{
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(120,160,200,0.05) 1px, transparent 1px) 0 0 / 30px 100%,
                linear-gradient(rgba(120,160,200,0.05) 1px, transparent 1px) 0 0 / 100% 30px;
    mask: radial-gradient(80% 80% at 50% 0%, #000, transparent 78%);
    pointer-events: none;
}}
.eren-hero > * {{ position: relative; z-index: 1; }}
.eren-kicker {{
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 12px;
    display: inline-flex; align-items: center; gap: 9px;
}}
.eren-kicker .dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px 2px rgba(52,211,238,0.8);
    animation: pulse 1.8s ease-in-out infinite;
}}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
.eren-hero h1 {{
    font-size: 2.5rem;
    line-height: 1.05;
    margin: 0 0 12px;
    font-weight: 700;
}}
.eren-hero h1 .accent {{ color: var(--cyan); }}
.eren-hero p {{ color: var(--muted); font-size: 1.02rem; max-width: 62ch; margin: 0; }}

/* faixas Red x Blue */
.eren-vs {{ display: flex; gap: 14px; margin-top: 22px; flex-wrap: wrap; }}
.eren-team {{
    flex: 1 1 220px;
    border-radius: 13px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    background: var(--surface-2);
}}
.eren-team.red {{ border-left: 3px solid var(--red); }}
.eren-team.blue {{ border-left: 3px solid var(--cyan); }}
.eren-team .tag {{
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 2px;
    text-transform: uppercase;
}}
.eren-team.red .tag {{ color: var(--red); }}
.eren-team.blue .tag {{ color: var(--cyan); }}
.eren-team .name {{ font-family: var(--font-display); font-weight: 600; font-size: 1.05rem; margin: 4px 0 5px; }}
.eren-team .desc {{ color: var(--muted); font-size: 0.86rem; line-height: 1.45; }}

/* cabeçalho de seção */
.eren-section {{ margin: 30px 0 14px; animation: riseIn .5s ease both; }}
.eren-section .eyebrow {{
    font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--dim);
}}
.eren-section h2 {{ font-size: 1.5rem; margin: 3px 0 2px; }}
.eren-section .sub {{ color: var(--muted); font-size: 0.92rem; margin: 0; }}

/* grade de KPIs */
.eren-kpis {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
.eren-kpi {{
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px 17px;
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    position: relative; overflow: hidden;
    animation: riseIn .5s ease both;
}}
.eren-kpi::after {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, var(--cyan)); opacity: 0.85;
}}
.eren-kpi .k-label {{
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted);
}}
.eren-kpi .k-value {{
    font-family: var(--font-display); font-weight: 700; font-size: 1.85rem;
    line-height: 1.1; margin-top: 6px; color: var(--text);
    font-variant-numeric: tabular-nums;
}}
.eren-kpi .k-delta {{ font-family: var(--font-mono); font-size: 0.78rem; margin-top: 4px; }}
.eren-kpi .k-delta.up {{ color: var(--good); }}
.eren-kpi .k-delta.down {{ color: var(--red); }}
.eren-kpi .k-delta.flat {{ color: var(--dim); }}
.eren-kpi .k-hint {{ font-size: 0.72rem; color: var(--dim); margin-top: 2px; }}

/* card genérico */
.eren-card {{
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
    padding: 18px 20px;
    animation: riseIn .5s ease both;
}}

/* badges */
.eren-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.5px;
    padding: 3px 10px; border-radius: 999px;
    border: 1px solid var(--border-strong); background: var(--surface-2);
    color: var(--muted);
}}
.eren-badge.good {{ color: var(--good); border-color: rgba(53,192,122,0.4); background: rgba(53,192,122,0.08); }}
.eren-badge.warn {{ color: var(--amber); border-color: rgba(245,165,36,0.4); background: rgba(245,165,36,0.08); }}
.eren-badge.bad  {{ color: var(--red);  border-color: rgba(255,70,85,0.4);  background: rgba(255,70,85,0.08); }}
.eren-badge.red  {{ color: var(--red);  border-color: rgba(255,70,85,0.4); }}
.eren-badge.blue {{ color: var(--cyan); border-color: rgba(52,211,238,0.4); }}

/* passos numerados */
.eren-steps {{ display: grid; gap: 12px; }}
.eren-step {{
    display: grid; grid-template-columns: 42px 1fr; gap: 14px; align-items: start;
    border: 1px solid var(--border); border-radius: 13px; padding: 14px 16px;
    background: var(--surface);
}}
.eren-step .num {{
    font-family: var(--font-display); font-weight: 700; font-size: 1.1rem;
    width: 42px; height: 42px; display: grid; place-items: center;
    border-radius: 10px; color: var(--bg);
    background: linear-gradient(135deg, var(--cyan), var(--series-blue, #3987e5));
}}
.eren-step.red .num {{ background: linear-gradient(135deg, var(--red), #b3122b); color: #fff; }}
.eren-step .st-title {{ font-family: var(--font-display); font-weight: 600; margin: 2px 0 3px; }}
.eren-step .st-body {{ color: var(--muted); font-size: 0.88rem; line-height: 1.5; }}

/* terminal de logs */
.eren-term {{
    border: 1px solid var(--border-strong);
    border-radius: 13px; overflow: hidden;
    background: #070b11;
    box-shadow: inset 0 0 60px rgba(0,0,0,0.5);
}}
.eren-term .bar {{
    display: flex; align-items: center; gap: 8px;
    padding: 9px 14px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0c121b, #080d14);
}}
.eren-term .bar .lamp {{ width: 11px; height: 11px; border-radius: 50%; }}
.eren-term .bar .lamp.r {{ background: #ff5f57; }}
.eren-term .bar .lamp.y {{ background: #febc2e; }}
.eren-term .bar .lamp.g {{ background: #28c840; }}
.eren-term .bar .title {{
    font-family: var(--font-mono); font-size: 0.74rem; color: var(--muted);
    margin-left: 6px; letter-spacing: 0.5px;
}}
.eren-term .body {{
    padding: 14px 16px; max-height: 460px; overflow-y: auto;
    font-family: var(--font-mono); font-size: 0.82rem; line-height: 1.62;
}}
.eren-term .ln {{ display: flex; gap: 10px; white-space: pre-wrap; word-break: break-word; }}
.eren-term .ln .gutter {{ color: #33404f; user-select: none; min-width: 30px; text-align: right; }}
.eren-term .ln .chan {{ min-width: 92px; font-weight: 700; }}
.eren-term .ln .msg {{ color: #c6d3e0; flex: 1; }}
.eren-term .ln.orch  .chan {{ color: var(--cyan); }}
.eren-term .ln.valid .chan {{ color: var(--amber); }}
.eren-term .ln.demo  .chan {{ color: #9085e9; }}
.eren-term .ln.err   .chan {{ color: var(--red); }}
.eren-term .ln.err   .msg  {{ color: #ffb3ba; }}
.eren-term .ln.sys   .chan {{ color: var(--dim); }}
.eren-term .body::-webkit-scrollbar {{ width: 9px; }}
.eren-term .body::-webkit-scrollbar-thumb {{ background: #1c2735; border-radius: 6px; }}

/* matriz de confusão */
.eren-cm {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; max-width: 520px; }}
.eren-cm .lbl {{
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted);
    display: grid; place-items: center; text-align: center; padding: 6px;
}}
.eren-cm .cell {{
    border-radius: 11px; padding: 16px 10px; text-align: center;
    border: 1px solid var(--border);
}}
.eren-cm .cell .v {{ font-family: var(--font-display); font-weight: 700; font-size: 1.5rem; font-variant-numeric: tabular-nums; }}
.eren-cm .cell .t {{ font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 1px; text-transform: uppercase; margin-top: 3px; }}
.eren-cm .cell.correct {{ background: rgba(53,192,122,0.10); border-color: rgba(53,192,122,0.35); }}
.eren-cm .cell.correct .v {{ color: var(--good); }}
.eren-cm .cell.correct .t {{ color: rgba(53,192,122,0.8); }}
.eren-cm .cell.error {{ background: rgba(255,70,85,0.09); border-color: rgba(255,70,85,0.3); }}
.eren-cm .cell.error .v {{ color: var(--red); }}
.eren-cm .cell.error .t {{ color: rgba(255,120,130,0.85); }}

/* barras de importância de features */
.eren-feat {{ display: grid; gap: 9px; }}
.eren-feat .row {{ display: grid; grid-template-columns: 200px 1fr 62px; gap: 12px; align-items: center; }}
.eren-feat .name {{ font-family: var(--font-mono); font-size: 0.8rem; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.eren-feat .track {{ height: 12px; border-radius: 6px; background: var(--surface-3); overflow: hidden; }}
.eren-feat .fill {{ height: 100%; border-radius: 6px; background: linear-gradient(90deg, #256abf, #3987e5); }}
.eren-feat .pct {{ font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }}

/* card de ataque (catálogo) */
.eren-attack {{
    border: 1px solid var(--border); border-radius: 13px; padding: 15px 16px;
    background: var(--surface); height: 100%;
}}
.eren-attack .code {{ font-family: var(--font-mono); font-size: 0.68rem; color: var(--red); letter-spacing: 1px; }}
.eren-attack .an {{ font-family: var(--font-display); font-weight: 600; font-size: 1rem; margin: 3px 0 6px; }}
.eren-attack .ad {{ color: var(--muted); font-size: 0.83rem; line-height: 1.45; }}

/* rótulo pequeno mono */
.eren-mono-label {{ font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--dim); }}

/* nav lateral custom */
.eren-brand {{ display:flex; align-items:center; gap:11px; margin-bottom: 6px; }}
.eren-brand .logo {{
    width: 38px; height: 38px; border-radius: 10px; flex: none;
    background: linear-gradient(135deg, rgba(52,211,238,0.25), rgba(255,70,85,0.22));
    border: 1px solid var(--border-strong);
    display: grid; place-items: center; font-size: 1.2rem;
}}
.eren-brand .bt {{ font-family: var(--font-display); font-weight: 700; font-size: 1.05rem; line-height: 1; }}
.eren-brand .bs {{ font-family: var(--font-mono); font-size: 0.62rem; letter-spacing: 2px; color: var(--dim); text-transform: uppercase; }}
"""


def inject() -> None:
    """Injeta o CSS do design system. Idempotente por rerun do Streamlit."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)
