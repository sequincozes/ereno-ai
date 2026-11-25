"""Página · Resultados — dashboard de desempenho e explicabilidade."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from adversarial_ids.config.settings import GOLDEN_HISTORY_PATH, ITERATION_HISTORY_PATH
from adversarial_ids.domain import IterationRecord

from . import charts
from . import components as ui
from .state import get_records, load_history_file


# --------------------------------------------------------------------------- #
# Fontes de dados                                                             #
# --------------------------------------------------------------------------- #
def _source_picker() -> None:
    ui.section("Fonte", "De onde vêm os resultados",
               "Use a execução desta sessão ou carregue um histórico salvo.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.caption("Última execução salva pela CLI/loop (`outputs/`).")
        if st.button("Carregar última execução", width="stretch",
                     disabled=not ITERATION_HISTORY_PATH.exists()):
            load_history_file(ITERATION_HISTORY_PATH, "Última execução (outputs/)")
            st.rerun()
    with c2:
        st.caption("Histórico golden versionado (`data/`).")
        if st.button("Carregar demo golden", width="stretch",
                     disabled=not GOLDEN_HISTORY_PATH.exists()):
            load_history_file(GOLDEN_HISTORY_PATH, "Demo golden (data/)")
            st.rerun()
    with c3:
        st.caption("Resultado em memória desta sessão.")
        has = get_records() is not None
        st.markdown(ui.badge_html("Disponível" if has else "Nenhum",
                                  "good" if has else ""), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Blocos do dashboard                                                         #
# --------------------------------------------------------------------------- #
def _f1_series(records: list[IterationRecord]) -> list[tuple[int, float]]:
    return [(r.iteration, r.metrics.f1_score_attack)
            for r in sorted(records, key=lambda x: x.iteration)
            if r.metrics.f1_score_attack is not None]


def _summary(records: list[IterationRecord]) -> None:
    series = _f1_series(records)
    cards: list[dict[str, Any]] = [{"label": "Registros", "value": len(records), "accent": "#3987e5"}]
    if series:
        initial, final = series[0][1], series[-1][1]
        lowest_it, lowest = min(series, key=lambda t: t[1])
        delta = final - initial
        cards += [
            {"label": "F1 inicial", "value": f"{initial:.4f}"},
            {"label": "F1 final", "value": f"{final:.4f}",
             "accent": "#35c07a" if delta >= 0 else "#ff4655"},
            {"label": "Variação (Δ)", "value": f"{delta:+.4f}",
             "delta": "detecção mais fraca" if delta < 0 else ("estável" if delta == 0 else "detecção mais forte"),
             "delta_dir": "down" if delta < 0 else ("flat" if delta == 0 else "up"),
             "accent": "#ff4655" if delta < 0 else "#35c07a"},
            {"label": "Menor F1", "value": f"{lowest:.4f}",
             "hint": f"iteração {lowest_it}", "accent": "#f5a524"},
        ]
    else:
        cards.append({"label": "F1", "value": "indisponível", "accent": "#f5a524"})
    ui.kpi_row(cards)


def _evolution(records: list[IterationRecord]) -> None:
    ui.section("Desempenho", "Evolução das métricas de detecção",
               "F1, precisão e recall da classe de ataque a cada iteração. Quanto menor, mais o ataque evadiu.")
    chart = charts.metrics_evolution(records)
    if chart is None:
        st.info("Nenhuma métrica numérica disponível neste histórico (execução pulou avaliações).")
        return
    st.altair_chart(chart, width="stretch")
    ratio = charts.attack_ratio_bars(records)
    if ratio is not None:
        with st.expander("Razão de contagem de ataque vs. baseline (integridade da variante)"):
            st.altair_chart(ratio, width="stretch")
            st.caption("Abaixo de 50% o ataque é considerado degenerado (raro/fraco demais).")


def _confusion(records: list[IterationRecord]) -> None:
    ui.section("Erros do detector", "Matriz de confusão")
    by_it = {r.iteration: r for r in records}
    selected = st.selectbox("Iteração", options=sorted(by_it), key="cm_it")
    m = by_it[selected].metrics
    if any(v is None for v in (m.tn, m.fp, m.fn, m.tp)):
        st.info("A matriz de confusão binária não está disponível nesta iteração.")
        return
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        ui.confusion_matrix(int(m.tn), int(m.fp), int(m.fn), int(m.tp))
    with c2:
        total = (m.tn or 0) + (m.fp or 0) + (m.fn or 0) + (m.tp or 0)
        st.markdown(
            f"**Leitura.** De **{total:,}** amostras avaliadas, o IDS classificou "
            f"corretamente **{(m.tn or 0) + (m.tp or 0):,}**. "
            f"Falsos negativos (ataque não detectado): **{m.fn or 0:,}**; "
            f"falsos positivos: **{m.fp or 0:,}**.".replace(",", "."),
        )
        if m.attack_label_original:
            st.markdown(ui.badge_html(f"classe de ataque: {m.attack_label_original}", "red"),
                        unsafe_allow_html=True)


def _features(records: list[IterationRecord]) -> None:
    ui.section("Explicabilidade", "Features mais determinantes",
               "Importâncias do detector treinado — quais campos do tráfego pesaram na decisão. "
               "A escala varia por detector: Gini para as árvores, |coef_| para o SVM linear.")
    by_it = {r.iteration: r for r in records}
    selected = st.selectbox("Iteração", options=sorted(by_it), key="feat_it")
    feats = [f.model_dump() if hasattr(f, "model_dump") else f
             for f in by_it[selected].metrics.top_feature_importances]
    ui.feature_bars(feats)


def _config_diff(records: list[IterationRecord]) -> None:
    ui.section("Rastreabilidade", "O que o Estrategista mudou",
               "Diferença de configuração entre duas iterações.")
    by_it = {r.iteration: r for r in records}
    its = sorted(by_it)
    if len(its) < 2:
        st.info("São necessárias ao menos duas iterações para comparar configurações.")
        return
    c1, c2 = st.columns(2)
    prev = c1.selectbox("Iteração anterior", its, index=0, key="diff_prev")
    curr = c2.selectbox("Iteração posterior", its, index=len(its) - 1, key="diff_curr")
    diffs = _flatten_diff(by_it[prev].attack_config, by_it[curr].attack_config)
    if diffs.empty:
        st.success("As configurações selecionadas são idênticas.")
    else:
        st.dataframe(diffs, hide_index=True, width="stretch")

    so = by_it[curr].strategist_output
    if so is not None:
        with st.expander(f"Raciocínio do Estrategista · iteração {curr}"):
            st.markdown(ui.badge_html(f"persona: {so.persona}", "red"), unsafe_allow_html=True)
            st.write(so.reasoning)
            if so.changes:
                st.dataframe(pd.DataFrame([c.model_dump() for c in so.changes]),
                             hide_index=True, width="stretch")


def _analyst(records: list[IterationRecord]) -> None:
    ui.section("Blue Team", "Diagnóstico do Analista",
               "Interpretação das métricas + mitigações recomendadas, validadas contra os números.")
    by_it = {r.iteration: r for r in records}
    with_analyst = [it for it in sorted(by_it) if by_it[it].analyst_output is not None]
    if not with_analyst:
        st.info("Nenhuma iteração possui análise do Blue Team neste histórico.")
        return
    selected = st.selectbox("Iteração analisada", with_analyst, key="analyst_it")
    out = by_it[selected].analyst_output
    data = out.model_dump(mode="json")

    sev = str(data.get("severity", "")).lower()
    sev_kind = {"low": "good", "medium": "warn", "high": "bad"}.get(sev, "")
    st.markdown(ui.badge_html(f"severidade: {sev or '—'}", sev_kind), unsafe_allow_html=True)

    if data.get("diagnosis"):
        st.markdown(f"**Diagnóstico.** {data['diagnosis']}")

    feats = data.get("deceptive_features") or []
    if feats:
        st.markdown("**Features que enganaram o detector**")
        st.dataframe(pd.DataFrame(feats), hide_index=True, width="stretch")

    mits = data.get("mitigations") or []
    if mits:
        st.markdown("**Mitigações recomendadas**")
        for mit in mits:
            st.markdown(
                f'{ui.badge_html(mit.get("type", "—"), "blue")} '
                f'{mit.get("recommendation", "")}',
                unsafe_allow_html=True,
            )


def _table(records: list[IterationRecord]) -> None:
    ui.section("Detalhe", "Tabela por iteração")
    rows = []
    prev_f1 = None
    for r in sorted(records, key=lambda x: x.iteration):
        m = r.metrics
        f1 = m.f1_score_attack
        var = None if f1 is None or prev_f1 is None else f1 - prev_f1
        rows.append({
            "Iteração": r.iteration,
            "F1": f1, "Precisão": m.precision_attack, "Recall": m.recall_attack,
            "Acurácia": m.accuracy, "Δ F1": var,
            "Config alterada": m.config_changed,
            "Degenerada": m.degenerate_variant,
            "Análise": r.analyst_output is not None,
        })
        if f1 is not None:
            prev_f1 = f1
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _flatten_diff(prev: dict, curr: dict) -> pd.DataFrame:
    def flat(v: Any, p: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(v, dict):
            for k, i in v.items():
                out.update(flat(i, f"{p}.{k}" if p else k))
        elif isinstance(v, list):
            for idx, i in enumerate(v):
                out.update(flat(i, f"{p}[{idx}]"))
        else:
            out[p] = v
        return out
    a, b = flat(prev), flat(curr)
    keys = sorted(a.keys() | b.keys())
    rows = [{"Campo": k, "Antes": a.get(k), "Depois": b.get(k)}
            for k in keys if a.get(k) != b.get(k)]
    return pd.DataFrame(rows, columns=["Campo", "Antes", "Depois"])


# --------------------------------------------------------------------------- #
# Entrada da página                                                           #
# --------------------------------------------------------------------------- #
def render() -> None:
    ui.hero(
        kicker="Passo 3 de 3 · Panorama",
        title_html='Dashboard de <span class="accent">resultados</span>',
        subtitle="Desempenho quantitativo e explicabilidade do loop — o que mudou, "
                 "o quanto o IDS foi enganado e por quê.",
    )

    # auto-carrega algo útil na primeira visita: prefere a última execução real
    # (outputs/), mas cai no golden se ela não tiver F1 aproveitável.
    if get_records() is None:
        try:
            loaded = None
            if ITERATION_HISTORY_PATH.exists():
                recs = load_history_file(ITERATION_HISTORY_PATH, "Última execução (outputs/)")
                if any(r.metrics.f1_score_attack is not None for r in recs):
                    loaded = recs
            if loaded is None and GOLDEN_HISTORY_PATH.exists():
                load_history_file(GOLDEN_HISTORY_PATH, "Demo golden (data/)")
        except Exception:
            pass

    _source_picker()

    records = get_records()
    if not records:
        st.info("Nenhum resultado carregado. Rode um experimento em **Execução** ou "
                "carregue um histórico acima.")
        return

    st.markdown(
        f'<div style="margin:6px 0 2px">'
        f'{ui.badge_html("fonte: " + (st.session_state.get("records_source") or "—"), "blue")}'
        f'</div>', unsafe_allow_html=True)

    _summary(records)
    _evolution(records)

    tab1, tab2, tab3 = st.tabs(["Erros & Features", "Rastreabilidade", "Blue Team"])
    with tab1:
        _confusion(records)
        _features(records)
    with tab2:
        _config_diff(records)
    with tab3:
        _analyst(records)

    _table(records)
