"""Página · Configuração — monta os parâmetros da execução pela interface."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from adversarial_ids.config.attacks_registry import (
    DEFAULT_ATTACK_KEY,
    get_attack_spec,
    list_attack_keys,
)
from adversarial_ids.config.settings import GENERATOR_JAR_PATH, MODEL_IDS
from adversarial_ids.shared.editable_fields import derive_editable_fields
from adversarial_ids.shared.json_io import load_json

from . import components as ui
from .bridge import ExperimentConfig
from .state import get_config


def _env_status() -> list[tuple[str, str]]:
    groq = bool(os.getenv("GROQ_API_KEY", "").strip())
    jar = GENERATOR_JAR_PATH.exists()
    return [
        ("GROQ_API_KEY " + ("detectada" if groq else "ausente"), "good" if groq else "warn"),
        ("JAR ERENO " + ("presente" if jar else "ausente"), "good" if jar else "warn"),
    ]


def render() -> None:
    cfg: ExperimentConfig = get_config()

    ui.hero(
        kicker="Passo 1 de 3 · Parametrização",
        title_html='Configurar <span class="accent">execução</span>',
        subtitle="Defina o motor, o ataque e os agentes. Tudo pela interface — os "
                 "valores viram a linha de comando internamente.",
    )

    ui.section("Ambiente", "Status do host", "Requisitos detectados nesta máquina.")
    ui.badges(_env_status())

    # ---------------- motor -------------------------------------------
    ui.section("Motor", "Como o loop será executado")
    engine = st.radio(
        "Motor de execução",
        options=["demo", "live"],
        format_func=lambda e: {
            "demo": "🟢 Demonstração — replay do histórico golden (sem Groq/Java)",
            "live": "🔴 Ao vivo — loop real com agentes (exige GROQ_API_KEY)",
        }[e],
        index=0 if cfg.engine == "demo" else 1,
        horizontal=False,
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        iterations = st.slider(
            "Iterações adversariais (após o baseline)",
            min_value=1, max_value=30, value=cfg.iterations,
        )
    with col_b:
        generator_mode = st.selectbox(
            "Modo do gerador",
            options=["cached", "jar"],
            index=0 if cfg.generator_mode == "cached" else 1,
            help="cached: serve o baseline versionado (sem Java). "
                 "jar: regenera o dataset com o ERENO real por configuração.",
            disabled=engine == "demo",
        )

    live_disabled = engine == "demo"

    # ---------------- ataque ------------------------------------------
    ui.section("Alvo", "Tipo de ataque a otimizar",
               "No modo demo o ataque é fixo (uc03 golden); no live os agentes otimizam o escolhido.")
    attack_keys = list_attack_keys()
    attack = st.selectbox(
        "Ataque ERENO",
        options=attack_keys,
        index=attack_keys.index(cfg.attack) if cfg.attack in attack_keys else 0,
        disabled=live_disabled,
    )
    if live_disabled:
        attack = DEFAULT_ATTACK_KEY

    spec = get_attack_spec(attack)
    st.markdown(
        ui.attack_card(
            f"{spec.segment_name.split('_')[0].upper()} · {spec.attack_type}",
            spec.key, spec.description,
        ),
        unsafe_allow_html=True,
    )

    # prévia dos campos editáveis derivados do baseline do ataque
    with st.expander("Ver parâmetros editáveis derivados deste ataque"):
        try:
            baseline = load_json(spec.baseline_path)
            fields = derive_editable_fields(baseline, max_fields=40)
            df = pd.DataFrame(
                [{"Campo": f["field"], "Valor atual": str(f["current_value"]),
                  "Tipo": f["type"]} for f in fields]
            )
            st.caption(f"{len(fields)} campo(s) — derivados automaticamente das folhas do JSON.")
            st.dataframe(df, hide_index=True, width="stretch")
        except Exception as exc:  # fronteira de leitura
            st.warning(f"Não foi possível ler o baseline do ataque: {exc}")

    # ---------------- agentes -----------------------------------------
    ui.section("Agentes", "Estrategista e orquestração", "Válido apenas no motor ao vivo.")
    col_c, col_d, col_e = st.columns(3, gap="large")
    with col_c:
        persona = st.selectbox(
            "Persona do Estrategista",
            options=["conservative", "aggressive"],
            index=0 if cfg.persona == "conservative" else 1,
            help="conservative: 1 alteração/iteração · aggressive: até 3.",
            disabled=live_disabled,
        )
    with col_d:
        orchestration = st.selectbox(
            "Orquestração",
            options=["team", "direct"],
            index=0 if cfg.orchestration == "team" else 1,
            help="team: agno.team.Team (route mode) · direct: encadeamento determinístico.",
            disabled=live_disabled,
        )
    with col_e:
        model_id = st.selectbox(
            "Modelo (Groq)",
            options=MODEL_IDS,
            # Cai no primeiro da lista quando a config guardada aponta para um
            # modelo que saiu de MODEL_IDS — indexar por um id fixo levantaria
            # ValueError justamente quando a Groq retira aquele modelo.
            index=MODEL_IDS.index(cfg.model_id) if cfg.model_id in MODEL_IDS else 0,
            disabled=live_disabled,
        )

    # ---------------- persistir + resumo ------------------------------
    new_cfg = ExperimentConfig(
        engine=engine,
        iterations=int(iterations),
        attack=attack,
        persona=persona,
        orchestration=orchestration,
        generator_mode=generator_mode if engine == "live" else "cached",
        model_id=model_id,
    )
    st.session_state["config"] = new_cfg

    ui.section("Resumo", "Comando equivalente", "É exatamente isto que a execução dispara.")
    summary_cols = st.columns(len(new_cfg.as_summary()))
    for col, (k, v) in zip(summary_cols, new_cfg.as_summary().items()):
        col.markdown(
            f'<div class="eren-mono-label">{k}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;color:#e6edf3;font-size:0.9rem">{v}</div>',
            unsafe_allow_html=True,
        )

    cli = _equivalent_cli(new_cfg)
    st.code(cli, language="bash")

    if engine == "live" and not os.getenv("GROQ_API_KEY", "").strip():
        st.warning("Motor **ao vivo** selecionado, mas `GROQ_API_KEY` não foi detectada. "
                   "Defina a chave no `.env` antes de executar, ou use o motor de demonstração.")

    st.success("Configuração salva. Abra **Execução** no menu para rodar com logs ao vivo.")


def _equivalent_cli(cfg: ExperimentConfig) -> str:
    parts = ["uv run adversarial-ids", f"--engine {cfg.engine}",
             f"--iterations {cfg.iterations}"]
    if cfg.engine == "live":
        parts += [f"--generator-mode {cfg.generator_mode}", f"--attack {cfg.attack}",
                  f"--persona {cfg.persona}", f"--orchestration {cfg.orchestration}",
                  f"--model-id {cfg.model_id}"]
    return " \\\n    ".join(parts)
