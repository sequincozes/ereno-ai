"""Página · Visão Geral — o que é o laboratório adversarial."""

from __future__ import annotations

import streamlit as st

from adversarial_ids.config.attacks_registry import _SPECS  # ordem estável p/ exibição

from . import components as ui


def render() -> None:
    ui.hero(
        kicker="Laboratório adversarial de IDS · Smart Grids IEC-61850 / GOOSE",
        title_html='ERENO <span class="accent">AI</span> Studio',
        subtitle=(
            "Um laboratório Red Team × Blue Team que gera ataques sintéticos "
            "parametrizáveis e mede — a cada iteração — o quanto eles enganam um "
            "sistema de detecção de intrusão, explicando o porquê."
        ),
        show_teams=True,
    )

    # --- o problema -----------------------------------------------------
    ui.section("Contexto", "O problema que resolvemos",
               "Por que avaliar um IDS contra um conjunto estático de ataques esconde fragilidades.")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown(
            "Sistemas de detecção de intrusão para subestações de energia costumam "
            "ser avaliados contra um conjunto **estático** de ataques. Isso esconde "
            "uma fragilidade: pequenas variações nos parâmetros do ataque "
            "(temporização, campos manipulados, intensidade) podem **derrubar a taxa "
            "de detecção** sem que a equipe de defesa perceba **por quê**."
        )
    with col2:
        st.markdown(
            "Este sistema fecha o ciclo: transforma a avaliação do IDS em um **loop "
            "adversarial iterativo e explicável**, produzindo tanto evidência "
            "quantitativa (F1, matriz de confusão, diffs de configuração) quanto "
            "diagnóstico qualitativo (features enganosas + mitigações sugeridas)."
        )
    ui.badges([
        ("100% tráfego sintético (ERENO)", "good"),
        ("Sem dados sensíveis reais", "good"),
        ("Padrão IEC-61850 / GOOSE", "blue"),
        ("Agentes Agno + Groq", "blue"),
    ])

    # --- o fluxo --------------------------------------------------------
    ui.section("Como funciona", "O loop, iteração a iteração",
               "A cada rodada, o núcleo encadeia Estrategista → geração → avaliação do IDS → Analista.")
    ui.steps([
        {"num": "0", "title": "Baseline",
         "body": "O núcleo gera/serve o dataset de ataque inicial, treina o IDS "
                 "(Random Forest) e mede as métricas de referência."},
        {"num": "1", "title": "Estrategista propõe (Red Team)", "team": "red",
         "body": "Recebe a configuração atual, as métricas e o histórico e propõe, via "
                 "tool calling estruturado, uma nova configuração de ataque — "
                 "conservador altera 1 campo; agressivo, até 3."},
        {"num": "2", "title": "Núcleo materializa a variante",
         "body": "Aplica o patch, valida/clampa a configuração, regenera o dataset "
                 "(modo cached sem Java, ou jar com o gerador ERENO real) e re-avalia o IDS."},
        {"num": "3", "title": "Analista explica (Blue Team)",
         "body": "Interpreta as novas métricas, identifica as features que enganaram o "
                 "detector e recomenda mitigações — validado contra as próprias métricas."},
        {"num": "4", "title": "Memória + Apresentação",
         "body": "Cada passo vira um IterationRecord tipado e persistido. Este Studio "
                 "mostra desempenho e explicabilidade sem passar pelo terminal."},
    ])

    # --- catálogo de ataques -------------------------------------------
    ui.section("Arsenal", "Tipos de ataque suportados",
               "Onze variantes do gerador ERENO que os agentes podem otimizar (modo jar).")
    specs = list(_SPECS)
    for start in range(0, len(specs), 3):
        cols = st.columns(3, gap="medium")
        for col, spec in zip(cols, specs[start:start + 3]):
            with col:
                code = spec.segment_name.split("_")[0].upper()
                st.markdown(
                    ui.attack_card(f"{code} · {spec.attack_type}", spec.key, spec.description),
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.info(
        "Pronto para rodar? Vá para **Configuração** no menu à esquerda para montar "
        "os parâmetros pela interface — nada de linha de comando."
    )
