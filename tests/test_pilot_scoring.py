"""Testes da estatística do piloto E0 (`scripts/run_pilot.py`).

O que está sob teste é o juízo do piloto — quando uma coluna conta como movida —,
não a geração: essa precisa do JAR e é justamente o que o piloto existe para
exercitar à mão. As três propriedades que importam:

1. separação zero quando os dois recortes têm a mesma distribuição;
2. coluna constante em valores diferentes é separação total, e não infinito;
3. uma coluna só é "movida" se passar do piso absoluto **e** do ruído dela
   mesma — foi comparar contra um piso global que fazia o veredito da família
   oscilar entre execuções idênticas (ver `docs/pilot_e0.md`).
"""

from __future__ import annotations

import argparse

import pandas as pd

from scripts.run_pilot import _count_above, _moved_features, _separation


def _args(**overrides: float) -> argparse.Namespace:
    defaults = {"min_drift": 0.05, "drift_margin": 2.0}
    defaults.update(overrides)

    return argparse.Namespace(**defaults)


def test_distribuicoes_iguais_nao_separam():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})

    result = _separation(frame, frame.copy())

    assert result["max"] == 0.0
    assert result["constant_shift"] == []


def test_medias_distantes_separam_proporcionalmente_ao_desvio():
    left = pd.DataFrame({"a": [10.0, 11.0, 12.0]})
    right = pd.DataFrame({"a": [0.0, 1.0, 2.0]})

    result = _separation(left, right)

    # Mesma variância dos dois lados (1.0): a separação é a diferença de médias.
    assert result["max"] == 10.0
    assert result["top"][0]["feature"] == "a"


def test_coluna_constante_em_valores_diferentes_vira_constant_shift():
    left = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
    right = pd.DataFrame({"a": [7.0, 7.0, 7.0]})

    result = _separation(left, right)

    # Desvio zero dos dois lados: dividir daria infinito, que não cabe em JSON
    # nem em comparação com limiar. A evidência entra por presença.
    assert result["constant_shift"] == ["a"]
    assert result["max"] == 0.0


def test_coluna_ausente_de_um_lado_e_ignorada():
    left = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    right = pd.DataFrame({"a": [5.0, 6.0]})

    result = _separation(left, right)

    assert "b" not in result["by_feature"]


def test_deriva_abaixo_do_piso_absoluto_nao_conta():
    drift = {"by_feature": {"a": 0.04}, "constant_shift": []}

    moved = _moved_features(drift, {"a": 0.0}, set(), _args())

    # Ruído zero naquela coluna não basta: sem piso absoluto qualquer
    # deslocamento passaria.
    assert moved == []


def test_deriva_dentro_do_ruido_da_propria_coluna_nao_conta():
    drift = {"by_feature": {"a": 0.30}, "constant_shift": []}

    moved = _moved_features(drift, {"a": 0.20}, set(), _args())

    assert moved == []


def test_deriva_acima_dos_dois_criterios_conta():
    drift = {"by_feature": {"a": 0.30}, "constant_shift": []}

    moved = _moved_features(drift, {"a": 0.10}, set(), _args())

    assert [item["feature"] for item in moved] == ["a"]
    assert moved[0]["noise"] == 0.10


def test_o_ruido_e_por_coluna_e_nao_o_maior_da_tabela():
    drift = {"by_feature": {"estavel": 0.30, "ruidosa": 0.30}, "constant_shift": []}
    noise = {"estavel": 0.01, "ruidosa": 0.90}

    moved = _moved_features(drift, noise, set(), _args())

    # Com um piso global (0.90) nenhuma das duas passaria, e a coluna estável
    # perderia o veredito por azar de outra coluna.
    assert [item["feature"] for item in moved] == ["estavel"]


def test_coluna_que_o_gerador_desloca_sozinho_nao_serve_de_evidencia():
    drift = {"by_feature": {"a": 0.9}, "constant_shift": ["b"]}

    moved = _moved_features(drift, {"a": 0.0}, {"a", "b"}, _args())

    assert moved == []


def test_constant_shift_na_variante_conta_como_movida():
    drift = {"by_feature": {}, "constant_shift": ["b"]}

    moved = _moved_features(drift, {}, set(), _args())

    assert moved == [{"feature": "b", "drift": None, "noise": None, "kind": "constant_shift"}]


def test_contagem_acima_do_limiar_soma_as_constantes():
    separation = {"by_feature": {"a": 0.5, "b": 0.1}, "constant_shift": ["c"]}

    assert _count_above(separation, 0.2) == 2
