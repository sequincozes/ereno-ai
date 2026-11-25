"""Modo de features de protocolo GOOSE (``PROTOCOL_FEATURES_MODE``).

A lista de sempre-descartar herdada do commit inicial do framework tratava
identidade e delta como a mesma coisa. Medidos contra a classe no dataset de
referência, os dois grupos não se parecem: identidade dá MI ~0, delta dá 30% a
81% do teto. Estes testes fixam a separação, o default conservador, e o fato de
que virar a chave é uma decisão do experimento — não um efeito colateral.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from adversarial_ids.config.settings import DATA_DIR
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.core.preprocessor import (
    _ALWAYS_DROP,
    _IDENTITY_DROP,
    _PROTOCOL_DELTAS,
    FeaturePreprocessor,
    PreprocessorError,
    always_drop_for,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isbATrapAreaSum": [1.0, 2.0, 3.0, 4.0],
            "cbStatus": [0, 1, 0, 1],
            # Deltas: variam, então só somem por always_drop, nunca por constante.
            "stDiff": [0, 1, 0, 2],
            "sqDiff": [1, 1, 5, 1],
            "SqNum": [1, 2, 7, 3],
            "tDiff": [0.1, 0.1, 0.9, 0.1],
            "timestampDiff": [0.2, 0.2, 0.8, 0.2],
            "timeFromLastChange": [1.0, 2.0, 0.1, 3.0],
            "gooseLengthDiff": [0, 0, 4, 0],
            "apduSizeDiff": [0, 0, 3, 0],
            "frameLengthDiff": [0, 0, 2, 0],
            # Identidade: variam também, e mesmo assim somem nos dois modos.
            "GooseTimestamp": [10.0, 11.0, 12.0, 13.0],
            "StNum": [1, 2, 3, 4],
            "ethSrc": ["a", "b", "c", "d"],
            "class": ["normal", "attack", "attack", "normal"],
        }
    )


# --------------------------------------------------------------------------- #
# A separação em si                                                           #
# --------------------------------------------------------------------------- #
def test_the_two_groups_partition_the_original_list():
    """Nada se perdeu nem se duplicou ao separar a lista herdada."""

    assert set(_IDENTITY_DROP) & set(_PROTOCOL_DELTAS) == set()
    assert set(_IDENTITY_DROP) | set(_PROTOCOL_DELTAS) == set(_ALWAYS_DROP)
    assert len(_ALWAYS_DROP) == len(_IDENTITY_DROP) + len(_PROTOCOL_DELTAS)


def test_raw_counters_and_clocks_stay_in_the_identity_group():
    """MI ~0 contra a classe, e num gerador sintético vazariam a geração."""

    assert {
        "Time",
        "t",
        "GooseTimestamp",
        "receivedTimestamp",
        "StNum",
        "delay",
    } <= set(_IDENTITY_DROP)


def test_publisher_identity_stays_in_the_identity_group():
    """Manter estas ensinaria o detector a reconhecer o publisher, não o ataque."""

    assert {"ethSrc", "ethDst", "gocbRef", "goID", "datSet", "gooseAppid"} <= set(
        _IDENTITY_DROP
    )


def test_sequence_and_timing_deltas_are_the_recoverable_group():
    assert {"stDiff", "sqDiff", "tDiff", "timestampDiff", "timeFromLastChange"} <= set(
        _PROTOCOL_DELTAS
    )


def test_sqnum_is_a_delta_and_stnum_is_an_identifier():
    """Assimetria deliberada, e o caso menos claro do grupo.

    Em GOOSE o sqNum zera quando o stNum incrementa, então ele carrega estado,
    não índice — e a medição concorda (SqNum 79% do teto de MI, StNum 0.03%).
    Parte disso ainda pode ser artefato de como o ERENO emite as rajadas, que é
    justamente por que o modo é medível por ablação em vez de estar ligado.
    """

    assert "SqNum" in _PROTOCOL_DELTAS
    assert "StNum" in _IDENTITY_DROP


# --------------------------------------------------------------------------- #
# always_drop_for                                                             #
# --------------------------------------------------------------------------- #
def test_drop_mode_reproduces_the_inherited_list():
    assert always_drop_for("drop") == _ALWAYS_DROP


def test_deltas_mode_drops_only_the_identity_group():
    assert always_drop_for("deltas") == _IDENTITY_DROP


def test_unknown_mode_raises():
    with pytest.raises(PreprocessorError, match="protocol_features inválido"):
        always_drop_for("keep_everything")


# --------------------------------------------------------------------------- #
# FeaturePreprocessor                                                         #
# --------------------------------------------------------------------------- #
def test_default_still_drops_the_protocol_deltas():
    """O default é o comportamento herdado: nada muda sem alguém pedir."""

    pre = FeaturePreprocessor().fit(_frame().drop(columns="class"), label_column="class")

    assert set(pre.feature_columns) == {"isbATrapAreaSum", "cbStatus"}


def test_deltas_mode_lets_the_sequence_features_through():
    pre = FeaturePreprocessor(protocol_features="deltas").fit(
        _frame().drop(columns="class"), label_column="class"
    )

    assert {"stDiff", "sqDiff", "SqNum", "tDiff"} <= set(pre.feature_columns)


def test_deltas_mode_still_drops_identity():
    """Recuperar semântica não é recuperar identificador."""

    pre = FeaturePreprocessor(protocol_features="deltas").fit(
        _frame().drop(columns="class"), label_column="class"
    )

    assert {"GooseTimestamp", "StNum", "ethSrc"} & set(pre.feature_columns) == set()


def test_the_manifest_records_which_columns_were_dropped_in_each_mode():
    """O artefato diz sozinho sob que regime a execução rodou."""

    frame = _frame().drop(columns="class")

    strict = FeaturePreprocessor().fit(frame, label_column="class").manifest()
    loose = (
        FeaturePreprocessor(protocol_features="deltas")
        .fit(frame, label_column="class")
        .manifest()
    )

    dropped = {c.name for c in strict.dropped_columns} - {
        c.name for c in loose.dropped_columns
    }

    assert dropped == set(_PROTOCOL_DELTAS) & set(frame.columns)


def test_explicit_always_drop_still_wins():
    pre = FeaturePreprocessor(always_drop=("cbStatus",)).fit(
        _frame().drop(columns="class"), label_column="class"
    )

    assert "cbStatus" not in pre.feature_columns
    assert "GooseTimestamp" in pre.feature_columns


def test_always_drop_and_protocol_features_together_are_rejected():
    """Uma lista literal e um modo que também produz lista: ambíguo, não fusível."""

    with pytest.raises(PreprocessorError, match="informados juntos"):
        FeaturePreprocessor(always_drop=("cbStatus",), protocol_features="deltas")


def test_preprocessor_rejects_an_invalid_mode_at_construction():
    with pytest.raises(PreprocessorError, match="protocol_features inválido"):
        FeaturePreprocessor(protocol_features="deltas_maybe")


# --------------------------------------------------------------------------- #
# IdsEvaluator                                                                #
# --------------------------------------------------------------------------- #
def test_evaluator_defaults_to_the_inherited_behavior():
    assert IdsEvaluator().protocol_features == "drop"


def test_evaluator_rejects_an_invalid_mode_when_constructed():
    """Falha ao criar o avaliador, não no meio de um treino."""

    with pytest.raises(PreprocessorError, match="protocol_features inválido"):
        IdsEvaluator(protocol_features="todas")


def test_evaluator_passes_the_mode_down_to_the_preprocessor(tmp_path):
    frame = pd.concat([_frame()] * 10, ignore_index=True)
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)

    evaluator = IdsEvaluator(protocol_features="deltas", drop_cb_status=False)
    metrics = evaluator.train_baseline(str(path))

    assert {"sqDiff", "tDiff"} <= set(metrics["used_features"])


# --------------------------------------------------------------------------- #
# O espaço de features real do ERENO                                          #
# --------------------------------------------------------------------------- #
def _header() -> list[str]:
    with (DATA_DIR / "baseline_dataset.csv").open(encoding="utf-8") as handle:
        return next(csv.reader(handle))


def test_the_mode_is_what_separates_a_protocol_aware_run_from_an_analog_only_one():
    """Fixa o tamanho dos dois espaços sobre o dataset versionado.

    Em ``drop`` sobram 20 colunas e 18 são grandezas elétricas — é por isso que
    só a falta forjada é detectável. Em ``deltas`` sobram 29, e as 9 novas são
    exatamente a semântica de que replay, flooding e grayhole dependem.
    """

    header = set(_header()) - {"class"}

    strict = header - set(always_drop_for("drop"))
    loose = header - set(always_drop_for("deltas"))

    assert len(strict) == 20
    assert len(loose) == 29
    assert loose - strict == set(_PROTOCOL_DELTAS)
