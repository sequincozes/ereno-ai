"""Catálogo feature→técnica e playbooks IEC-61850 (épico E5, janela D47-54).

O teste central deste módulo é o de cobertura: o catálogo precisa mapear
exatamente as features que **podem sobreviver ao preprocessador sob algum
modo**. Uma entrada a menos deixa evidência real sem técnica; uma a mais é
catálogo morto, porque aquela chave não tem como chegar a um
``DetectionReport``. Como ``PROTOCOL_FEATURES_MODE`` decide se os deltas de
protocolo passam, a referência é a união dos modos.
"""

from __future__ import annotations

import csv

import pytest

from adversarial_ids.config.attacks_registry import list_attack_keys
from adversarial_ids.config.defense_techniques import (
    FEATURE_FAMILIES,
    FEATURE_TECHNIQUES,
    METRIC_TECHNIQUES,
    PLAYBOOKS,
    catalogued_techniques,
    playbook_for_attack,
    playbooks_for_technique,
    techniques_for_feature,
    techniques_for_metric,
)
from adversarial_ids.config.settings import DATA_DIR
from adversarial_ids.core.preprocessor import _PROTOCOL_DELTAS, always_drop_for
from adversarial_ids.domain import DEFENSE_TECHNIQUES, VALIDATION_METRICS
from adversarial_ids.domain.defense_plan import _BUCKETS_BY_TECHNIQUE


def _usable_features(mode: str = "deltas") -> set[str]:
    """As colunas do ERENO que podem aparecer em ``top_features`` sob ``mode``.

    Sai do dataset versionado, não de uma lista escrita à mão: se o schema do
    ERENO mudar, é aqui que a divergência aparece. O default é ``deltas``
    porque o catálogo cobre a união dos modos — ``drop`` é um subconjunto.
    """

    with (DATA_DIR / "baseline_dataset.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    return set(header) - set(always_drop_for(mode)) - {"class"}


# --------------------------------------------------------------------------- #
# Cobertura do espaço de features                                             #
# --------------------------------------------------------------------------- #
def test_catalog_covers_exactly_the_features_that_survive_preprocessing():
    """Nem uma feature utilizável sem técnica, nem uma chave inalcançável.

    A união dos modos, porque o catálogo não deve depender de qual
    PROTOCOL_FEATURES_MODE o experimento escolheu.
    """

    assert set(FEATURE_TECHNIQUES) == _usable_features("deltas")


def test_the_drop_mode_space_is_a_strict_subset_of_the_catalog():
    """Trocar de modo nunca deixa evidência real sem técnica catalogada."""

    assert _usable_features("drop") < set(FEATURE_TECHNIQUES)


def test_the_default_mode_space_is_almost_entirely_analog():
    """Fixa por que só a falta forjada é detectável no modo default.

    Em ``drop`` sobram 20 colunas, 18 delas grandezas elétricas: não há uma
    única feature de sequência ou temporização para ancorar replay, flooding
    ou grayhole.
    """

    usable = _usable_features("drop")

    assert len(usable) == 20
    assert {"StNum", "SqNum", "stDiff", "sqDiff"} & usable == set()
    assert {"cbStatus", "cbStatusDiff"} <= usable


def test_the_deltas_mode_adds_exactly_the_protocol_semantics():
    usable = _usable_features("deltas")

    assert len(usable) == 29
    assert usable - _usable_features("drop") == set(_PROTOCOL_DELTAS)


def test_publisher_identity_is_never_usable_in_either_mode():
    """Recuperar semântica não é recuperar identificador."""

    for mode in ("drop", "deltas"):
        assert {"ethSrc", "gocbRef", "goID", "datSet", "StNum"} & _usable_features(
            mode
        ) == set()


def test_feature_families_partition_the_catalog():
    """Toda feature pertence a exatamente uma família."""

    flat = [feature for features in FEATURE_FAMILIES.values() for feature in features]

    assert len(flat) == len(set(flat))
    assert set(flat) == set(FEATURE_TECHNIQUES)


@pytest.mark.parametrize("feature", sorted(_usable_features()))
def test_every_usable_feature_yields_at_least_one_technique(feature: str):
    assert techniques_for_feature(feature)


def test_unknown_feature_yields_nothing_instead_of_raising():
    """Um dataset novo pode trazer coluna que o catálogo não conhece.

    Isso é motivo para o plano não ter técnica ancorada nela, não para o
    pipeline quebrar no meio de uma execução.
    """

    assert techniques_for_feature("featureQueNaoExiste") == ()


def test_analog_features_recommend_the_physical_check_first():
    """A resposta a uma forma de onda fabricada é confrontá-la com a física.

    Se a primeira recomendação fosse ``detector_retraining``, o catálogo
    estaria devolvendo a resposta genérica de ML que este épico existe para
    eliminar.
    """

    for feature in FEATURE_FAMILIES["analog_trap_area"]:
        assert techniques_for_feature(feature)[0] == "physical_consistency_check"


def test_breaker_state_recommends_sequence_validation_first():
    """No modo default, ``cbStatus`` é o único ponto onde a semântica de
    sequência do GOOSE ainda é observável — os deltas saem todos."""

    assert techniques_for_feature("cbStatus")[0] == "goose_sequence_validation"


# --------------------------------------------------------------------------- #
# Lado ancorado em métrica                                                    #
# --------------------------------------------------------------------------- #
def test_every_validation_metric_has_a_technique():
    """Sem isto, um plano de ``svm_rbf`` ficaria sem técnica nenhuma.

    Aquele detector não expõe importâncias, então ``top_features`` vem vazio e
    o lado ancorado em feature não oferece nada (épico E8).
    """

    assert set(METRIC_TECHNIQUES) == set(VALIDATION_METRICS)


@pytest.mark.parametrize("metric", sorted(VALIDATION_METRICS))
def test_metric_techniques_are_non_empty_and_unique(metric: str):
    techniques = techniques_for_metric(metric)

    assert techniques
    assert len(set(techniques)) == len(techniques)


def test_recall_and_false_negatives_lead_with_the_same_priority():
    """Falso negativo é o risco central: o ataque passou."""

    assert techniques_for_metric("recall")[0] == "detector_threshold_tuning"
    assert techniques_for_metric("confusion_matrix.fn")[0] == "detector_threshold_tuning"


def test_techniques_for_metric_rejects_an_unknown_metric():
    with pytest.raises(KeyError, match="Métrica sem técnica catalogada"):
        techniques_for_metric("roc_auc")


# --------------------------------------------------------------------------- #
# Playbooks                                                                   #
# --------------------------------------------------------------------------- #
def test_every_registered_attack_has_a_playbook():
    """Um ataque sem resposta catalogada é uma lacuna, não uma omissão."""

    for attack_key in list_attack_keys():
        assert playbook_for_attack(attack_key)


def test_playbook_for_an_unregistered_attack_raises():
    with pytest.raises(KeyError, match="Ataque sem playbook"):
        playbook_for_attack("ataque_inexistente")


def test_the_six_replay_variants_share_one_playbook():
    """Backoff, batch dump e double drop são temporizações do mesmo cenário."""

    keys = {
        playbook_for_attack(attack).key
        for attack in (
            "random_replay",
            "inverse_replay",
            "delayed_replay",
            "delayed_replay_backoff",
            "delayed_replay_batch_dump",
            "delayed_replay_double_drop",
        )
    }

    assert keys == {"replayed_goose_frames"}


@pytest.mark.parametrize("key", sorted(PLAYBOOKS))
def test_playbook_techniques_are_valid_and_ordered_without_repeats(key: str):
    playbook = PLAYBOOKS[key]

    assert playbook.techniques
    assert len(set(playbook.techniques)) == len(playbook.techniques)
    assert set(playbook.techniques) <= set(DEFENSE_TECHNIQUES)


@pytest.mark.parametrize("key", sorted(PLAYBOOKS))
def test_playbook_signature_features_are_all_usable(key: str):
    assert set(PLAYBOOKS[key].signature_features) <= set(FEATURE_TECHNIQUES)


@pytest.mark.parametrize("key", sorted(PLAYBOOKS))
def test_every_playbook_cites_an_iec_reference(key: str):
    """Um playbook "IEC-61850" sem cláusula é só uma lista de boas intenções."""

    assert "IEC" in PLAYBOOKS[key].reference


def test_the_protocol_scenarios_are_only_observable_under_the_deltas_mode():
    """Replay, flooding e grayhole são invisíveis ao IDS no modo default.

    As features que os denunciam existem, mas ``drop`` as descarta — então o
    playbook continua válido e o detector é que fica cego. É a diferença entre
    "não há defesa" e "esta execução não podia enxergar", e o catálogo precisa
    dizer qual das duas é.
    """

    strict = _usable_features("drop")

    for key in ("replayed_goose_frames", "goose_flooding", "frame_suppression"):
        signature = set(PLAYBOOKS[key].signature_features)

        assert signature, key
        assert signature & strict == set(), key
        assert signature <= _usable_features("deltas"), key


def test_playbooks_for_technique_finds_the_scenarios():
    scenarios = {p.key for p in playbooks_for_technique("goose_timing_analysis")}

    assert scenarios == {
        "replayed_goose_frames",
        "goose_flooding",
        "frame_suppression",
    }


def test_playbooks_for_an_unused_technique_is_empty():
    assert playbooks_for_technique("class_rebalancing") == ()


# --------------------------------------------------------------------------- #
# Sincronia com o contrato                                                    #
# --------------------------------------------------------------------------- #
def test_the_catalog_reaches_every_technique_in_the_vocabulary():
    """Uma técnica que nenhuma evidência e nenhum cenário alcança é nomeável e
    inútil — o Defensor poderia escolhê-la, mas nada a recomendaria."""

    assert catalogued_techniques() == set(DEFENSE_TECHNIQUES)


def test_the_catalog_never_names_a_technique_outside_the_contract():
    assert catalogued_techniques() <= set(DEFENSE_TECHNIQUES)


def test_playbook_techniques_span_more_than_one_bucket():
    """Um playbook só de detecção não é resposta: não contém nem endurece."""

    for playbook in PLAYBOOKS.values():
        buckets = {
            bucket
            for technique in playbook.techniques
            for bucket in _BUCKETS_BY_TECHNIQUE[technique]
        }

        assert len(buckets) > 1, playbook.key
