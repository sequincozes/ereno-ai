"""Quem decide o undersampler, e o que a decisão não pode ver (épico E7).

Dois decisores: ``config`` (estratégia fixada) e ``policy`` (regra declarada
sobre o desequilíbrio do treino). O teste mais importante do arquivo é o
último: a assinatura de ``decide_undersampling`` não aceita métrica nenhuma, e
é isso que impede o laço de ajustar o próprio pré-processamento até o número
melhorar.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from adversarial_ids.core.undersampling_policy import (
    UNDERSAMPLING_MODES,
    decide_undersampling,
    observed_imbalance,
)
from adversarial_ids.core.undersamplers import UndersamplerError
from adversarial_ids.domain.undersampling import (
    UndersamplingDecision,
    UndersamplingPolicy,
)

BALANCED = {"normal": 100, "masquerade_fault": 80}
SKEWED = {"normal": 10_000, "masquerade_fault": 100}


def _policy(**overrides) -> UndersamplingPolicy:
    return UndersamplingPolicy(**{"max_imbalance_ratio": 10.0, "target_ratio": 3.0, **overrides})


# --------------------------------------------------------------------------- #
# Desequilíbrio observado                                                      #
# --------------------------------------------------------------------------- #
def test_imbalance_is_the_ratio_between_the_largest_and_smallest_class():
    assert observed_imbalance(SKEWED) == 100.0


def test_a_single_class_has_no_imbalance_to_correct():
    assert observed_imbalance({"normal": 42}) == 1.0


def test_classes_with_zero_rows_do_not_create_an_infinite_imbalance():
    # Um rótulo com zero linhas é uma classe que não está no treino, não uma
    # classe infinitamente rara — dividir por ela produziria um desequilíbrio
    # inventado, e a política reamostraria por causa dele.
    assert observed_imbalance({"normal": 100, "grayhole": 0, "injection": 50}) == 2.0


def test_no_positive_class_is_an_error_and_not_a_silent_one():
    with pytest.raises(UndersamplerError, match="contagem de classe positiva"):
        observed_imbalance({"normal": 0})


# --------------------------------------------------------------------------- #
# Decisão por configuração                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["none", "random", "ratio"])
def test_a_configured_strategy_is_passed_through_untouched(strategy):
    decision = decide_undersampling(
        class_counts=SKEWED, mode=strategy, policy=_policy()
    )

    assert decision.strategy == strategy
    assert decision.decided_by == "config"


def test_a_configured_strategy_ignores_the_imbalance_but_still_records_it():
    # 100:1 está muito acima do teto da política, e mesmo assim "none" é
    # respeitado: configuração explícita não é sugestão. O número fica no
    # artefato para quem for ler o resultado depois.
    decision = decide_undersampling(class_counts=SKEWED, mode="none", policy=_policy())

    assert decision.strategy == "none"
    assert decision.observed_imbalance == 100.0
    assert "não participou da escolha" in decision.reason


def test_a_configured_ratio_takes_the_target_from_the_policy():
    decision = decide_undersampling(
        class_counts=SKEWED, mode="ratio", policy=_policy(target_ratio=4.0)
    )

    assert decision.target_ratio == 4.0


# --------------------------------------------------------------------------- #
# Decisão pela política                                                        #
# --------------------------------------------------------------------------- #
def test_policy_leaves_a_tolerable_imbalance_alone():
    decision = decide_undersampling(class_counts=BALANCED, mode="policy", policy=_policy())

    assert decision.strategy == "none"
    assert decision.decided_by == "policy"
    assert "dentro do teto declarado" in decision.reason


def test_policy_caps_an_intolerable_imbalance_with_ratio_not_with_balancing():
    # "ratio" e não "random" é a decisão de desenho: balancear um trace de
    # prevalência baixa descarta quase todo o tráfego benigno e o detector
    # passa a treinar numa distribuição que nenhuma subestação produz.
    decision = decide_undersampling(class_counts=SKEWED, mode="policy", policy=_policy())

    assert decision.strategy == "ratio"
    assert decision.decided_by == "policy"
    assert decision.target_ratio == 3.0


def test_the_ceiling_is_inclusive_so_the_policy_is_reproducible_at_the_boundary():
    exactly_at_ceiling = {"normal": 1000, "injection": 100}

    decision = decide_undersampling(
        class_counts=exactly_at_ceiling, mode="policy", policy=_policy(max_imbalance_ratio=10.0)
    )

    assert decision.strategy == "none"


def test_every_policy_decision_carries_the_measure_it_decided_on():
    decision = decide_undersampling(class_counts=SKEWED, mode="policy", policy=_policy())

    assert decision.observed_imbalance == 100.0
    assert "100.00:1" in decision.reason


def test_an_unknown_mode_lists_the_valid_ones():
    with pytest.raises(UndersamplerError, match="Modo de undersampling desconhecido"):
        decide_undersampling(class_counts=BALANCED, mode="aggressive", policy=_policy())


def test_policy_is_a_decider_and_never_a_registered_strategy():
    # "policy" vive em UNDERSAMPLING_MODES mas não em UndersamplerKey: nenhum
    # undersampler se chama assim, e o manifest grava a estratégia resolvida.
    from adversarial_ids.core.undersamplers import list_undersampler_keys

    assert "policy" in UNDERSAMPLING_MODES
    assert "policy" not in list_undersampler_keys()


# --------------------------------------------------------------------------- #
# A política como contrato                                                     #
# --------------------------------------------------------------------------- #
def test_a_target_above_the_ceiling_is_refused():
    with pytest.raises(ValidationError, match="não pode ser maior que"):
        UndersamplingPolicy(max_imbalance_ratio=3.0, target_ratio=10.0)


def test_a_target_below_one_is_refused():
    with pytest.raises(ValidationError):
        UndersamplingPolicy(target_ratio=0.5)


def test_a_decision_without_a_reason_is_refused():
    with pytest.raises(ValidationError):
        UndersamplingDecision(
            strategy="none", decided_by="policy", observed_imbalance=2.0, reason=""
        )


def test_a_decision_records_the_cut_its_strategy_actually_makes():
    with pytest.raises(ValidationError, match="exige target_ratio"):
        UndersamplingDecision(
            strategy="ratio", decided_by="config", observed_imbalance=2.0, reason="x"
        )
    with pytest.raises(ValidationError, match="não usa target_ratio"):
        UndersamplingDecision(
            strategy="random",
            decided_by="config",
            target_ratio=3.0,
            observed_imbalance=2.0,
            reason="x",
        )


# --------------------------------------------------------------------------- #
# O guardrail                                                                  #
# --------------------------------------------------------------------------- #
def test_the_decision_cannot_see_a_metric_a_report_or_an_intent():
    """O guardrail está na assinatura, e este teste é o que o mantém lá.

    Um laço que ajusta o próprio pré-processamento até a métrica melhorar
    deixa de medir o que o detector enfrenta; e deixar a intenção do atacante
    escolher o balanceamento do treino do defensor é enfraquecer o detector
    por um caminho que não é o ataque. As duas coisas se fecham sozinhas
    enquanto esta função só receber contagem de classes e política.
    """

    parameters = set(inspect.signature(decide_undersampling).parameters)

    assert parameters == {"class_counts", "mode", "policy"}
    forbidden = {"report", "detection_report", "metrics", "intent", "intent_spec", "history"}
    assert not (parameters & forbidden)
