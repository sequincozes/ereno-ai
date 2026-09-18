"""Nenhuma configuração compilada sai com um intervalo degenerado.

Um par ``{min, max}`` com ``min == max`` passa por tudo que este repo tem — o
clamp aceitava, o schema do ataque valida ``min <= max``, o portão de
capacidade não olha pares — e morre no ERENO: *"The lower limit (2000) must be
less than the upper limit (2000)"*. Foi assim que o ``random_replay`` em
intensidade alta deixou de gerar no piloto E0 (18/09/2026, ver
``docs/pilot_e0.md``): ``windowS`` fechava em ``{2.0, 2.0}``.

O teste que fecha o buraco é o varredor: **todo** ataque registrado, **todo**
efeito suportado, **toda** intensidade, todos os campos que couberem — e
nenhum par degenerado em nenhuma config resultante. Um par novo num ataque
novo entra nele sem ninguém lembrar de escrever um caso.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from adversarial_ids.config.attack_capabilities import get_attack_capability
from adversarial_ids.config.attacks_registry import list_attack_keys
from adversarial_ids.core.intent_compiler import (
    IntentCompilerError,
    _clamp_paired,
    compile_attack_candidate,
)
from adversarial_ids.domain.intent_spec import (
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)

ATTACK_KEYS = list_attack_keys()


def _ranges(node: Any, path: str = "") -> Iterator[tuple[str, Any, Any]]:
    """Todo objeto ``{min, max}`` numérico da config, com o caminho até ele."""

    if not isinstance(node, dict):
        return

    lower, upper = node.get("min"), node.get("max")
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        if not isinstance(lower, bool) and not isinstance(upper, bool):
            yield (path or "<raiz>", lower, upper)

    for key, value in node.items():
        yield from _ranges(value, f"{path}.{key}" if path else key)


def _intent(attack_key: str, effect: Any, intensity: IntentIntensity, **overrides: Any) -> IntentSpec:
    data: dict[str, Any] = {
        "source_prompt": f"Piloto de intervalos: {attack_key}/{effect.value}/{intensity.value}.",
        "objective": IntentObjective.EVADE_DETECTION,
        "base_attack": attack_key,
        "desired_effect": effect,
        "intensity": intensity,
        # Teto alto de propósito: quanto mais campos selecionados, mais pares
        # o compilador tem chance de fechar.
        "restrictions": IntentRestrictions(max_fields_changed=16),
    }
    data.update(overrides)

    return IntentSpec.model_validate(data)


@pytest.mark.parametrize("attack_key", ATTACK_KEYS)
@pytest.mark.parametrize("intensity", list(IntentIntensity))
def test_nenhum_ataque_compila_intervalo_degenerado(attack_key, intensity):
    capability = get_attack_capability(attack_key)

    for effect in sorted(capability.supported_effects, key=lambda item: item.value):
        objective = (
            IntentObjective.ASSESS_IDS_ROBUSTNESS
            if IntentObjective.ASSESS_IDS_ROBUSTNESS in capability.supported_objectives
            else next(iter(capability.supported_objectives))
        )
        intent = _intent(attack_key, effect, intensity, objective=objective)
        try:
            candidate = compile_attack_candidate(intent)
        except IntentCompilerError:
            # Nenhum campo se moveu: é resultado legítimo (e o compilador o diz
            # em voz alta), não uma config inválida escapando.
            continue

        for path, lower, upper in _ranges(candidate.config):
            assert lower < upper, (
                f"{attack_key}/{effect.value}/{intensity.value}: intervalo "
                f"degenerado em {path} ({lower} == {upper}) — o ERENO recusa."
            )


@pytest.mark.parametrize("attack_key", ATTACK_KEYS)
@pytest.mark.parametrize("intensity", list(IntentIntensity))
def test_nenhum_campo_sozinho_fecha_o_proprio_par(attack_key, intensity):
    """A varredura acima com o teto de campos alto não basta.

    Com muitos campos selecionados, os **dois** limites do par costumam se mover
    juntos e o intervalo não colapsa. O caso perigoso é o oposto: um limite
    empurrado contra um irmão **congelado**, que é o que acontece quando
    ``max_fields_changed`` é baixo — foi exatamente assim que o ``windowS`` do
    ``random_replay`` fechou no piloto, com três campos selecionados de quinze.
    Aqui cada campo é compilado sozinho.
    """

    capability = get_attack_capability(attack_key)
    objective = (
        IntentObjective.ASSESS_IDS_ROBUSTNESS
        if IntentObjective.ASSESS_IDS_ROBUSTNESS in capability.supported_objectives
        else next(iter(capability.supported_objectives))
    )

    for field in capability.fields:
        for effect in sorted(field.effects, key=lambda item: item.value):
            intent = _intent(
                attack_key,
                effect,
                intensity,
                objective=objective,
                restrictions=IntentRestrictions(allowed_fields=(field.path,)),
            )
            try:
                candidate = compile_attack_candidate(intent)
            except IntentCompilerError:
                continue

            for path, lower, upper in _ranges(candidate.config):
                assert lower < upper, (
                    f"{attack_key}/{field.path}/{effect.value}/{intensity.value}: "
                    f"intervalo degenerado em {path} ({lower} == {upper}) — "
                    "o ERENO recusa."
                )


def test_a_regressao_do_piloto_windowS_do_random_replay():
    """O caso exato que matou a variante `high` do `random_replay` no E0."""

    intent = IntentSpec.model_validate(
        {
            "source_prompt": "Reduza o recall do replay aleatório encurtando a janela.",
            "objective": IntentObjective.ASSESS_IDS_ROBUSTNESS,
            "base_attack": "random_replay",
            "desired_effect": "lower_recall",
            "intensity": IntentIntensity.HIGH,
            "restrictions": IntentRestrictions(allowed_fields=("windowS.max",)),
        }
    )
    candidate = compile_attack_candidate(intent)

    window = candidate.config["windowS"]
    assert window["min"] == pytest.approx(2.0)
    # baseline 8.0, piso do par 2.0, passo 0.85: 8 + 0.85 * (2 - 8) = 2.9.
    assert window["max"] == pytest.approx(2.9)
    assert window["min"] < window["max"]


def test_intensidade_maior_chega_mais_perto_do_irmao_sem_encostar():
    def compiled(intensity: IntentIntensity) -> float:
        intent = IntentSpec.model_validate(
            {
                "source_prompt": "Encurte a janela do replay aleatório.",
                "objective": IntentObjective.ASSESS_IDS_ROBUSTNESS,
                "base_attack": "random_replay",
                "desired_effect": "lower_recall",
                "intensity": intensity,
                "restrictions": IntentRestrictions(allowed_fields=("windowS.max",)),
            }
        )
        return compile_attack_candidate(intent).config["windowS"]["max"]

    low = compiled(IntentIntensity.LOW)
    medium = compiled(IntentIntensity.MEDIUM)
    high = compiled(IntentIntensity.HIGH)

    # A escada continua monotônica depois do clamp — o irmão vira o limite
    # efetivo, então a intensidade ainda ordena as variantes.
    assert low > medium > high > 2.0


def test_par_de_inteiros_recua_uma_unidade_quando_o_arredondamento_cola_no_irmao():
    # burst = {min: 2, max: 5} no random_replay; 5 + 0.85 * (2 - 5) = 2.45, que
    # arredonda para 2 e encostaria no piso.
    config = {"burst": {"min": 2, "max": 5}}

    value = _clamp_paired(config, "burst.max", 1, baseline=5, step=0.85)

    assert value == 3


def test_par_de_inteiros_sem_espaco_nao_move_o_campo():
    # Teto 3 e piso 2: não existe inteiro estritamente entre eles na direção do
    # efeito, então o campo fica onde está e some do diff.
    config = {"burst": {"min": 2, "max": 3}}

    value = _clamp_paired(config, "burst.max", 1, baseline=3, step=0.85)

    assert value == 3


def test_valor_que_nao_cruza_o_irmao_passa_intacto():
    config = {"windowS": {"min": 2.0, "max": 8.0}}

    assert _clamp_paired(config, "windowS.max", 6.5, baseline=8.0, step=0.5) == 6.5


def test_campo_sem_irmao_de_par_nao_e_tocado():
    config = {"reorderProb": 0.05}

    assert _clamp_paired(config, "reorderProb", 0.01, baseline=0.05, step=0.85) == 0.01
