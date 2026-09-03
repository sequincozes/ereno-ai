"""Compilador spec→AttackCandidate (épico E2, ação 72h #5 do plano de 60 dias).

Cobre o caminho ``IntentSpec`` → ``AttackCandidate`` para ``masquerade_fault``:
seleção determinística de campos, direção do efeito, clamps de pares
min/max, alternância de campos não numéricos e reprodutibilidade por seed.
Reaproveita o fixture de golden prompts do IntentAgent — cada intenção válida
ou ambígua já autorizada deve compilar num candidato consistente.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adversarial_ids.core.intent_compiler import (
    IntentCompilerError,
    compile_attack_candidate,
)
from adversarial_ids.domain.attack_candidate import AttackCandidate
from adversarial_ids.domain.attack_configs import MasqueradeFaultConfig
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)
from adversarial_ids.shared.json_io import load_json
from adversarial_ids.config.attacks_registry import get_attack_spec
from tests.fakes.fake_intent_agent import FakeIntentAgent

FIXTURE_PATH = Path(__file__).parent / "golden_prompts" / "masquerade_fault_intents.json"
ENTRIES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
VALID_ENTRIES = [entry for entry in ENTRIES if entry["category"] in ("valid", "ambiguous")]

_BASELINE = load_json(get_attack_spec("masquerade_fault").baseline_path)


def _intent(**overrides: object) -> IntentSpec:
    data: dict[str, object] = {
        "source_prompt": "Reduza o recall variando a temporização da falha.",
        "objective": IntentObjective.EVADE_DETECTION,
        "base_attack": "masquerade_fault",
        "desired_effect": DesiredEffect.LOWER_RECALL,
    }
    data.update(overrides)
    return IntentSpec.model_validate(data)


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=[e["id"] for e in VALID_ENTRIES])
def test_valid_and_ambiguous_intents_compile_to_a_consistent_candidate(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])
    candidate = compile_attack_candidate(intent)

    assert isinstance(candidate, AttackCandidate)
    assert candidate.attack_key == intent.base_attack
    assert candidate.source_intent == intent
    MasqueradeFaultConfig.model_validate(candidate.config)

    changed_paths = {change.path for change in candidate.diff}
    assert len(changed_paths) == len(candidate.diff)
    assert len(candidate.diff) <= intent.restrictions.max_fields_changed

    if intent.restrictions.allowed_fields:
        assert changed_paths <= set(intent.restrictions.allowed_fields)
    assert changed_paths.isdisjoint(intent.restrictions.forbidden_fields)

    for change in candidate.diff:
        assert change.old_value != change.new_value


@pytest.mark.parametrize("entry", VALID_ENTRIES, ids=[e["id"] for e in VALID_ENTRIES])
def test_compilation_is_deterministic_given_the_same_intent(entry):
    intent = FakeIntentAgent.compile(entry["payload"], entry["prompt"])

    first = compile_attack_candidate(intent)
    second = compile_attack_candidate(intent)

    assert first.config == second.config
    assert first.diff == second.diff


def test_mimic_normal_traffic_moves_values_toward_the_baseline_floor():
    intent = _intent(
        desired_effect=DesiredEffect.MIMIC_NORMAL_TRAFFIC,
        restrictions=IntentRestrictions(
            allowed_fields=("analog.deltaAbs.min", "analog.deltaAbs.max")
        ),
    )
    candidate = compile_attack_candidate(intent)

    for change in candidate.diff:
        assert change.new_value < change.old_value


def test_increase_attack_activity_moves_the_lone_candidate_upward():
    intent = _intent(
        objective=IntentObjective.ASSESS_IDS_ROBUSTNESS,
        desired_effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        restrictions=IntentRestrictions(forbidden_fields=("fault.prob",)),
    )
    candidate = compile_attack_candidate(intent)

    assert [c.path for c in candidate.diff] == ["trapArea.spikeProb"]
    change = candidate.diff[0]
    assert change.new_value > change.old_value
    assert change.new_value <= 1.0


def test_sqnum_mode_toggles_to_the_alternate_choice():
    intent = _intent(
        restrictions=IntentRestrictions(allowed_fields=("sqnumMode",)),
    )
    candidate = compile_attack_candidate(intent)

    assert candidate.diff[0].path == "sqnumMode"
    assert candidate.diff[0].old_value == "fast"
    assert candidate.diff[0].new_value == "normal"


def test_ttl_ms_values_are_scaled_elementwise_and_stay_positive():
    intent = _intent(restrictions=IntentRestrictions(allowed_fields=("ttlMsValues",)))
    candidate = compile_attack_candidate(intent)

    baseline_values = _BASELINE["ttlMsValues"]
    change = candidate.diff[0]
    assert change.old_value == baseline_values
    assert len(change.new_value) == len(baseline_values)
    assert all(value >= 1 for value in change.new_value)
    assert all(new <= old for new, old in zip(change.new_value, baseline_values))


def test_a_field_outside_the_effects_capability_is_rejected_before_compiling():
    intent = _intent(
        desired_effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        objective=IntentObjective.ASSESS_IDS_ROBUSTNESS,
        intensity=IntentIntensity.HIGH,
        restrictions=IntentRestrictions(allowed_fields=("fault.durationMs.min",)),
    )
    # increase_attack_activity só toca fault.prob/trapArea.spikeProb no catálogo;
    # fault.durationMs.min não produz esse efeito — a resolução deve rejeitar.
    with pytest.raises(ValueError, match="removem todos os campos"):
        compile_attack_candidate(intent)


def test_analog_delta_max_alone_is_clamped_to_the_unchanged_min_floor():
    """Decrescer só o teto do par sem tocar o piso não pode invertê-los.

    Em intensidade alta, decrescer ``analog.deltaAbs.max`` sozinho levaria o
    valor abaixo do ``analog.deltaAbs.min`` (0.2) da baseline, que não foi
    selecionado. ``_clamp_paired`` trava o teto no piso ainda vigente em vez
    de deixar o schema por ataque rejeitar a config compilada.
    """
    intent = _intent(
        desired_effect=DesiredEffect.MIMIC_NORMAL_TRAFFIC,
        intensity=IntentIntensity.HIGH,
        restrictions=IntentRestrictions(allowed_fields=("analog.deltaAbs.max",)),
    )
    candidate = compile_attack_candidate(intent)

    change = candidate.diff[0]
    assert change.path == "analog.deltaAbs.max"
    assert change.new_value == pytest.approx(0.2)
    assert candidate.config["analog"]["deltaAbs"]["min"] == pytest.approx(0.2)


def test_seed_changes_which_fields_are_selected_when_candidates_exceed_the_limit():
    base = dict(
        objective=IntentObjective.EVADE_DETECTION,
        desired_effect=DesiredEffect.LOWER_F1,
        intensity=IntentIntensity.HIGH,
    )
    first = compile_attack_candidate(_intent(seed=1, **base))
    second = compile_attack_candidate(_intent(seed=2, **base))

    assert len(first.diff) == 3
    assert len(second.diff) == 3
    assert {c.path for c in first.diff} != {c.path for c in second.diff}


def test_unsupported_effect_raises_before_touching_the_config():
    intent = IntentSpec.model_validate(
        {
            "source_prompt": "Sobrecarregue os recursos do sistema.",
            "objective": IntentObjective.EVADE_DETECTION,
            "base_attack": "masquerade_fault",
            "desired_effect": DesiredEffect.INCREASE_RESOURCE_PRESSURE,
        }
    )
    with pytest.raises(ValueError, match="não é suportado"):
        compile_attack_candidate(intent)


def test_compiled_config_round_trips_through_json_like_the_baseline_file():
    intent = _intent(restrictions=IntentRestrictions(allowed_fields=("fault.prob",)))
    candidate = compile_attack_candidate(intent)

    reloaded = json.loads(json.dumps(candidate.config))
    assert reloaded == candidate.config
    MasqueradeFaultConfig.model_validate(reloaded)


def test_clamp_pairs_a_range_the_old_hardcoded_map_never_listed():
    """``_paired_sibling_path`` é estrutural — detecta qualquer objeto
    ``{min, max}``, não só os 3 pares do ``masquerade_fault`` que o antigo
    ``_PAIRED_SIBLING`` listava à mão. ``random_replay.burst`` tem ``min``/
    ``max`` convivendo com ``prob`` e ``gapMs`` no mesmo objeto — um caso que
    uma lista de 6 caminhos nunca cobriria.
    """
    intent = IntentSpec.model_validate(
        {
            "source_prompt": "Reduza a atividade do replay aleatório mexendo só no teto da rajada.",
            "objective": IntentObjective.EVADE_DETECTION,
            "base_attack": "random_replay",
            "desired_effect": DesiredEffect.MIMIC_NORMAL_TRAFFIC,
            "intensity": IntentIntensity.HIGH,
            "restrictions": IntentRestrictions(allowed_fields=("burst.max",)),
        }
    )
    candidate = compile_attack_candidate(intent)

    change = candidate.diff[0]
    assert change.path == "burst.max"
    # baseline: burst.min=2, burst.max=5 — decrescer só o teto não pode
    # deixá-lo abaixo do piso ainda vigente.
    assert change.new_value >= candidate.config["burst"]["min"]
