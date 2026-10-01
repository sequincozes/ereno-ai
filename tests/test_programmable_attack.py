"""Ataque programável (uc11) — comportamentos fora do catálogo (Fases 2.1/2.2).

O comportamento é uma lista de regras (op/campo/valor/fração/seleção por slot),
não uma classe por comportamento. Estes testes fixam, sem Java nem Groq:

- o schema congelado das regras (dict de slots, não array);
- a capacidade: só `fraction` é alavanca de intensidade, o resto é author-only
  (efeito vazio) — o LLM preenche via `target_values`;
- o compilador materializa as regras a partir da intenção;
- **seleção** (`when`) e **temporização** (`add` em t/timestamp, `drop`,
  `duplicate`) da Fase 2.2, incluindo as regras de coerência entre campos da
  mesma regra, que nenhuma allowlist campo-a-campo consegue expressar;
- o ataque está registrado e resolve capacidade/schema/playbook como os demais.

A geração real com o JAR (a classe `programmable` no trace) é verificada à parte,
fora da suíte — ver `docs/programmable_attack.md`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adversarial_ids.config.attack_capabilities import (
    get_attack_capability,
    resolve_candidate_paths,
)
from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.config.defense_techniques import playbook_for_attack
from adversarial_ids.core.intent_compiler import (
    IntentCompilerError,
    compile_attack_candidate,
)
from adversarial_ids.domain.attack_configs import config_model_for
from adversarial_ids.domain.attack_configs.programmable import (
    MAX_EXTRA_COPIES,
    MUTABLE_FIELDS,
    TIME_FIELDS,
    MutationRule,
    ProgrammableConfig,
)
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentSpec,
)
from adversarial_ids.shared.json_io import load_json

BEHAVIOUR = "força o disjuntor em 1 em 30% das mensagens"

# Seleção neutra: mira toda mensagem. É o default do baseline, e é o que faz
# uma regra sem seleção se comportar como antes da Fase 2.2.
ALWAYS = {"field": "stNum", "cmp": "always", "value": 0}


def _rule(**overrides):
    rule = {"op": "set", "field": "cbStatus", "value": 1, "fraction": 0.5, "when": ALWAYS}
    rule.update(overrides)
    return rule


def _intent(target_values, *, effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
            objective=IntentObjective.ASSESS_IDS_ROBUSTNESS, max_fields=8,
            intensity=IntentIntensity.MEDIUM):
    return IntentSpec.model_validate(
        {
            "source_prompt": BEHAVIOUR,
            "objective": objective,
            "base_attack": "programmable",
            "desired_effect": effect,
            "intensity": intensity,
            "restrictions": {
                "max_fields_changed": max_fields,
                "target_values": [
                    {"path": p, "value": v} for p, v in target_values.items()
                ],
            },
        }
    )


# --------------------------------------------------------------------------- #
# Registro e resolução                                                        #
# --------------------------------------------------------------------------- #
def test_programmable_is_a_registered_attack():
    assert "programmable" in list_attack_keys()
    spec = get_attack_spec("programmable")
    assert spec.label == "programmable"
    assert spec.segment_name == "uc11_programmable"


def test_programmable_resolves_capability_schema_and_playbook():
    assert get_attack_capability("programmable").attack_key == "programmable"
    assert config_model_for("programmable") is ProgrammableConfig
    assert "programmable" in playbook_for_attack("programmable").attack_keys


def test_the_baseline_validates_against_the_schema():
    baseline = load_json(get_attack_spec("programmable").baseline_path)
    model = ProgrammableConfig.model_validate(baseline)
    assert model.attackType == "programmable"
    assert set(model.rules) == {"r0", "r1"}


def test_the_baseline_selects_every_message():
    """O `when` do baseline é o neutro — ligar a seleção não muda o default.

    Importa porque `target_values` não cria chave que não existe: o `when`
    precisa estar no baseline para ser autorável, e precisa ser inerte para
    que o ataque default continue sendo o mesmo de antes da Fase 2.2.
    """

    baseline = ProgrammableConfig.model_validate(
        load_json(get_attack_spec("programmable").baseline_path)
    )
    assert all(rule.when.cmp == "always" for rule in baseline.rules.values())


# --------------------------------------------------------------------------- #
# Schema das regras                                                           #
# --------------------------------------------------------------------------- #
def test_rules_are_a_dict_of_slots_not_an_array():
    cfg = ProgrammableConfig.model_validate(
        {"attackType": "programmable", "enabled": True, "rules": {"r0": _rule()}}
    )
    assert isinstance(cfg.rules, dict)
    assert cfg.rules["r0"].op == "set"


def test_an_empty_rules_object_is_rejected():
    with pytest.raises(ValidationError):
        ProgrammableConfig.model_validate(
            {"attackType": "programmable", "enabled": True, "rules": {}}
        )


def test_a_rule_with_an_unknown_op_is_rejected():
    with pytest.raises(ValidationError):
        MutationRule.model_validate(_rule(op="toggle"))


def test_a_rule_fraction_outside_zero_one_is_rejected():
    with pytest.raises(ValidationError):
        MutationRule.model_validate(_rule(fraction=1.5))


def test_a_rule_without_a_selection_is_rejected():
    """`when` é obrigatório no schema de propósito: o compilador não cria
    chaves ausentes, então uma regra sem seleção nunca seria autorável."""

    rule = _rule()
    del rule["when"]
    with pytest.raises(ValidationError):
        MutationRule.model_validate(rule)


def test_an_unknown_comparator_is_rejected():
    with pytest.raises(ValidationError):
        MutationRule.model_validate(_rule(when={**ALWAYS, "cmp": "approx"}))


# --------------------------------------------------------------------------- #
# Fase 2.2 — temporização: coerência entre op e campo                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", TIME_FIELDS)
@pytest.mark.parametrize("op", ["set", "scale"])
def test_absolute_time_fields_accept_only_a_shift(field, op):
    """`set`/`scale` num relógio absoluto realocam a mensagem para um ponto
    arbitrário da captura em vez de atrasá-la — o schema recusa."""

    with pytest.raises(ValidationError, match="campo de tempo"):
        MutationRule.model_validate(_rule(op=op, field=field))


@pytest.mark.parametrize("field", TIME_FIELDS)
@pytest.mark.parametrize("value", [0.01, -0.01])
def test_a_signed_shift_on_a_time_field_is_the_timing_lever(field, value):
    """Atraso e reordenação são a MESMA op: `add` com o sinal do operando.

    Não há op `delay`/`reorder` porque não poderia haver — o fluxo escrito é
    ordenado por `timestamp`, então deslocar o relógio *é* reordenar.
    """

    rule = MutationRule.model_validate(_rule(op="add", field=field, value=value))
    assert rule.op == "add"
    assert rule.value == value


def test_the_integer_fields_still_take_every_mutation_op():
    for op in ("set", "add", "scale"):
        assert MutationRule.model_validate(_rule(op=op, field="sqNum", value=2)).op == op


# --------------------------------------------------------------------------- #
# Fase 2.2 — cardinalidade: drop e duplicate                                  #
# --------------------------------------------------------------------------- #
def test_drop_is_a_legal_op():
    assert MutationRule.model_validate(_rule(op="drop")).op == "drop"


@pytest.mark.parametrize("copies", [1, MAX_EXTRA_COPIES])
def test_duplicate_accepts_a_bounded_number_of_extra_copies(copies):
    assert MutationRule.model_validate(_rule(op="duplicate", value=copies)).value == copies


@pytest.mark.parametrize("copies", [0, -1, MAX_EXTRA_COPIES + 1])
def test_duplicate_outside_the_copy_bound_is_rejected(copies):
    """O teto não é zelo: `duplicate` multiplica o trace inteiro, e `value` é
    um campo sem faixa no catálogo (ele serve a `set`/`scale` também), então
    esta é a única barreira do lado Python."""

    with pytest.raises(ValidationError, match="cópias extras"):
        MutationRule.model_validate(_rule(op="duplicate", value=copies))


# --------------------------------------------------------------------------- #
# Capacidade: fraction é alavanca, o resto é author-only                      #
# --------------------------------------------------------------------------- #
def test_only_fraction_fields_carry_an_effect():
    capability = get_attack_capability("programmable")
    for field in capability.fields:
        if field.path.endswith(".fraction"):
            assert field.effects  # alavanca de intensidade
        else:
            assert field.effects == frozenset()  # author-only via target_values


def test_the_capability_exposes_the_selection_of_every_slot():
    paths = get_attack_capability("programmable").editable_paths
    for slot in ("r0", "r1"):
        assert {f"rules.{slot}.when.{leaf}" for leaf in ("field", "cmp", "value")} <= paths


def test_the_capability_offers_the_time_fields_the_creator_can_touch():
    capability = get_attack_capability("programmable")
    field = next(f for f in capability.fields if f.path == "rules.r0.field")
    assert set(field.choices) == set(MUTABLE_FIELDS)
    assert set(TIME_FIELDS) <= set(field.choices)


def test_the_heuristic_sweeps_only_the_fraction_fields():
    """Sem valores ditados, a intenção só move os `fraction` — o comportamento
    (op/field/value/when) fica como no baseline."""

    intent = _intent({}, max_fields=8)
    candidate = compile_attack_candidate(intent)

    assert candidate.diff  # mudou algo
    assert all(change.path.endswith(".fraction") for change in candidate.diff)


# --------------------------------------------------------------------------- #
# Compilador materializa o comportamento autorado                            #
# --------------------------------------------------------------------------- #
def test_the_llm_authors_a_behaviour_through_target_values():
    intent = _intent(
        {
            "rules.r0.op": "scale",
            "rules.r0.field": "sqNum",
            "rules.r0.value": 3,
            "rules.r0.fraction": 0.4,
        },
        max_fields=4,
    )

    config = compile_attack_candidate(intent).config

    assert config["rules"]["r0"] == {
        "op": "scale", "field": "sqNum", "value": 3, "fraction": 0.4, "when": ALWAYS
    }


def test_the_llm_authors_a_selection_through_target_values():
    """"só nas mensagens com o disjuntor aberto" → uma condição, não uma
    família de ataque nova."""

    intent = _intent(
        {
            "rules.r0.when.field": "cbStatus",
            "rules.r0.when.cmp": "eq",
            "rules.r0.when.value": 0,
        },
        max_fields=3,
    )

    config = compile_attack_candidate(intent).config

    assert config["rules"]["r0"]["when"] == {"field": "cbStatus", "cmp": "eq", "value": 0}


def test_the_llm_authors_a_delay_through_target_values():
    intent = _intent(
        {"rules.r0.op": "add", "rules.r0.field": "timestamp", "rules.r0.value": 0.05},
        max_fields=3,
    )

    config = compile_attack_candidate(intent).config

    assert config["rules"]["r0"]["field"] == "timestamp"
    assert config["rules"]["r0"]["value"] == 0.05


def test_an_authored_value_is_never_clamped():
    """Um campo mutável fora da allowlist do creator é recusado pelo portão."""

    intent = _intent({"rules.r0.field": "frameLen"}, max_fields=1)
    with pytest.raises(ValueError, match="fora das opções"):
        resolve_candidate_paths(intent)


def test_an_authored_comparator_outside_the_grammar_is_refused():
    intent = _intent({"rules.r0.when.cmp": "approx"}, max_fields=1)
    with pytest.raises(ValueError, match="fora das opções"):
        resolve_candidate_paths(intent)


def test_an_incoherent_authored_rule_fails_the_compile():
    """O portão de capacidades aprova `op=scale` e `field=timestamp` separados
    — cada um está na sua allowlist. Quem recusa a combinação é o schema, no
    compilador: é a razão de a regra de coerência existir."""

    intent = _intent(
        {"rules.r0.op": "scale", "rules.r0.field": "timestamp", "rules.r0.value": 2},
        max_fields=3,
    )
    resolve_candidate_paths(intent)  # o portão campo-a-campo passa

    with pytest.raises(IntentCompilerError, match="campo de tempo"):
        compile_attack_candidate(intent)


def test_the_compiled_config_validates_against_the_schema():
    intent = _intent(
        {"rules.r0.op": "add", "rules.r0.field": "confRev", "rules.r0.value": 2,
         "rules.r0.fraction": 0.6},
        max_fields=4,
    )
    candidate = compile_attack_candidate(intent)

    # compile_attack_candidate já valida internamente; reconfirmar é barato.
    ProgrammableConfig.model_validate(candidate.config)
    assert candidate.attack_key == "programmable"


def test_the_same_authored_intent_compiles_deterministically():
    tv = {"rules.r0.op": "scale", "rules.r0.field": "stNum",
          "rules.r0.value": 2, "rules.r0.fraction": 0.5}
    first = compile_attack_candidate(_intent(tv, max_fields=4))
    second = compile_attack_candidate(_intent(tv, max_fields=4))

    assert first.config == second.config
    assert first.diff == second.diff
