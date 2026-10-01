"""Testes dos valores ditados pelo pedido (``IntentRestrictions.target_values``).

O prompt deixa de escolher só *qual* ataque e *quanto* de intensidade: passa a
poder dizer o valor exato de um campo ("duração entre 50 e 80 ms"). A regra que
todos estes testes cercam é uma só — **um valor ditado nunca é corrigido em
silêncio**: ou ele vale exatamente como pedido, ou a compilação falha dizendo
por quê.

Três camadas, nesta ordem:

- contrato (``domain/intent_spec.py``): contradições internas do pedido;
- portão de capacidades (``config/attack_capabilities.py``): o que o catálogo
  decide sem abrir a baseline do ataque;
- compilador (``core/intent_compiler.py``): o que só a baseline decide.

Baseline de ``masquerade_fault`` (``inputs/attacks/uc03_masquerade_fault.json``):
``fault.prob=0.6``, ``fault.durationMs={50, 800}``, ``cbStatus=1``,
``ttlMsValues=[20, 40, 80]``.
"""

from __future__ import annotations

import json

import pytest

from adversarial_ids.config.attack_capabilities import (
    get_attack_capability,
    resolve_candidate_paths,
    validate_target_values,
)
from adversarial_ids.core.intent_compiler import (
    IntentCompilerError,
    compile_attack_candidate,
)
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
    TargetValue,
)

PROMPT = "Reduza o recall com a duração da falha entre 50 e 80 ms."


def _intent(
    *,
    pins: dict | None = None,
    max_fields_changed: int = 3,
    intensity: IntentIntensity = IntentIntensity.MEDIUM,
    effect: DesiredEffect = DesiredEffect.LOWER_RECALL,
    attack: str = "masquerade_fault",
    allowed: tuple[str, ...] | None = None,
    forbidden: tuple[str, ...] = (),
    seed: int = 42,
) -> IntentSpec:
    return IntentSpec.model_validate(
        {
            "source_prompt": PROMPT,
            "objective": IntentObjective.EVADE_DETECTION,
            "base_attack": attack,
            "desired_effect": effect,
            "intensity": intensity,
            "seed": seed,
            "restrictions": {
                "max_fields_changed": max_fields_changed,
                "allowed_fields": allowed,
                "forbidden_fields": forbidden,
                "target_values": tuple(
                    {"path": path, "value": value}
                    for path, value in sorted((pins or {}).items())
                ),
            },
        }
    )


def _config_of(intent: IntentSpec) -> dict:
    return compile_attack_candidate(intent).config


# --------------------------------------------------------------------------- #
# Contrato — contradições internas do pedido                                  #
# --------------------------------------------------------------------------- #
def test_a_spec_without_target_values_keeps_loading_as_before():
    """Aditivo com default: um artefato gravado antes desta versão carrega."""

    spec = IntentSpec.model_validate(
        {
            "source_prompt": PROMPT,
            "objective": "evade_detection",
            "base_attack": "masquerade_fault",
            "desired_effect": "lower_recall",
        }
    )

    assert spec.restrictions.target_values == ()
    assert spec.schema_version == 1


def test_target_values_survive_a_json_round_trip():
    intent = _intent(pins={"fault.durationMs.max": 80})

    restored = IntentSpec.model_validate(json.loads(intent.model_dump_json()))

    assert restored == intent
    assert restored.restrictions.target_values[0].path == "fault.durationMs.max"
    assert restored.restrictions.target_values[0].value == 80


def test_the_same_field_cannot_be_pinned_twice():
    with pytest.raises(ValueError, match="dois valores fixados"):
        IntentRestrictions(
            target_values=(
                TargetValue(path="fault.prob", value=0.2),
                TargetValue(path="fault.prob", value=0.9),
            )
        )


def test_a_pinned_field_cannot_also_be_forbidden():
    with pytest.raises(ValueError, match="fixado e ser proibidos"):
        IntentRestrictions(
            target_values=(TargetValue(path="fault.prob", value=0.2),),
            forbidden_fields=("fault.prob",),
        )


def test_a_pinned_field_must_be_inside_allowed_fields_when_it_is_set():
    with pytest.raises(ValueError, match="precisam estar em allowed_fields"):
        IntentRestrictions(
            target_values=(TargetValue(path="fault.prob", value=0.2),),
            allowed_fields=("cbStatus",),
        )


def test_a_quoted_number_inside_a_list_is_refused_instead_of_converted():
    with pytest.raises(ValueError, match="sem aspas"):
        TargetValue(path="ttlMsValues", value=(20, "40"))


def test_target_value_rejects_an_unknown_key():
    with pytest.raises(ValueError):
        TargetValue(path="fault.prob", value=0.2, unit="ms")


# --------------------------------------------------------------------------- #
# Portão de capacidades — o que o catálogo decide sem a baseline              #
# --------------------------------------------------------------------------- #
def test_a_pinned_path_outside_the_catalog_is_refused():
    intent = _intent(pins={"fault.naoExiste": 5})

    with pytest.raises(ValueError, match="fora da allowlist"):
        resolve_candidate_paths(intent)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("fault.durationMs.max", 80.5, "esperava inteiro"),
        ("fault.durationMs.max", True, "esperava inteiro"),
        ("fault.prob", "alta", "esperava número"),
        ("incrementStNumOnFault", 1, "esperava booleano"),
        ("sqnumMode", 1, "esperava texto"),
        ("ttlMsValues", 20, "esperava lista"),
    ],
)
def test_a_pinned_value_of_the_wrong_type_is_refused(path, value, message):
    capability = get_attack_capability("masquerade_fault")
    intent = _intent(pins={path: value})

    with pytest.raises(ValueError, match=message):
        validate_target_values(capability, intent)


def test_a_boolean_is_never_accepted_as_an_integer_field():
    """``isinstance(True, int)`` é verdadeiro em Python — sem a checagem
    explícita, ``cbStatus: true`` chegaria ao JSON do ERENO."""

    capability = get_attack_capability("masquerade_fault")

    with pytest.raises(ValueError, match="esperava inteiro"):
        validate_target_values(capability, _intent(pins={"cbStatus": True}))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("fault.prob", 1.4, "acima do máximo"),
        ("fault.prob", -0.1, "abaixo do mínimo"),
        ("fault.durationMs.min", -5, "abaixo do mínimo"),
        ("cbStatus", 7, "fora das opções"),
        ("sqnumMode", "turbo", "fora das opções"),
    ],
)
def test_a_pinned_value_outside_the_fields_domain_is_refused(path, value, message):
    capability = get_attack_capability("masquerade_fault")

    with pytest.raises(ValueError, match=message):
        validate_target_values(capability, _intent(pins={path: value}))


def test_an_integer_is_accepted_where_the_field_expects_a_number():
    """O pedido diz "probabilidade 1", não "1.0" — exigir o ponto seria exigir
    sintaxe de ponto flutuante do texto em linguagem natural."""

    capability = get_attack_capability("masquerade_fault")

    assert validate_target_values(capability, _intent(pins={"fault.prob": 1})) == {
        "fault.prob": 1
    }


def test_a_pinned_range_that_is_inverted_is_refused_before_the_baseline():
    capability = get_attack_capability("masquerade_fault")
    intent = _intent(
        pins={"fault.durationMs.min": 80, "fault.durationMs.max": 50},
        max_fields_changed=2,
    )

    with pytest.raises(ValueError, match="intervalo pedido.*invertido"):
        validate_target_values(capability, intent)


def test_a_pinned_range_that_collapses_onto_itself_is_refused():
    """``min == max`` passa por todo gate deste repo e o ERENO recusa a execução
    inteira — mesma invariante do piloto E0, agora para valores ditados."""

    capability = get_attack_capability("masquerade_fault")
    intent = _intent(
        pins={"fault.durationMs.min": 80, "fault.durationMs.max": 80},
        max_fields_changed=2,
    )

    with pytest.raises(ValueError, match="invertido"):
        validate_target_values(capability, intent)


def test_pinning_more_fields_than_the_budget_is_refused_with_the_number_to_use():
    intent = _intent(
        pins={"fault.durationMs.min": 40, "fault.durationMs.max": 90, "cbStatus": 0},
        max_fields_changed=2,
    )

    with pytest.raises(ValueError, match="max_fields_changed >= 3"):
        validate_target_values(get_attack_capability("masquerade_fault"), intent)


def test_pins_alone_are_enough_when_no_candidate_field_carries_the_effect():
    """Um pedido inteiramente ditado não "removeu todos os campos" — ele nomeou
    os campos um por um.

    ``cbStatus`` não carrega ``increase_attack_activity``, então a interseção
    com ``allowed_fields`` fica vazia; antes dos valores ditados isso era um
    erro, e continua sendo quando não há nada fixado.
    """

    pinned = _intent(
        pins={"cbStatus": 0},
        allowed=("cbStatus",),
        effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        max_fields_changed=1,
    )
    _, candidates = resolve_candidate_paths(pinned)
    assert candidates == frozenset()
    assert compile_attack_candidate(pinned).config["cbStatus"] == 0

    bare = _intent(
        allowed=("cbStatus",),
        effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        max_fields_changed=1,
    )
    with pytest.raises(ValueError, match="removem todos os campos"):
        resolve_candidate_paths(bare)


# --------------------------------------------------------------------------- #
# Compilador — o valor pedido chega intacto                                   #
# --------------------------------------------------------------------------- #
def test_a_pinned_value_lands_exactly_in_the_compiled_config():
    config = _config_of(
        _intent(
            pins={"fault.durationMs.min": 50, "fault.durationMs.max": 80},
            max_fields_changed=2,
        )
    )

    assert config["fault"]["durationMs"] == {"min": 50, "max": 80}


@pytest.mark.parametrize(
    "intensity", [IntentIntensity.LOW, IntentIntensity.MEDIUM, IntentIntensity.HIGH]
)
def test_intensity_never_moves_a_pinned_field(intensity):
    config = _config_of(
        _intent(
            pins={"fault.prob": 0.42},
            allowed=("fault.prob",),
            max_fields_changed=1,
            intensity=intensity,
        )
    )

    assert config["fault"]["prob"] == 0.42


def test_a_pinned_field_appears_in_the_diff_with_its_baseline_as_old_value():
    candidate = compile_attack_candidate(
        _intent(pins={"fault.prob": 0.42}, allowed=("fault.prob",), max_fields_changed=1)
    )

    (change,) = candidate.diff
    assert change.path == "fault.prob"
    assert change.old_value == 0.6
    assert change.new_value == 0.42


def test_the_rationale_says_which_fields_came_from_the_request():
    candidate = compile_attack_candidate(
        _intent(pins={"fault.prob": 0.42}, allowed=("fault.prob",), max_fields_changed=1)
    )

    assert "valor fixado no pedido" in candidate.rationale
    assert "fault.prob" in candidate.rationale


def test_a_pin_consumes_the_budget_the_heuristic_would_have_used():
    pinned_only = compile_attack_candidate(
        _intent(pins={"fault.prob": 0.42}, max_fields_changed=1)
    )

    assert [change.path for change in pinned_only.diff] == ["fault.prob"]


def test_the_heuristic_fills_whatever_the_pins_left_of_the_budget():
    candidate = compile_attack_candidate(
        _intent(pins={"cbStatus": 0}, max_fields_changed=3)
    )

    paths = {change.path for change in candidate.diff}
    assert "cbStatus" in paths
    # Dois campos de folga para a heurística, e ela não repete o campo fixado.
    assert 1 < len(paths) <= 3


def test_a_pinned_value_equal_to_the_baseline_is_not_reported_as_a_change():
    candidate = compile_attack_candidate(
        _intent(pins={"cbStatus": 1}, max_fields_changed=3)
    )

    assert "cbStatus" not in {change.path for change in candidate.diff}


def test_a_request_that_only_pins_the_baseline_fails_saying_so():
    intent = _intent(pins={"cbStatus": 1}, allowed=("cbStatus",), max_fields_changed=1)

    with pytest.raises(IntentCompilerError, match="já são os da baseline"):
        compile_attack_candidate(intent)


def test_an_integer_list_pin_reaches_the_config_as_a_json_list():
    config = _config_of(
        _intent(
            pins={"ttlMsValues": (10, 15)},
            allowed=("ttlMsValues",),
            max_fields_changed=1,
        )
    )

    assert config["ttlMsValues"] == [10, 15]
    assert isinstance(config["ttlMsValues"], list)


def test_a_pin_against_a_baseline_sibling_that_inverts_the_range_fails():
    """``min=900`` contra o ``max=800`` da baseline: o compilador recusa em vez
    de aproximar o valor, que é o que ele faria com um campo da heurística."""

    intent = _intent(
        pins={"fault.durationMs.min": 900},
        allowed=("fault.durationMs.min",),
        max_fields_changed=1,
    )

    with pytest.raises(IntentCompilerError, match="não forma um intervalo"):
        compile_attack_candidate(intent)


def test_the_heuristic_sibling_yields_to_the_pin_instead_of_the_other_way_round():
    """Com ``max`` fixado, o ``min`` escolhido pela heurística ainda precisa
    ficar abaixo dele — o clamp enxerga o valor pedido porque os fixados são
    escritos antes."""

    config = _config_of(
        _intent(
            pins={"fault.durationMs.max": 120},
            allowed=("fault.durationMs.min", "fault.durationMs.max"),
            max_fields_changed=2,
            intensity=IntentIntensity.HIGH,
        )
    )

    assert config["fault"]["durationMs"]["max"] == 120
    assert config["fault"]["durationMs"]["min"] < 120


def test_the_compiled_config_still_validates_against_the_attacks_schema():
    candidate = compile_attack_candidate(
        _intent(
            pins={"fault.durationMs.min": 50, "fault.durationMs.max": 80},
            max_fields_changed=2,
        )
    )

    assert candidate.config["attackType"] == "masquerade_fault"
    assert candidate.attack_key == "masquerade_fault"


def test_the_same_pinned_request_compiles_to_the_same_candidate():
    first = compile_attack_candidate(_intent(pins={"fault.prob": 0.42}))
    second = compile_attack_candidate(_intent(pins={"fault.prob": 0.42}))

    assert first.config == second.config
    assert first.diff == second.diff


# --------------------------------------------------------------------------- #
# A tool do IntentAgent                                                       #
# --------------------------------------------------------------------------- #
def test_the_tool_turns_the_dictated_object_into_target_values():
    from adversarial_ids.agents.intent.tools import submit_intent_spec

    payload = json.loads(
        submit_intent_spec.entrypoint(
            objective="evade_detection",
            base_attack="masquerade_fault",
            desired_effect="lower_recall",
            target_values={"fault.durationMs.max": 80, "fault.durationMs.min": 50},
            max_fields_changed=2,
        )
    )

    assert payload["restrictions"]["target_values"] == [
        {"path": "fault.durationMs.max", "value": 80},
        {"path": "fault.durationMs.min", "value": 50},
    ]


def test_the_tool_orders_the_pins_so_the_same_request_gives_the_same_artifact():
    """Um dict vindo de JSON preserva a ordem em que a LLM escreveu as chaves,
    que varia entre chamadas idênticas — o artefato persistido não pode."""

    from adversarial_ids.agents.intent.tools import submit_intent_spec

    def _call(pins):
        return json.loads(
            submit_intent_spec.entrypoint(
                objective="evade_detection",
                base_attack="masquerade_fault",
                desired_effect="lower_recall",
                target_values=pins,
                max_fields_changed=2,
            )
        )["restrictions"]["target_values"]

    assert _call({"cbStatus": 0, "fault.prob": 0.42}) == _call(
        {"fault.prob": 0.42, "cbStatus": 0}
    )


def test_a_prompt_without_dictated_values_still_produces_an_empty_tuple():
    from adversarial_ids.agents.intent.tools import submit_intent_spec

    payload = json.loads(
        submit_intent_spec.entrypoint(
            objective="evade_detection",
            base_attack="masquerade_fault",
            desired_effect="lower_recall",
        )
    )

    assert payload["restrictions"]["target_values"] == []


# --------------------------------------------------------------------------- #
# Escada de feedback (E10) com campos fixados                                 #
# --------------------------------------------------------------------------- #
def test_the_escalation_ceiling_counts_the_pinned_fields():
    """O teto não pode ficar abaixo do ``max_fields_changed`` que a rodada já
    usa, ou a escada derivaria uma intenção que o próprio portão recusa."""

    from adversarial_ids.core.feedback_policy import _fields_ceiling

    intent = _intent(
        pins={"cbStatus": 0},
        allowed=("cbStatus",),
        effect=DesiredEffect.INCREASE_ATTACK_ACTIVITY,
        max_fields_changed=1,
    )

    assert _fields_ceiling(intent) >= intent.restrictions.max_fields_changed


def test_an_escalated_intent_with_pins_still_compiles():
    """A rodada 2 nasce da política, não de uma nova chamada de LLM — se ela
    escalar uma intenção com valores fixados, o resultado ainda tem de compilar."""

    from adversarial_ids.core.feedback_policy import _escalate

    intent = _intent(
        pins={"fault.durationMs.max": 120}, max_fields_changed=2, intensity=IntentIntensity.MEDIUM
    )

    escalated, _reason = _escalate(intent)

    assert escalated is not None
    assert escalated.intensity is IntentIntensity.HIGH
    # O valor fixado atravessa a escalada intacto: a intensidade não o move.
    assert compile_attack_candidate(escalated).config["fault"]["durationMs"]["max"] == 120


# --------------------------------------------------------------------------- #
# Ponta a ponta — o valor pedido chega ao artefato em disco                    #
# --------------------------------------------------------------------------- #
def test_a_dictated_value_reaches_the_attack_candidate_on_disk(tmp_path):
    """Os sete estágios em modo cacheado, com o valor ditado atravessando o
    pipeline inteiro até o JSON que o ERENO leria."""

    from adversarial_ids.domain.loop_record import LoopStageStatus
    from adversarial_ids.shared.json_io import load_json
    from tests.test_intent_loop_orchestrator import _StubIntentAgent, _orchestrator

    intent = _intent(pins={"fault.durationMs.max": 120}, max_fields_changed=3)
    record = _orchestrator(tmp_path, _StubIntentAgent(intent)).run(PROMPT)

    assert [stage.status for stage in record.stages] == [
        LoopStageStatus.SUCCEEDED
    ] * 7

    run_dir = tmp_path / "artifacts" / record.run_id
    saved_intent = load_json(run_dir / "intent.json")
    assert saved_intent["restrictions"]["target_values"] == [
        {"path": "fault.durationMs.max", "value": 120}
    ]

    candidate = load_json(run_dir / "attack_candidate.json")
    assert candidate["config"]["fault"]["durationMs"]["max"] == 120
    assert "valor fixado no pedido" in candidate["rationale"]


def test_a_contradictory_dictated_value_fails_the_generator_stage(tmp_path):
    """O pedido impossível para no estágio que o compilou, com a causa
    registrada — não vira um ataque ajustado em silêncio."""

    from adversarial_ids.domain.loop_record import LoopStageStatus
    from tests.test_intent_loop_orchestrator import _StubIntentAgent, _orchestrator

    intent = _intent(
        pins={"fault.durationMs.min": 900},
        allowed=("fault.durationMs.min",),
        max_fields_changed=1,
    )
    record = _orchestrator(tmp_path, _StubIntentAgent(intent)).run(PROMPT)

    assert [stage.name for stage in record.stages] == ["intent", "compiler"]
    assert record.stages[-1].status is LoopStageStatus.FAILED
    assert "não forma um intervalo" in record.stages[-1].error
