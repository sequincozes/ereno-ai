"""Testes do Injector: intent/model/data-driven atrás de um contrato só.

Cobre tudo que não é o LLM — e é de propósito que isso é quase tudo: ler o
creator Java, perfilar o trace, montar as instruções e o manifest, o portão que
confere se a saída é um ataque novo na DSL do programmable, a gravação dos
artefatos pela rodada e a recusa de retomar com outra estratégia. Sem Groq e
sem Java.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from adversarial_ids.agents.injector import (
    InjectorError,
    InjectorOutputError,
    enforce_injector_output,
    injector_spec,
    list_injector_keys,
    prepare_injector,
)
from adversarial_ids.agents.injector.java_sources import (
    JAVA_CREATOR_FILES,
    JavaSourceError,
    list_source_attacks,
    load_creator_source,
)
from adversarial_ids.agents.orchestrator.campaign_resume import CampaignResumeError
from adversarial_ids.config.attacks_registry import list_attack_keys
from adversarial_ids.core.trace_profile import (
    TraceProfileError,
    profile_trace,
    render_trace_profile,
)
from adversarial_ids.domain.injector_manifest import (
    INJECTOR_KEYS,
    InjectorManifest,
)
from adversarial_ids.interfaces.cli import run_cli
from adversarial_ids.shared.json_io import load_json
from tests.fakes.fake_intent_agent import FakeIntentAgent
from tests.test_intent_loop_orchestrator import (
    _FailingIntentAgent,
    _intent,
    _orchestrator,
    _StubIntentAgent,
)

_CREATOR_BODY = """package br.ufu.facom.ereno.attacks.uc02.creator;

import java.util.ArrayList;
import java.util.logging.Logger;


/** Replays a block of legitimate messages in reverse order. */
public class InverseReplayCreatorC {


    int blockLen = 3;
}
"""


def _ereno_checkout(root: Path, attack_key: str = "inverse_replay") -> Path:
    path = root / "src/main/java/br/ufu/facom/ereno/attacks" / JAVA_CREATOR_FILES[attack_key]
    path.parent.mkdir(parents=True)
    path.write_text(_CREATOR_BODY, encoding="utf-8")
    return root


def _trace(path: Path, *, extra_class: str | None = None) -> Path:
    rows = ["timestampDiff,stDiff,confRev,constantOnBoth,class"]
    for i in range(20):
        rows.append(f"{0.7 + (i % 4) * 0.01},{i % 2},1,5,normal")
    for i in range(20):
        rows.append(f"{0.002 + (i % 4) * 0.001},{i % 2},2,5,inverse_replay")
    if extra_class:
        rows += [f"0.5,0,1,5,{extra_class}"] * 3
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _programmable(**overrides):
    payload = {
        "objective": "evade_detection",
        "base_attack": "programmable",
        "desired_effect": "lower_recall",
        "target_values": {
            "rules.r0.op": "add",
            "rules.r0.field": "timestamp",
            "rules.r0.value": 0.02,
        },
        "max_fields_changed": 3,
    }
    payload.update(overrides)
    return FakeIntentAgent.compile(payload, "proponha um ataque novo")


# --------------------------------------------------------------------------- #
# Registro                                                                    #
# --------------------------------------------------------------------------- #
def test_registry_matches_the_domain_vocabulary():
    assert list_injector_keys() == INJECTOR_KEYS == ("intent", "model", "data")


def test_unknown_injector_is_an_actionable_error():
    with pytest.raises(InjectorError, match="Registrados"):
        injector_spec("genetic")


def test_every_source_attack_is_a_registered_attack_and_programmable_is_not_one():
    # Ler o interpretador da DSL para escrever na própria DSL é circular.
    assert set(list_source_attacks()) <= set(list_attack_keys())
    assert "programmable" not in list_source_attacks()


@pytest.mark.parametrize(
    ("key", "options", "message"),
    [
        ("model", {}, "source_attack"),
        ("data", {}, "dataset_path"),
        ("intent", {"source_attack": "inverse_replay"}, "não usa"),
        ("model", {"source_attack": "inverse_replay", "dataset_path": "x.csv"}, "não usa"),
        ("intent", {"attack_label": "x"}, "não usa"),
    ],
)
def test_prepare_refuses_missing_or_foreign_inputs(key, options, message):
    with pytest.raises(InjectorError, match=message):
        prepare_injector(key, **options)


def test_prepare_intent_needs_no_context():
    prepared = prepare_injector("intent")
    assert prepared.instructions is None
    assert prepared.manifest.injector == "intent"
    assert prepared.manifest.base_attack_constraint is None


# --------------------------------------------------------------------------- #
# Model-driven                                                                #
# --------------------------------------------------------------------------- #
def test_creator_source_drops_package_and_imports_and_hashes_the_original(tmp_path):
    root = _ereno_checkout(tmp_path)
    source = load_creator_source("inverse_replay", root)

    raw = (root / "src/main/java/br/ufu/facom/ereno/attacks" / JAVA_CREATOR_FILES["inverse_replay"]).read_bytes()
    assert source.sha256 == hashlib.sha256(raw).hexdigest()
    assert "package " not in source.text and "import " not in source.text
    assert "\n\n\n" not in source.text
    assert "Replays a block" in source.text


def test_missing_creator_points_at_the_source_dir_setting(tmp_path):
    with pytest.raises(JavaSourceError, match="ERENO_SOURCE_DIR"):
        load_creator_source("inverse_replay", tmp_path)


def test_prepare_model_reads_the_creator_and_only_the_programmable_catalog(tmp_path):
    root = _ereno_checkout(tmp_path)
    prepared = prepare_injector("model", source_attack="inverse_replay", source_dir=root)

    text = prepared.instructions
    assert "class InverseReplayCreatorC" in text
    assert "### `programmable`" in text
    # Só o programmable: o catálogo inteiro estouraria o TPM e convidaria o
    # modelo a devolver um ataque que o portão recusa.
    assert "### `masquerade_fault`" not in text
    assert "### `inverse_replay`" not in text

    manifest = prepared.manifest
    assert manifest.injector == "model"
    assert manifest.base_attack_constraint == "programmable"
    assert manifest.subject == "inverse_replay"
    assert manifest.sources[0].ref.endswith("InverseReplayCreatorC.java")
    assert manifest.context_chars == len(text)


# --------------------------------------------------------------------------- #
# Data-driven                                                                 #
# --------------------------------------------------------------------------- #
def test_profile_ranks_the_separating_column_first(tmp_path):
    profile = profile_trace(_trace(tmp_path / "t.csv"))

    assert profile.attack_label == "inverse_replay"
    assert profile.class_counts == {"normal": 20, "inverse_replay": 20}
    assert profile.top_features[0].feature == "timestampDiff"
    # stDiff idêntico nas duas classes: separação zero, mas medida.
    st = next(c for c in profile.top_features if c.feature == "stDiff")
    assert st.separation == 0.0


def test_constant_columns_split_by_presence_not_infinity(tmp_path):
    profile = profile_trace(_trace(tmp_path / "t.csv"))

    shifted = {c.feature for c in profile.constant_shift}
    assert shifted == {"confRev"}  # 1 vs 2, sem desvio
    # Constante e igual nos dois lados: nada a separar, e não é listado.
    assert "constantOnBoth" not in shifted
    assert "constantOnBoth" not in {c.feature for c in profile.top_features}


def test_lever_features_are_always_shown(tmp_path):
    profile = profile_trace(_trace(tmp_path / "t.csv"), top_k=1)

    assert [c.feature for c in profile.top_features] == ["timestampDiff"]
    assert {c.feature for c in profile.lever_features} == {"confRev", "stDiff", "timestampDiff"}


def test_profile_needs_an_unambiguous_attack_class(tmp_path):
    trace = _trace(tmp_path / "t.csv", extra_class="flooding")
    with pytest.raises(TraceProfileError, match="Informe a classe"):
        profile_trace(trace)
    assert profile_trace(trace, attack_label="flooding").attack_label == "flooding"
    with pytest.raises(TraceProfileError, match="não está no trace"):
        profile_trace(trace, attack_label="grayhole")


def test_profile_needs_a_normal_class(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,class\n1,x\n2,x\n3,y\n", encoding="utf-8")
    with pytest.raises(TraceProfileError, match="normal"):
        profile_trace(path)


def test_render_is_deterministic_and_names_the_classes(tmp_path):
    profile = profile_trace(_trace(tmp_path / "t.csv"))
    text = render_trace_profile(profile)

    assert text == render_trace_profile(profile)
    assert "`inverse_replay` contra `normal`" in text
    assert "| `timestampDiff` |" in text


def test_prepare_data_persists_the_profile_next_to_the_manifest(tmp_path):
    trace = _trace(tmp_path / "t.csv")
    prepared = prepare_injector("data", dataset_path=trace)

    assert prepared.manifest.subject == "inverse_replay"
    assert prepared.manifest.sources[0].sha256 == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert prepared.extra_artifacts["trace_profile.json"]["attack_label"] == "inverse_replay"
    assert "Perfil do trace observado" in prepared.instructions


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
def test_manifest_refuses_a_model_injector_outside_the_dsl():
    with pytest.raises(ValidationError, match="base_attack_constraint"):
        InjectorManifest(
            injector="model",
            context_kind="java_source",
            base_attack_constraint=None,
            subject="inverse_replay",
            sources=({"ref": "x.java", "sha256": "0" * 64},),
        )


def test_manifest_refuses_a_contextual_injector_without_context():
    with pytest.raises(ValidationError, match="ao menos uma fonte"):
        InjectorManifest(
            injector="data",
            context_kind="trace_profile",
            base_attack_constraint="programmable",
        )


# --------------------------------------------------------------------------- #
# Portão de saída                                                             #
# --------------------------------------------------------------------------- #
def test_gate_lets_the_intent_injector_choose_any_attack():
    intent = _intent()
    assert enforce_injector_output(intent, "intent") is intent


@pytest.mark.parametrize("injector", ["model", "data"])
def test_gate_accepts_an_authored_programmable_attack(injector):
    intent = _programmable()
    assert enforce_injector_output(intent, injector) is intent


@pytest.mark.parametrize("injector", ["model", "data"])
def test_gate_refuses_a_catalogue_attack(injector):
    with pytest.raises(InjectorOutputError, match="masquerade_fault"):
        enforce_injector_output(_intent(), injector)


@pytest.mark.parametrize("injector", ["model", "data"])
def test_gate_refuses_a_programmable_without_authored_rules(injector):
    # Sem regra autorada o que roda é o programmable padrão — não um ataque novo.
    intent = _programmable(target_values=None)
    with pytest.raises(InjectorOutputError, match="nenhuma regra"):
        enforce_injector_output(intent, injector)


# --------------------------------------------------------------------------- #
# Orquestrador                                                                #
# --------------------------------------------------------------------------- #
class _DeclaredInjector(_StubIntentAgent):
    """Stub que se declara um injetor, como os agentes reais fazem."""

    def __init__(self, intent, manifest: InjectorManifest, extra=None) -> None:
        super().__init__(intent)
        self._manifest = manifest
        self._extra = extra or {}

    def injector_artifacts(self):
        return {"injector_manifest.json": self._manifest.model_dump(mode="json"), **self._extra}


class _DeclaredFailingInjector(_FailingIntentAgent):
    def __init__(self, manifest: InjectorManifest) -> None:
        self._manifest = manifest

    def injector_artifacts(self):
        return {"injector_manifest.json": self._manifest.model_dump(mode="json")}


def _manifest(injector: str = "data") -> InjectorManifest:
    if injector == "intent":
        return InjectorManifest(injector="intent", context_kind="catalog")
    return InjectorManifest(
        injector=injector,
        context_kind="trace_profile" if injector == "data" else "java_source",
        base_attack_constraint="programmable",
        subject="inverse_replay",
        sources=({"ref": "t.csv", "sha256": "a" * 64},),
    )


def test_round_that_calls_the_llm_writes_the_injector_artifacts(tmp_path):
    agent = _DeclaredInjector(_intent(), _manifest(), extra={"trace_profile.json": {"k": 1}})
    orchestrator = _orchestrator(tmp_path, agent)
    assert orchestrator.injector == "data"

    record = orchestrator.run("prompt")
    run_dir = tmp_path / "artifacts" / record.run_id

    assert InjectorManifest.model_validate(load_json(run_dir / "injector_manifest.json")) == _manifest()
    assert load_json(run_dir / "trace_profile.json") == {"k": 1}


def test_injector_artifacts_are_written_even_when_the_llm_call_fails(tmp_path):
    orchestrator = _orchestrator(tmp_path, _DeclaredFailingInjector(_manifest()))
    record = orchestrator.run("prompt")

    assert (tmp_path / "artifacts" / record.run_id / "injector_manifest.json").exists()


def test_an_undeclared_agent_gets_no_manifest(tmp_path):
    # O orquestrador não afirma sobre um agente o que ele não disse.
    orchestrator = _orchestrator(tmp_path, _StubIntentAgent(_intent()))
    record = orchestrator.run("prompt")

    assert orchestrator.injector is None
    assert not (tmp_path / "artifacts" / record.run_id / "injector_manifest.json").exists()


def test_inherited_rounds_do_not_rewrite_the_manifest(tmp_path):
    agent = _DeclaredInjector(_intent(), _manifest())
    records = _orchestrator(tmp_path, agent, feedback_min_delta=0.0).run_campaign("p", rounds=2)

    assert len(records) == 2
    assert (tmp_path / "artifacts" / records[0].run_id / "injector_manifest.json").exists()
    assert not (tmp_path / "artifacts" / records[1].run_id / "injector_manifest.json").exists()


def test_resume_that_calls_the_llm_refuses_another_injector(tmp_path):
    # A intenção da rodada 1 falhou: refazê-la chama o LLM de novo, e aí a
    # estratégia que propõe o ataque precisa ser a mesma.
    first = _orchestrator(tmp_path, _DeclaredFailingInjector(_manifest("model"))).run("p")

    other = _orchestrator(tmp_path, _DeclaredInjector(_intent(), _manifest("intent")))
    with pytest.raises(CampaignResumeError, match="'model'"):
        other.resume_campaign(first.run_id, rounds=1)


def test_resume_with_the_same_injector_retries_the_round(tmp_path):
    first = _orchestrator(tmp_path, _DeclaredFailingInjector(_manifest("model"))).run("p")

    same = _orchestrator(tmp_path, _DeclaredInjector(_intent(), _manifest("model")))
    (retry,) = same.resume_campaign(first.run_id, rounds=1)
    assert retry.retry_of == first.run_id


def test_resume_that_inherits_the_intent_ignores_the_injector(tmp_path):
    # A rodada 1 tem intenção (falhou depois, no gate do E4): a retomada a
    # reaproveita sem LLM, então trocar de injetor não muda nada no experimento.
    first = _orchestrator(
        tmp_path, _DeclaredInjector(_intent(), _manifest("model")), min_attack_rows=10_000
    ).run("p")

    other = _orchestrator(tmp_path, _DeclaredInjector(_intent(), _manifest("intent")))
    (retry,) = other.resume_campaign(first.run_id, rounds=1)
    assert retry.retry_of == first.run_id


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _cli(argv, **kwargs):
    from io import StringIO

    stdout, stderr = StringIO(), StringIO()
    code = run_cli(argv=argv, stdout=stdout, stderr=stderr, **kwargs)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--engine", "live", "--injector", "model"], "--engine intent"),
        (["--engine", "intent", "--prompt", "p", "--injector", "model"], "--injector-source"),
        (["--engine", "intent", "--prompt", "p", "--injector", "data"], "--injector-dataset"),
        (["--engine", "intent", "--prompt", "p", "--injector-source", "grayhole"], "--injector model"),
        (
            ["--engine", "intent", "--prompt", "p", "--injector", "model",
             "--injector-source", "grayhole", "--injector-label", "x"],
            "--injector data",
        ),
    ],
)
def test_cli_refuses_inconsistent_injector_flags(argv, message):
    code, _, stderr = _cli(argv, run_intent_loop=lambda **_: pytest.fail("não deveria rodar"))
    assert code == 2
    assert message in stderr


def test_cli_forwards_the_model_injector_and_its_source():
    received = {}

    def fake(**kwargs):
        received.update(kwargs)
        return ()

    code, _, _ = _cli(
        ["--engine", "intent", "--prompt", "p", "--injector", "model",
         "--injector-source", "inverse_replay"],
        run_intent_loop=fake,
    )

    assert code == 0
    assert received["injector"] == "model"
    assert received["injector_options"] == {"source_attack": "inverse_replay"}


def test_cli_forwards_the_data_injector_with_its_label():
    received = {}

    def fake(**kwargs):
        received.update(kwargs)
        return ()

    _cli(
        ["--engine", "intent", "--prompt", "p", "--injector", "data",
         "--injector-dataset", "t.csv", "--injector-label", "flooding"],
        run_intent_loop=fake,
    )

    assert received["injector_options"] == {"dataset_path": "t.csv", "attack_label": "flooding"}
