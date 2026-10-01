"""Registro de geradores — o backend que produz o trace, resolvido por chave.

Espelha ``tests/test_detectors.py`` (épico E8) a montante: um registro, um
protocolo para a superfície consumida, e um manifest por execução. O que estes
testes fixam:

- registro e vocabulário do domínio descrevem o mesmo conjunto (a checagem que
  roda na importação), e o orquestrador aceita exatamente essas chaves;
- construir é validado: chave desconhecida e opção faltando falham com o nome
  do que falta, não com ``KeyError`` no meio de uma campanha;
- as capacidades do manifest vêm do **registro**, não de quem o grava — é o que
  impede um manifest de afirmar que o modo cacheado varia com a config, a
  afirmação que faria um platô de campanha passar por resultado experimental.
"""

from __future__ import annotations

import pytest

from adversarial_ids.agents.orchestrator.intent_loop import _VALID_GENERATOR_MODES
from adversarial_ids.core.generator_runner import GeneratorRunner
from adversarial_ids.core.generators import (
    DEFAULT_GENERATOR_KEY,
    GeneratorError,
    GeneratorLike,
    GeneratorRequest,
    build_generator,
    generator_spec,
    list_generator_keys,
    manifest_for,
)
from adversarial_ids.domain.generator_manifest import GENERATOR_KEYS, GeneratorManifest


def _options(tmp_path, **overrides):
    options = {
        "runtime_dir": tmp_path,
        "output_dataset_path": tmp_path / "out.csv",
        "run_command": ["echo", "noop"],
        "suggested_config_path": str(tmp_path / "attack_config.json"),
        "action_config_relative_path": None,
        "benign_action_config_relative_path": None,
        "benign_seed_path": None,
        "cached_dataset_path": tmp_path / "baseline.csv",
        "timeout_seconds": None,
        "max_retries": 0,
        "retry_backoff_seconds": 2.0,
    }
    options.update(overrides)
    return options


def _request(tmp_path, **overrides):
    return GeneratorRequest(
        attack_key="masquerade_fault",
        segment_name="uc03_masquerade_fault",
        options=_options(tmp_path),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Registro e vocabulário                                                      #
# --------------------------------------------------------------------------- #
def test_the_registry_and_the_domain_vocabulary_agree():
    assert list_generator_keys() == GENERATOR_KEYS


def test_the_orchestrator_accepts_exactly_the_registered_keys():
    """Se divergirem, ``--generator-mode`` aceita uma chave que não constrói
    nada, ou recusa uma que constrói."""

    assert tuple(_VALID_GENERATOR_MODES) == GENERATOR_KEYS


def test_the_default_generator_needs_no_java():
    """O default precisa rodar no CI e na máquina de quem não tem o JAR."""

    assert generator_spec(DEFAULT_GENERATOR_KEY).requires_java is False


@pytest.mark.parametrize("key", GENERATOR_KEYS)
def test_every_registered_generator_describes_itself(key):
    spec = generator_spec(key)
    assert spec.key == key
    assert spec.simulator
    assert spec.description
    assert spec.required_options


def test_an_unknown_generator_lists_the_registered_ones():
    with pytest.raises(GeneratorError, match="Gerador desconhecido"):
        generator_spec("mininet")


# --------------------------------------------------------------------------- #
# Construção                                                                  #
# --------------------------------------------------------------------------- #
def test_the_cached_backend_serves_a_versioned_trace(tmp_path):
    generator = build_generator("cached", _request(tmp_path))

    assert isinstance(generator, GeneratorRunner)
    assert generator.is_cached is True


def test_the_jar_backend_runs_the_simulator(tmp_path):
    generator = build_generator("jar", _request(tmp_path))

    assert generator.is_cached is False


def test_the_seed_reaches_the_backend(tmp_path):
    """Fase 2.0: sem a semente chegando ao gerador, réplica e repetição são a
    mesma coisa."""

    generator = build_generator("jar", _request(tmp_path, random_seed=4242))

    assert generator.random_seed == 4242


def test_a_missing_option_fails_with_its_name(tmp_path):
    options = _options(tmp_path)
    del options["run_command"]
    request = GeneratorRequest(attack_key="masquerade_fault", options=options)

    with pytest.raises(GeneratorError, match="run_command"):
        build_generator("jar", request)


def test_the_runner_satisfies_the_consumed_protocol(tmp_path):
    """O protocolo é a lista fechada do que o orquestrador chama — se o runner
    deixasse de cumpri-la, um backend novo não teria contrato para imitar."""

    assert isinstance(build_generator("cached", _request(tmp_path)), GeneratorLike)


# --------------------------------------------------------------------------- #
# Manifest                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", GENERATOR_KEYS)
def test_the_manifest_copies_the_capabilities_from_the_registry(key):
    spec = generator_spec(key)
    manifest = manifest_for(key, attack_key="flooding")

    assert manifest.simulator == spec.simulator
    assert manifest.requires_java == spec.requires_java
    assert manifest.produces_per_config_variation == spec.produces_per_config_variation


def test_only_the_jar_backend_claims_per_config_variation():
    """O campo que separa 'medi o efeito da intenção' de 'exercitei o fluxo'."""

    assert manifest_for("cached", attack_key="flooding").produces_per_config_variation is False
    assert manifest_for("jar", attack_key="flooding").produces_per_config_variation is True


def test_a_manifest_cannot_lie_about_the_generator_capabilities():
    with pytest.raises(ValueError, match="produces_per_config_variation"):
        GeneratorManifest(
            generator="cached",
            simulator="none",
            requires_java=False,
            produces_per_config_variation=True,  # mentira: cached serve o mesmo arquivo
            attack_key="flooding",
            duration_seconds=1.0,
        )


def test_a_manifest_cannot_rename_the_simulator():
    with pytest.raises(ValueError, match="simulator"):
        GeneratorManifest(
            generator="cached",
            simulator="ereno",  # nenhum simulador rodou
            requires_java=False,
            produces_per_config_variation=False,
            attack_key="flooding",
            duration_seconds=1.0,
        )


def test_the_manifest_records_the_seed_that_was_passed():
    manifest = manifest_for("jar", attack_key="flooding", random_seed=4242)

    assert manifest.random_seed == 4242
    assert manifest.schema_version == 1
