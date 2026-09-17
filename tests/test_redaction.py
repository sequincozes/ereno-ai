"""Testes da redação de segredo em erros e eventos (épico E11, linha Operação)."""

from __future__ import annotations

import pytest

from adversarial_ids.shared.redaction import PLACEHOLDER, redact, secret_values


# --------------------------------------------------------------------------- #
# O valor real do ambiente — a defesa forte                                   #
# --------------------------------------------------------------------------- #
def test_the_real_key_is_removed_even_without_a_recognizable_shape():
    """Não depende de formato: depende de a chave estar no ambiente."""

    environ = {"GROQ_API_KEY": "chave-sem-formato-nenhum-123456"}

    cleaned = redact(
        "401 Unauthorized ao chamar a Groq com chave-sem-formato-nenhum-123456",
        environ=environ,
    )

    assert "chave-sem-formato-nenhum-123456" not in cleaned
    assert PLACEHOLDER in cleaned


def test_the_message_around_the_secret_survives():
    """"401 Unauthorized para ***" diagnostica; "***" sozinho não."""

    environ = {"GROQ_API_KEY": "gsk_abcdefghijklmnop"}

    cleaned = redact("401 Unauthorized: gsk_abcdefghijklmnop expirou", environ=environ)

    assert cleaned == f"401 Unauthorized: {PLACEHOLDER} expirou"


@pytest.mark.parametrize(
    "name",
    ["GROQ_API_KEY", "MY_SECRET", "SERVICE_TOKEN", "DB_PASSWORD", "X_CREDENTIAL"],
)
def test_every_secret_looking_variable_name_is_covered(name: str):
    assert secret_values({name: "valor-longo-o-suficiente"}) == (
        "valor-longo-o-suficiente",
    )


def test_a_short_value_is_not_treated_as_a_secret():
    """`DEBUG_TOKEN=1` apagaria todo dígito 1 das mensagens."""

    assert secret_values({"DEBUG_TOKEN": "1"}) == ()
    assert redact("iteração 1 de 3", environ={"DEBUG_TOKEN": "1"}) == "iteração 1 de 3"


def test_a_harmless_variable_is_left_alone():
    environ = {"GENERATOR_MODE": "cached-mode-longo"}

    assert secret_values(environ) == ()
    assert redact("modo cached-mode-longo", environ=environ) == "modo cached-mode-longo"


# --------------------------------------------------------------------------- #
# O formato — para a chave que não está neste ambiente                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "leaked",
    [
        "gsk_AbCdEfGhIjKlMnOpQrSt",
        "sk-proj-AbCdEfGhIjKlMnOpQrSt",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghij",
    ],
)
def test_known_key_shapes_are_removed_without_being_in_the_environment(leaked: str):
    cleaned = redact(f"falhou com {leaked}", environ={})

    assert PLACEHOLDER in cleaned
    for fragment in ("gsk_AbCdEf", "sk-proj-AbCdEf", "eyJhbGciOiJIUzI1NiJ9"):
        assert fragment not in cleaned


def test_nothing_to_redact_is_returned_unchanged():
    assert redact("estágio ereno falhou: jar ausente", environ={}) == (
        "estágio ereno falhou: jar ausente"
    )


@pytest.mark.parametrize("empty", [None, ""])
def test_absent_text_stays_absent(empty):
    assert redact(empty, environ={}) == empty


# --------------------------------------------------------------------------- #
# O ponto de emissão                                                          #
# --------------------------------------------------------------------------- #
def test_the_orchestrator_redacts_a_stage_error_before_it_reaches_disk(
    tmp_path, monkeypatch
):
    """O caminho real pelo qual uma chave vazaria: `str(exc)` virando artefato."""

    from tests.test_intent_loop_orchestrator import _intent, _orchestrator, _StubIntentAgent

    monkeypatch.setenv("GROQ_API_KEY", "gsk_chave_de_teste_1234567890")

    class _LeakyIntentAgent:
        def interpret(self, prompt: str):
            raise RuntimeError(
                "401 ao POST /openai/v1/chat "
                "(Authorization: Bearer gsk_chave_de_teste_1234567890)"
            )

    events: list = []
    orchestrator = _orchestrator(
        tmp_path, _LeakyIntentAgent(), event_sink=events.append
    )

    record = orchestrator.run("Reduza o recall.")

    failed = record.stages[0]
    assert failed.status.value == "failed"
    assert "gsk_chave_de_teste_1234567890" not in (failed.error or "")
    assert PLACEHOLDER in (failed.error or "")
    # E a mesma coisa na timeline, que é outro arquivo em disco.
    messages = " ".join(event.message or "" for event in events)
    assert "gsk_chave_de_teste_1234567890" not in messages

    # O resto da mensagem sobrevive: a falha continua diagnosticável.
    assert "401" in (failed.error or "")

    # Usa o stub só para provar que a fixture de sucesso continua intacta.
    assert _StubIntentAgent(_intent()).interpret("x") is not None
