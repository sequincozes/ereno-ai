"""Cobertura da injeção do catálogo no prompt do IntentAgent.

Garante que o texto que a LLM lê (``build_intent_instructions``) nunca diverge
do catálogo real (``ATTACK_CAPABILITY_CATALOG``) que o portão determinístico
valida — mesma garantia que ``tests/test_defender_prompt.py`` dá do lado do
Defensor via ``select_evidence_candidates``.
"""

from __future__ import annotations

from adversarial_ids.agents.intent.catalog_context import (
    build_capability_context,
    build_intent_instructions,
)
from adversarial_ids.config.attack_capabilities import ATTACK_CAPABILITY_CATALOG
from adversarial_ids.config.attacks_registry import list_attack_keys


def test_every_registered_attack_key_appears_in_the_context():
    context = build_capability_context()
    for attack_key in list_attack_keys():
        assert f"`{attack_key}`" in context


def test_every_capability_id_appears_in_the_context():
    context = build_capability_context()
    for capability in ATTACK_CAPABILITY_CATALOG.values():
        assert capability.capability_id in context


def test_every_catalogued_field_path_appears_in_the_context():
    context = build_capability_context()
    for capability in ATTACK_CAPABILITY_CATALOG.values():
        for field in capability.fields:
            assert f"`{field.path}`" in context


def test_the_stale_single_attack_claim_is_gone_from_the_base_prompt():
    """Regressão: o prompt não pode mais afirmar que só o masquerade_fault
    tem capacidade intent-driven, agora que 11 ataques têm."""
    instructions = build_intent_instructions()
    assert "apenas `masquerade_fault`" not in instructions
    assert "Nesta versão, apenas" not in instructions


def test_instructions_append_the_catalog_after_the_base_prompt():
    instructions = build_intent_instructions()
    context = build_capability_context()

    assert instructions.endswith(context)
    assert "## Catálogo de capacidades" in instructions


def test_capability_context_is_stable_across_calls():
    assert build_capability_context() == build_capability_context()
