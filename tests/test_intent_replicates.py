"""Modo de réplicas: varredura de seeds sob a mesma intenção (Fase 2.R).

A 2.0 tornou cada seed reprodutível; a 2.R roda N seeds do mesmo experimento
para reportar a métrica-objetivo como média ± desvio. O invariante central: o
LLM é chamado **uma vez** para o lote inteiro (só a geração varia entre
réplicas), e cada réplica é um ``LoopRecord`` próprio com sua seed e um
``replicate_batch_id`` comum.

Stubs e seed cacheada, sem Groq/Java — como os demais testes do orquestrador.
Em modo cacheado o dataset é o mesmo, então as réplicas não divergem de fato; o
que se testa aqui é o encanamento (1 chamada de LLM, seeds gravadas, lote
ligado, estatística agregada), não a variância física — essa só aparece com o
JAR.
"""

from __future__ import annotations

import pytest

from adversarial_ids.agents.orchestrator.intent_loop import ReplicateResult
from adversarial_ids.domain.loop_record import LoopStageStatus
from adversarial_ids.domain.run_usage import AgentUsage
from adversarial_ids.shared.loop_record_store import load_loop_records
from tests.test_intent_loop_orchestrator import (
    _StubIntentAgent,
    _FailingIntentAgent,
    _intent,
    _orchestrator,
)

PROMPT = "Reduza o recall."


class _CountingIntentAgent(_StubIntentAgent):
    """Informa consumo, para contar as chamadas de LLM do lote."""

    def __init__(self, intent, usage: AgentUsage) -> None:
        super().__init__(intent)
        self.last_usage = usage


def test_run_replicates_rejects_an_empty_seed_list(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))
    with pytest.raises(ValueError, match="ao menos uma seed"):
        orch.run_replicates(PROMPT, seeds=[])


def test_run_replicates_rejects_duplicate_seeds(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))
    with pytest.raises(ValueError, match="distintas"):
        orch.run_replicates(PROMPT, seeds=[42, 42])


def test_each_replicate_records_its_own_seed(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[42, 43, 44])

    assert isinstance(result, ReplicateResult)
    assert result.n_total == 3
    assert [r.seed for r in result.records] == [42, 43, 44]


def test_all_replicates_share_one_batch_id(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[1, 2, 3])

    ids = {r.replicate_batch_id for r in result.records}
    assert ids == {result.batch_id}
    assert result.batch_id is not None


def test_the_llm_is_called_once_for_the_whole_batch(tmp_path):
    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    orch = _orchestrator(tmp_path, agent)

    result = orch.run_replicates(PROMPT, seeds=[10, 11, 12, 13])

    # Uma interpretação para as quatro réplicas — não uma por seed.
    assert agent.received_prompts == [PROMPT]
    assert result.llm_calls == 1


def test_only_the_first_replicate_is_charged_the_intent_tokens(tmp_path):
    agent = _CountingIntentAgent(_intent(), AgentUsage(input_tokens=600))
    orch = _orchestrator(tmp_path, agent)

    result = orch.run_replicates(PROMPT, seeds=[7, 8, 9])

    # A primeira réplica chamou o LLM (600); as demais reusaram a intenção e não
    # recontabilizam o consumo (só o defender stub, que não informa nada).
    assert result.records[0].total_tokens == 600
    assert all(r.total_tokens is None for r in result.records[1:])


def test_the_intent_stage_is_reused_not_recalled_after_the_first(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[42, 43])

    first_intent = result.records[0].stages[0]
    second_intent = result.records[1].stages[0]
    assert first_intent.status is LoopStageStatus.SUCCEEDED
    assert second_intent.status is LoopStageStatus.SKIPPED
    assert "reusada" in second_intent.error


def test_every_replicate_is_persisted_in_the_ledger(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[1, 2, 3])

    persisted = load_loop_records(tmp_path / "loop_records.json")
    assert [r.run_id for r in persisted] == [r.run_id for r in result.records]


def test_the_summary_aggregates_the_objective_metric(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[42, 43, 44])

    # Em cached o recall costuma sair 1.0 nas três; o que importa é que o resumo
    # coletou uma métrica por réplica que chegou ao FEEDBACK.
    assert len(result.objective_values) == result.n_total
    assert result.mean is not None
    assert result.objective_metric in ("recall", "f1")


def test_stdev_needs_at_least_two_measured_replicates(tmp_path):
    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    one = orch.run_replicates(PROMPT, seeds=[42])
    assert one.stdev is None  # uma medida só não tem desvio amostral

    many = orch.run_replicates(PROMPT, seeds=[42, 43])
    assert many.stdev is not None


def test_a_replicate_batch_is_not_a_campaign(tmp_path):
    """Réplica é uma rodada repetida sob outra seed — round=1, sem pai."""

    orch = _orchestrator(tmp_path, _StubIntentAgent(_intent()))

    result = orch.run_replicates(PROMPT, seeds=[42, 43, 44])

    assert all(r.round == 1 for r in result.records)
    assert all(r.parent_run_id is None for r in result.records)


def test_a_failed_replicate_stays_in_the_batch_but_out_of_the_stats(tmp_path):
    # Um seed com poucas linhas de ataque reprova o gate do E4 no preprocess.
    from tests.test_intent_loop_orchestrator import _tiny_seed

    thin = _tiny_seed(tmp_path / "thin.csv", attack_rows=1, normal_rows=10)
    orch = _orchestrator(
        tmp_path, _StubIntentAgent(_intent()), cached_dataset_path=thin
    )

    result = orch.run_replicates(PROMPT, seeds=[42, 43])

    assert result.n_total == 2
    # Nenhuma chegou ao FEEDBACK, então a estatística fica vazia — mas os
    # registros (com a causa) continuam no lote.
    assert result.objective_values == ()
    assert result.mean is None
    assert all(
        any(s.status is LoopStageStatus.FAILED for s in r.stages)
        for r in result.records
    )


def test_the_batch_recovers_the_llm_call_if_the_first_intent_fails(tmp_path):
    """Se o estágio INTENT da primeira réplica falha, a próxima tenta de novo em
    vez de o lote inteiro herdar um None."""

    orch = _orchestrator(tmp_path, _FailingIntentAgent())

    result = orch.run_replicates(PROMPT, seeds=[42, 43])

    # Todas falham no intent (o stub sempre levanta), mas cada uma tentou chamar
    # o LLM — nenhuma reusou uma intenção inexistente.
    assert result.n_total == 2
    assert all(r.stages[0].status is LoopStageStatus.FAILED for r in result.records)
