"""Injetor adaptativo do E1 (B3/B4): o LLM propõe, o pipeline decide.

Mesmo guardrail do pipeline: o LLM só produz a ``IntentSpec`` (via
``submit_intent_spec``, com os dois portões de ``compile_intent``); compilação,
caixa e validade são determinísticas, em ``experiments/search.py``.

B3 e B4 são a mesma classe com o mesmo modelo, temperatura, prompt e
ferramenta; a única diferença é o bloco do plano do Defensor, que o B4 obtém
do ``DefenderAgent`` sobre o ``DetectionReport`` do último candidato válido
(``experiments/defense_evidence.py::candidate_report``).
"""

from __future__ import annotations

import time
from typing import Any, Callable

from agno.agent import Agent
from agno.models.groq import Groq

from adversarial_ids.agents.attacker.context import (
    DAILY,
    MALFORMED,
    MINUTE,
    TRANSIENT,
    build_attacker_instructions,
    build_round_message,
    classify_provider_error,
    render_plan,
)
from adversarial_ids.agents.defender.tools import parse_defense_plan, validate_plan_against_report
from adversarial_ids.agents.intent.agent import IntentAgent, IntentCompilationError, compile_intent
from adversarial_ids.agents.intent.tools import submit_intent_spec
from adversarial_ids.agents.usage import usage_from_response
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.experiments.defense_evidence import candidate_report
from adversarial_ids.experiments.protocol import FrozenIds
from adversarial_ids.experiments.search import (
    VALID,
    Proposal,
    ProposalContext,
    ProposerUnavailable,
    stage_label,
)

ATTACKER_TEMPERATURE = 0.7


class Pacer:
    """Espaça as chamadas ao provedor — atacante e Defensor dividem o TPM."""

    def __init__(self, min_interval_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self.min_interval_s - (time.monotonic() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = time.monotonic()


# Como o agno devolve uma falha de transporte: resposta "COMPLETED", sem
# ferramenta, cujo conteúdo é só a mensagem do erro.
_AGNO_TRANSPORT_ERRORS = {"connection error", "request timed out"}
# Quantas vezes repetir o Defensor quando a Groq recusa o JSON que ele gerou.
DEFENDER_MALFORMED_RETRIES = 2


def _provider_text(response: Any) -> str | None:
    """O texto de erro do provedor numa resposta do agno, ou ``None``.

    Só o conteúdo de uma resposta que **é** um erro: status de erro, ou o
    transporte que o agno embrulha como resposta. Nunca o conteúdo de uma
    resposta normal — um ``DefensePlan`` traz números como 0,9502 e 429
    mensagens, que um marcador de erro confundiria com um 502 ou um 429.
    """

    content = str(getattr(response, "content", "") or "")
    if getattr(response, "status", None) and "error" in str(response.status).lower():
        return content or None
    if not getattr(response, "tools", None):
        stripped = content.strip()
        if stripped.lower().rstrip(".") in _AGNO_TRANSPORT_ERRORS or stripped.startswith('{"error"'):
            return stripped
    return None


def _call(
    agent: Any,
    message: str,
    pacer: Pacer,
    *,
    minute_retries: int,
    minute_wait_s: float,
    malformed_retries: int = 0,
    retries: list[str] | None = None,
):
    """``agent.run`` com o erro do provedor lido antes de qualquer conclusão.

    Devolve ``(response, provider_text)``: quando a chamada não produziu nada
    utilizável e o texto do erro não é limite do provedor, quem chama decide.
    Limite por minuto e falha de rede repetem até ``minute_retries``; JSON
    recusado pelo provedor repete até ``malformed_retries`` e depois volta para
    quem chama (o Defensor registra a recusa). Cota diária, prompt grande
    demais, ou rede/limite que persiste viram ``ProposerUnavailable``.
    ``retries`` (se dado) recebe o tipo de cada repetição.
    """

    minute = malformed = 0
    while True:
        pacer.wait()
        try:
            response = agent.run(message)
            text = _provider_text(response)
        except Exception as error:  # noqa: BLE001 — classificado abaixo
            response, text = None, f"{type(error).__name__}: {error}"
        kind = classify_provider_error(text)
        if kind in (MINUTE, TRANSIENT) and minute < minute_retries:
            minute += 1
            if retries is not None:
                retries.append(kind)
            pacer._sleep(minute_wait_s)
            continue
        if kind == MALFORMED and malformed < malformed_retries:
            malformed += 1
            if retries is not None:
                retries.append(kind)
            continue
        if kind is not None and kind != MALFORMED:
            raise ProposerUnavailable(f"limite do provedor ({kind}): {str(text)[:300]}")
        return response, text


class LlmAttacker:
    """Propositor B3 (métricas) ou B4 (métricas + plano do Defensor)."""

    def __init__(
        self,
        family: str,
        *,
        with_defense: bool,
        model_id: str,
        frozen: FrozenIds | None = None,
        defender_model_id: str | None = None,
        temperature: float = ATTACKER_TEMPERATURE,
        min_interval_s: float = 20.0,
        minute_retries: int = 4,
        minute_wait_s: float = 65.0,
        agent: Any | None = None,
        defender: Any | None = None,
        pacer: Pacer | None = None,
    ) -> None:
        if with_defense and frozen is None:
            raise ValueError("O B4 precisa do detector congelado para montar a evidência do Defensor.")
        self.family = family
        self.with_defense = with_defense
        self.strategy = "B4" if with_defense else "B3"
        self.model_id = model_id
        self.frozen = frozen
        self.instructions = build_attacker_instructions(family)
        self.pacer = pacer or Pacer(min_interval_s)
        self.minute_retries = minute_retries
        self.minute_wait_s = minute_wait_s
        self.agent = agent or Agent(
            model=Groq(id=model_id, temperature=temperature),
            tools=[submit_intent_spec],
            instructions=self.instructions,
            markdown=False,
        )
        self.defender = defender
        if with_defense and defender is None:
            from adversarial_ids.agents.defender import DefenderAgent

            self.defender = DefenderAgent(model_id=defender_model_id or model_id)

    # ------------------------------------------------------------------ #
    def propose(self, context: ProposalContext) -> Proposal:
        meta: dict[str, Any] = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "model": self.model_id}
        plan_block = None
        if self.with_defense:
            defense = self._defense(context, meta)
            meta["defense"] = defense
            plan_block = defense.get("rendered") or "(nenhum plano: " + defense["note"] + ")"

        message = build_round_message(context, plan_block=plan_block)
        response, text = _call(
            self.agent, message, self.pacer,
            minute_retries=self.minute_retries, minute_wait_s=self.minute_wait_s,
        )
        self._count(meta, response)
        if response is None:
            return Proposal(None, error=f"chamada falhou: {text}", meta=meta)
        tool_result = IntentAgent._extract_tool_result(response)
        if tool_result is None:
            content = str(getattr(response, "content", "") or "")[:300]
            return Proposal(None, error=f"o LLM não chamou submit_intent_spec: {content}", meta=meta)
        try:
            intent = compile_intent(
                tool_result, f"{stage_label(context)} {self.strategy} {context.family} rep{context.rep} proposta {context.index}"
            )
        except IntentCompilationError as error:
            return Proposal(None, error=str(error)[:600], meta=meta)
        if intent.base_attack != context.family:
            return Proposal(None, error=f"base_attack {intent.base_attack!r} fora da família da campanha", meta=meta)
        return Proposal(intent, meta=meta)

    @staticmethod
    def _count(meta: dict[str, Any], response: Any) -> None:
        meta["llm_calls"] += 1
        usage = usage_from_response(response) if response is not None else None
        if usage is not None:
            meta["input_tokens"] += usage.input_tokens
            meta["output_tokens"] += usage.output_tokens

    def _defense(self, context: ProposalContext, meta: dict[str, Any]) -> dict[str, Any]:
        """O plano sobre o último candidato válido — reaproveitado se já existe.

        Se uma rodada anterior já consultou o Defensor sobre o mesmo candidato,
        o plano (ou a recusa) dela é reusado: o candidato é o mesmo, a
        evidência é a mesma, e outra chamada só gastaria cota.
        """

        valid = [e for e in context.history if e.get("status") == VALID]
        if not valid:
            return {"source_index": None, "note": "ainda não há candidato válido"}
        source = valid[-1]
        for entry in reversed(context.history):
            previous = (entry.get("proposer") or {}).get("defense") or {}
            if previous.get("source_index") == source["index"] and "reused_from" not in previous:
                return {**previous, "reused_from": entry["index"]}

        report = candidate_report(self.frozen, source["config"], self.frozen.seeds.search)
        ref = f"e1://{context.family}/B4/rep{context.rep}/proposal{source['index']}"
        if context.race_round:
            ref = f"e3://{context.family}/B4/rep{context.rep}/round{context.race_round}/proposal{source['index']}"
        prompt = self.defender.build_prompt(report, report_ref=ref)
        retries: list[str] = []
        response, text = _call(
            self.defender.agent, prompt, self.pacer,
            minute_retries=self.minute_retries, minute_wait_s=self.minute_wait_s,
            malformed_retries=DEFENDER_MALFORMED_RETRIES, retries=retries,
        )
        self._count(meta, response)
        meta["llm_calls"] += retries.count(MALFORMED)  # as recusadas também foram chamadas
        out: dict[str, Any] = {"source_index": source["index"], "report_split": report.split}
        if retries:
            out["provider_retries"] = retries
        try:
            if response is None:
                raise ValueError(f"chamada falhou: {text}")
            plan: DefensePlan = validate_plan_against_report(
                parse_defense_plan(getattr(response, "content", None), expected_report_ref=ref),
                report,
                expected_report_ref=ref,
                attack_key=context.family,
            )
        except Exception as error:  # noqa: BLE001 — recusa do portão: rodada sem plano
            return {**out, "accepted": False, "note": f"plano recusado pelo portão: {str(error)[:300]}"}
        return {
            **out,
            "accepted": True,
            "note": "plano aceito",
            "plan": plan.model_dump(mode="json"),
            "rendered": render_plan(plan),
        }
