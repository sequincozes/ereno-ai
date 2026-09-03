"""Política determinística de retroalimentação da campanha (épico E10).

Recebe o que a rodada mediu (``DetectionReport``), o que o Defensor concluiu
(``DefensePlan``) e a intenção que produziu aquela variante, e devolve um
``FeedbackDecision``: continua ou para, por qual motivo, e qual é a próxima
``IntentSpec``. Não há LLM neste caminho — mesma divisão do compilador do E2:
o modelo produz a intenção ou o plano de defesa, a política decide o próximo
passo deterministicamente, e apenas validadores decidem o que é aceitável.

Escada de alavancas
-------------------
Só duas alavancas mudam de fato a variante compilada, então a escada usa
exatamente elas, na ordem "aprofundar antes de alargar":

1. ``intensity``: ``low`` → ``medium`` → ``high`` (o passo do compilador vai de
   0.2 a 0.85 da distância até a borda do campo).
2. ``restrictions.max_fields_changed``: ``+1`` por rodada, até o número de
   campos candidatos do efeito (12 em ``masquerade_fault``, teto do contrato
   também 12).

A ``seed`` **não** é alavanca: ``intent_compiler._select_field_paths`` embaralha
os candidatos com ``random.Random(intent.seed)`` e corta em ``[:limite]``, então
manter a seed fixa faz de ``limite+1`` um superconjunto do conjunto anterior —
a escalada fica monotônica ("os mesmos campos, mais um") e a campanha inteira
reproduzível. Mexer na seed daria variedade ao custo de comparabilidade.

``desired_effect`` também não é alavanca: ``lower_f1``, ``lower_recall`` e
``mimic_normal_traffic`` compartilham os mesmos campos candidatos e a mesma
direção no compilador — trocar entre eles não mudaria nada na variante, só
falsificaria a intenção original do usuário. A política nunca reescreve
``objective``, ``base_attack`` nem ``desired_effect``.

Limitação conhecida: em ``generator_mode="cached"`` o dataset é o mesmo em toda
rodada, então a métrica-objetivo não se move e a campanha para na rodada 2 com
``no_improvement``. É o comportamento honesto — a campanha só é fisicamente
mensurável em ``generator_mode="jar"``.
"""

from __future__ import annotations

from typing import Any

from adversarial_ids.config.attack_capabilities import (
    resolve_candidate_paths,
    validate_intent_capability,
)
from adversarial_ids.config.settings import FEEDBACK_MIN_DELTA
from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.domain.feedback_decision import (
    FeedbackDecision,
    FeedbackStopReason,
    RoundOutcome,
)
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentSpec,
)

# Teto do contrato para ``IntentRestrictions.max_fields_changed``.
_MAX_FIELDS_CEILING = 12

# Qual métrica do DetectionReport a intenção pede para minimizar. Os efeitos
# que não são de evasão não têm métrica correspondente no relatório: a política
# para e diz isso, em vez de escolher uma métrica arbitrária e fingir medição.
_OBJECTIVE_METRIC_BY_EFFECT: dict[DesiredEffect, str | None] = {
    DesiredEffect.LOWER_F1: "f1",
    DesiredEffect.LOWER_RECALL: "recall",
    DesiredEffect.MIMIC_NORMAL_TRAFFIC: "recall",
    DesiredEffect.INCREASE_ATTACK_ACTIVITY: None,
    DesiredEffect.INCREASE_RESOURCE_PRESSURE: None,
}

_INTENSITY_LADDER: tuple[IntentIntensity, ...] = (
    IntentIntensity.LOW,
    IntentIntensity.MEDIUM,
    IntentIntensity.HIGH,
)


class FeedbackPolicyError(ValueError):
    """Erro acionável: a política não pôde decidir a próxima rodada."""


def _next_intensity(current: IntentIntensity) -> IntentIntensity | None:
    index = _INTENSITY_LADDER.index(current)
    if index + 1 >= len(_INTENSITY_LADDER):
        return None
    return _INTENSITY_LADDER[index + 1]


def _fields_ceiling(intent: IntentSpec) -> int:
    """Teto de ``max_fields_changed``: nunca mais que os campos candidatos.

    Reaproveita ``resolve_candidate_paths`` (o mesmo portão que o compilador e
    o ``IntentAgent`` usam) em vez de recontar campos aqui.
    """

    _, candidates = resolve_candidate_paths(intent)
    return min(_MAX_FIELDS_CEILING, len(candidates))


def _derive_intent(intent: IntentSpec, **changes: Any) -> IntentSpec:
    """Deriva uma nova ``IntentSpec`` **revalidando** o contrato.

    Não usa ``model_copy(update=...)`` de propósito: no Pydantic v2 ele não
    re-roda os validadores, então um ``max_fields_changed`` fora de faixa
    passaria batido até estourar no compilador.
    """

    payload = intent.model_dump()
    restrictions_changes = changes.pop("restrictions", None)
    if restrictions_changes:
        payload["restrictions"] = {**payload["restrictions"], **restrictions_changes}
    payload.update(changes)
    return IntentSpec.model_validate(payload)


def _escalate(intent: IntentSpec) -> tuple[IntentSpec | None, str]:
    """Sobe um degrau da escada; ``None`` quando não há degrau restante."""

    deeper = _next_intensity(intent.intensity)
    if deeper is not None:
        return (
            _derive_intent(intent, intensity=deeper),
            f"intensidade {intent.intensity.value} -> {deeper.value}",
        )

    ceiling = _fields_ceiling(intent)
    current_width = intent.restrictions.max_fields_changed
    if current_width < ceiling:
        widened = current_width + 1
        return (
            _derive_intent(intent, restrictions={"max_fields_changed": widened}),
            f"campos alterados {current_width} -> {widened} (teto {ceiling})",
        )

    return None, ""


def _best_so_far(
    history: tuple[RoundOutcome, ...],
) -> tuple[float | None, int | None]:
    """Menor valor da métrica-objetivo já medido na campanha, e em que rodada."""

    measured = [
        outcome for outcome in history if outcome.objective_value is not None
    ]
    if not measured:
        return None, None
    best = min(measured, key=lambda outcome: outcome.objective_value)
    return best.objective_value, best.round


def decide_feedback(
    *,
    intent: IntentSpec,
    report: DetectionReport,
    plan: DefensePlan,
    run_id: str,
    round_index: int = 1,
    max_rounds: int = 1,
    parent_run_id: str | None = None,
    history: tuple[RoundOutcome, ...] = (),
    min_delta: float = FEEDBACK_MIN_DELTA,
    target_metric: float | None = None,
) -> FeedbackDecision:
    """Decide se a campanha continua e qual é a próxima intenção.

    ``history`` traz as rodadas **anteriores** desta campanha (a rodada atual
    não entra). ``target_metric`` é opcional: sem alvo absoluto a campanha para
    por platô, esgotamento de alavancas ou teto de rodadas — o que evita cravar
    um limiar arbitrário no código.
    """

    if round_index < 1:
        raise FeedbackPolicyError(f"round_index precisa ser >= 1 (veio {round_index}).")
    if max_rounds < 1:
        raise FeedbackPolicyError(f"max_rounds precisa ser >= 1 (veio {max_rounds}).")

    # A prioridade vem do plano do Defensor, não recalculada aqui: o portão do
    # E5 (``validate_plan_against_report``) já rejeita qualquer plano cuja
    # prioridade discorde de ``priority_from_report`` do mesmo relatório.
    priority = plan.priority
    metric_name = _OBJECTIVE_METRIC_BY_EFFECT[intent.desired_effect]
    objective_value = (
        None if metric_name is None else float(getattr(report, metric_name))
    )

    previous_best, previous_best_round = _best_so_far(history)
    improvement = (
        None
        if objective_value is None or previous_best is None
        else previous_best - objective_value
    )

    current = RoundOutcome(
        round=round_index, run_id=run_id, objective_value=objective_value
    )
    best_value, best_round = _best_so_far(history + (current,))

    def _stop(reason: FeedbackStopReason, rationale: str) -> FeedbackDecision:
        return FeedbackDecision(
            should_continue=False,
            stop_reason=reason,
            objective_metric=metric_name,
            objective_value=objective_value,
            best_value_so_far=best_value,
            best_round=best_round,
            improvement=improvement,
            round=round_index,
            run_id=run_id,
            parent_run_id=parent_run_id,
            defense_priority=priority,
            next_intent=None,
            rationale=rationale,
        )

    # 1) Efeito sem métrica de evasão a minimizar.
    if metric_name is None:
        return _stop(
            FeedbackStopReason.NO_OBJECTIVE_METRIC,
            f"O efeito {intent.desired_effect.value!r} não tem métrica de evasão "
            "no DetectionReport para minimizar — a campanha para na primeira "
            "rodada em vez de escolher uma métrica arbitrária.",
        )

    # 2) Alvo absoluto atingido (quando houver alvo).
    if target_metric is not None and objective_value <= target_metric:
        return _stop(
            FeedbackStopReason.OBJECTIVE_REACHED,
            f"{metric_name}={objective_value:.4f} atingiu o alvo "
            f"{target_metric:.4f} na rodada {round_index}.",
        )

    # 3) Platô: comparado ao melhor valor já medido antes desta rodada.
    if improvement is not None and improvement < min_delta:
        return _stop(
            FeedbackStopReason.NO_IMPROVEMENT,
            f"{metric_name}={objective_value:.4f} não melhorou o melhor valor "
            f"anterior da campanha ({previous_best:.4f} na rodada "
            f"{previous_best_round}): ganho {improvement:.4f} < "
            f"{min_delta:.4f}.",
        )

    # 4) Escada de alavancas esgotada. Vem antes do teto de rodadas de
    # propósito: quando os dois valem, o motivo científico ("não há mais o que
    # escalar") informa mais que o motivo de orçamento.
    next_intent, lever = _escalate(intent)
    if next_intent is None:
        return _stop(
            FeedbackStopReason.LEVERS_EXHAUSTED,
            "Sem alavanca restante: intensidade já é "
            f"{intent.intensity.value} e max_fields_changed já está no teto "
            f"({intent.restrictions.max_fields_changed}).",
        )

    # 5) Teto de rodadas.
    if round_index >= max_rounds:
        return _stop(
            FeedbackStopReason.MAX_ROUNDS_REACHED,
            f"Rodada {round_index} de {max_rounds}: teto de rodadas da campanha "
            "atingido.",
        )

    # Defesa em profundidade: a política nunca devolve uma intenção que o
    # compilador (ou o portão do IntentAgent) rejeitaria.
    validate_intent_capability(next_intent)

    return FeedbackDecision(
        should_continue=True,
        stop_reason=FeedbackStopReason.CONTINUE_CAMPAIGN,
        objective_metric=metric_name,
        objective_value=objective_value,
        best_value_so_far=best_value,
        best_round=best_round,
        improvement=improvement,
        round=round_index,
        run_id=run_id,
        parent_run_id=parent_run_id,
        defense_priority=priority,
        next_intent=next_intent,
        rationale=(
            f"{metric_name}={objective_value:.4f} na rodada {round_index} de "
            f"{max_rounds}; escalando {lever} para a próxima rodada."
        ),
    )
