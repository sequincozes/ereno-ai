"""Ferramentas determinísticas do agente Defensor (Blue Team, épico E5).

Estas funções selecionam as evidências citáveis de um ``DetectionReport`` e
validam a saída da LLM. A LLM não pode inventar métrica, feature, valor,
prioridade ou referência ao relatório de origem — mesmo papel que
``agents/analyst/tools.py`` cumpre para o ``AnalystOutput`` legado.
"""

from __future__ import annotations

import math
from typing import Any

from adversarial_ids.domain import DefenseAction, DefensePlan
from adversarial_ids.domain.detection_report import DetectionReport

PRIORITY_ORDER = ("low", "medium", "high", "critical")

_F1_HIGH_THRESHOLD = 0.5
_F1_MEDIUM_THRESHOLD = 0.8
_RECALL_ESCALATION_THRESHOLD = 0.5


class DefensePlanValidationError(ValueError):
    """Erro acionável: o plano proposto não está fundamentado no relatório."""


def priority_from_report(report: DetectionReport | dict[str, Any]) -> str:
    """Prioridade determinística a partir do F1 e do recall do relatório.

    Reaproveita os limiares 0.50/0.80 de ``analyst/tools.py::severity_from_f1``
    para o nível-base e escalona um degrau em ``PRIORITY_ORDER`` quando o
    recall cai abaixo de 0.50 — o mesmo risco defensivo ("falsos negativos")
    que o Analista trata como central, mas agora capaz de produzir o quarto
    nível (``critical``) quando F1 mediano esconde um recall péssimo.
    """

    report_model = DetectionReport.model_validate(report)

    if report_model.f1 < _F1_HIGH_THRESHOLD:
        base = "high"
    elif report_model.f1 < _F1_MEDIUM_THRESHOLD:
        base = "medium"
    else:
        base = "low"

    if report_model.recall < _RECALL_ESCALATION_THRESHOLD:
        index = min(PRIORITY_ORDER.index(base) + 1, len(PRIORITY_ORDER) - 1)
        return PRIORITY_ORDER[index]

    return base


def select_evidence_candidates(
    report: DetectionReport | dict[str, Any],
    *,
    top_n: int | None = None,
) -> dict[str, float | int | str]:
    """Allowlist de evidências citáveis: métricas do relatório + features.

    Cada chave é um caminho citável ao pé da letra em ``Evidence.metric_or_feature``
    — os campos escalares do ``DetectionReport`` pelo nome, os quatro campos da
    matriz de confusão com caminho ``confusion_matrix.<campo>``, e o nome de
    cada feature em ``top_features`` mapeado para sua ``importance``.
    ``top_n`` limita quantas features entram (usado pelo prompt, que mostra
    menos evidência do que o portão aceita); ``None`` inclui todas.
    """

    if top_n is not None and top_n <= 0:
        raise ValueError("top_n deve ser maior que zero.")

    report_model = DetectionReport.model_validate(report)

    allowlist: dict[str, float | int | str] = {
        "model_name": report_model.model_name,
        "split": report_model.split,
        "accuracy": report_model.accuracy,
        "precision": report_model.precision,
        "recall": report_model.recall,
        "f1": report_model.f1,
        "latency_ms": report_model.latency_ms,
        "confusion_matrix.tp": report_model.confusion_matrix.tp,
        "confusion_matrix.fp": report_model.confusion_matrix.fp,
        "confusion_matrix.fn": report_model.confusion_matrix.fn,
        "confusion_matrix.tn": report_model.confusion_matrix.tn,
    }

    features = report_model.top_features
    if top_n is not None:
        features = features[:top_n]

    for feature in features:
        if feature.feature in allowlist:
            raise DefensePlanValidationError(
                "Nome de feature colide com uma chave de métrica reservada: "
                f"{feature.feature!r}."
            )
        allowlist[feature.feature] = feature.importance

    return allowlist


def parse_defense_plan(content: Any) -> DefensePlan:
    """Converte a resposta Agno (modelo, dict ou string JSON) em ``DefensePlan``.

    Mesma tolerância de formato usada por ``AnalystAgent._parse_response_content``:
    o Agno pode devolver diretamente um modelo Pydantic, um dicionário, ou uma
    string JSON (às vezes cercada por ```` ``` ````), dependendo do modelo.
    """

    if content is None:
        raise RuntimeError("A resposta do Defensor não possui conteúdo.")

    if isinstance(content, DefensePlan):
        return content

    if isinstance(content, str):
        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return DefensePlan.model_validate_json(cleaned)
        except Exception as error:
            raise ValueError("O Defensor retornou um JSON inválido.") from error

    try:
        return DefensePlan.model_validate(content)
    except Exception as error:
        raise ValueError(
            "Não foi possível converter a resposta em DefensePlan."
        ) from error


def _evidence_value_matches(cited: Any, expected: Any) -> bool:
    """Compara um valor citado contra o valor real, por tipo do campo esperado."""

    if isinstance(expected, str):
        return isinstance(cited, str) and cited == expected

    if isinstance(expected, bool):  # nunca ocorre na allowlist atual, defensivo
        return isinstance(cited, bool) and cited == expected

    if isinstance(expected, int):
        if isinstance(cited, bool) or isinstance(cited, str):
            return False
        try:
            cited_float = float(cited)
        except (TypeError, ValueError):
            return False
        return cited_float.is_integer() and int(cited_float) == expected

    if isinstance(expected, float):
        if isinstance(cited, bool) or isinstance(cited, str):
            return False
        try:
            cited_float = float(cited)
        except (TypeError, ValueError):
            return False
        return math.isclose(cited_float, expected, rel_tol=1e-6, abs_tol=1e-9)

    return cited == expected


def _validate_actions(
    actions: tuple[DefenseAction, ...],
    *,
    bucket: str,
    allowed: dict[str, float | int | str],
    expected_report_ref: str | None,
) -> None:
    for action in actions:
        seen_keys: set[str] = set()

        for evidence in action.evidence:
            key = evidence.metric_or_feature

            if key in seen_keys:
                raise DefensePlanValidationError(
                    f"Evidência duplicada na ação ({bucket}): {key}"
                )
            seen_keys.add(key)

            if key not in allowed:
                raise DefensePlanValidationError(
                    f"Evidência inventada pelo Defensor: {key}"
                )

            if not _evidence_value_matches(evidence.value, allowed[key]):
                raise DefensePlanValidationError(
                    f"Valor de evidência incorreto para {key}: "
                    f"esperado={allowed[key]!r}, recebido={evidence.value!r}"
                )

            if (
                evidence.detection_report_ref is not None
                and expected_report_ref is not None
                and evidence.detection_report_ref != expected_report_ref
            ):
                raise DefensePlanValidationError(
                    "Referência de DetectionReport inválida: "
                    f"esperado={expected_report_ref!r}, "
                    f"recebido={evidence.detection_report_ref!r}"
                )


def validate_plan_against_report(
    plan: DefensePlan | dict[str, Any],
    report: DetectionReport | dict[str, Any],
    *,
    expected_report_ref: str | None = None,
) -> DefensePlan:
    """Valida se o ``DefensePlan`` está sustentado pelo ``DetectionReport``.

    Rejeita:

    - evidência (métrica ou feature) que não aparece no relatório;
    - valores de evidência alterados;
    - evidência duplicada dentro da mesma ação, em qualquer um dos três baldes;
    - prioridade incompatível com ``priority_from_report``;
    - ``detection_report_ref`` presente e diferente do relatório desta execução.
    """

    plan_model = DefensePlan.model_validate(plan)
    report_model = DetectionReport.model_validate(report)

    allowed = select_evidence_candidates(report_model)

    for bucket_name, actions in (
        ("detection_actions", plan_model.detection_actions),
        ("containment_actions", plan_model.containment_actions),
        ("hardening_actions", plan_model.hardening_actions),
    ):
        _validate_actions(
            actions,
            bucket=bucket_name,
            allowed=allowed,
            expected_report_ref=expected_report_ref,
        )

    expected_priority = priority_from_report(report_model)
    if plan_model.priority != expected_priority:
        raise DefensePlanValidationError(
            "Prioridade incompatível com o relatório: "
            f"esperado={expected_priority!r}, recebido={plan_model.priority!r}"
        )

    return plan_model
