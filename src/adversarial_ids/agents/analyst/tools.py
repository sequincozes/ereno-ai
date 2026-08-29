"""Ferramentas determinísticas do agente Analista (Blue Team).

Estas funções selecionam as evidências produzidas pelo IDS e validam
a saída da LLM. A LLM não pode inventar features, importâncias ou
classificações de severidade.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from adversarial_ids.domain import AnalystOutput, Metrics


def severity_from_f1(f1_score: float | None) -> str:
    """Classifica a severidade de acordo com o F1 da classe masquerade."""

    if f1_score is None:
        return "low"

    if f1_score < 0.5:
        return "high"

    if f1_score < 0.8:
        return "medium"

    return "low"


def _normalize_importance(item: Any) -> dict[str, Any]:
    """Converte FeatureImportance ou dict para um formato comum."""

    if hasattr(item, "model_dump"):
        item = item.model_dump()

    if not isinstance(item, Mapping):
        raise TypeError(
            "A importância deve ser um dicionário ou modelo Pydantic."
        )

    if "feature" not in item or "importance" not in item:
        raise ValueError(
            "Cada importância precisa conter 'feature' e 'importance'."
        )

    feature = str(item["feature"]).strip()
    importance = float(item["importance"])

    if not feature:
        raise ValueError("O nome da feature não pode estar vazio.")

    if importance < 0:
        raise ValueError("A importância da feature não pode ser negativa.")

    return {
        "feature": feature,
        "importance": importance,
    }


def select_feature_importances(
    metrics: Metrics | dict[str, Any],
    shap_importances: list[dict[str, Any]] | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Seleciona as evidências usadas pelo Analista.

    SHAP possui prioridade quando estiver disponível. Caso contrário,
    são utilizadas as importâncias Gini presentes em Metrics.
    """

    if top_n <= 0:
        raise ValueError("top_n deve ser maior que zero.")

    metrics_model = Metrics.model_validate(metrics)

    if shap_importances:
        source: list[Any] = shap_importances
    else:
        source = list(metrics_model.top_feature_importances)

    normalized = [
        _normalize_importance(item)
        for item in source
    ]

    normalized.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return normalized[:top_n]


def validate_output_against_metrics(
    output: AnalystOutput | dict[str, Any],
    metrics: Metrics | dict[str, Any],
    shap_importances: list[dict[str, Any]] | None = None,
) -> AnalystOutput:
    """Valida se a resposta do Analista está sustentada pelas métricas.

    Rejeita:

    - features que não aparecem nas evidências;
    - valores de importância alterados;
    - features duplicadas;
    - severidade incompatível com o F1;
    - ausência de features quando há evidências disponíveis.
    """

    output_model = AnalystOutput.model_validate(output)
    metrics_model = Metrics.model_validate(metrics)

    evidence = select_feature_importances(
        metrics=metrics_model,
        shap_importances=shap_importances,
        top_n=15,
    )

    allowed_importances = {
        item["feature"]: item["importance"]
        for item in evidence
    }

    if allowed_importances and not output_model.deceptive_features:
        raise ValueError(
            "O Analista não explicou nenhuma feature, "
            "embora existam evidências disponíveis."
        )

    seen_features: set[str] = set()

    for feature in output_model.deceptive_features:
        if feature.feature in seen_features:
            raise ValueError(
                f"Feature duplicada na saída: {feature.feature}"
            )

        seen_features.add(feature.feature)

        if feature.feature not in allowed_importances:
            raise ValueError(
                f"Feature inventada pelo Analista: {feature.feature}"
            )

        expected_importance = allowed_importances[feature.feature]

        if not math.isclose(
            feature.importance,
            expected_importance,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"Importância incorreta para {feature.feature}: "
                f"esperado={expected_importance}, "
                f"recebido={feature.importance}"
            )

    expected_severity = severity_from_f1(
        metrics_model.f1_score_attack
    )

    if output_model.severity != expected_severity:
        raise ValueError(
            "Severidade incompatível com o F1: "
            f"esperado={expected_severity}, "
            f"recebido={output_model.severity}"
        )

    return output_model
