"""FakeAnalyst — stub determinístico do agente Blue Team (issue #3).

Permite **rodar o loop / testar o Estrategista (M1)** sem depender do agente
Analista real (que ainda nem existe — #11). Emite um dict no formato do contrato
``AnalystOutput`` (#5, M2), derivado de forma determinística das ``Metrics``.
"""

from __future__ import annotations

from typing import Any


class FakeAnalyst:
    """Stand-in do Analista. Interface nova (contrato AnalystOutput)."""

    def analyze(
        self,
        iteration: int,
        metrics: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        metrics = metrics or {}
        importances = metrics.get("top_feature_importances") or []

        deceptive = [
            {
                "feature": item["feature"],
                "importance": item["importance"],
                "explanation": (
                    f"[fake] a feature '{item['feature']}' pesou na decisão do RF."
                ),
            }
            for item in importances[:3]
        ]

        f1 = metrics.get("f1_score_attack")
        severity = self._severity_from_f1(f1)

        return {
            "iteration": iteration,
            "deceptive_features": deceptive,
            "diagnosis": f"[fake] diagnóstico da iteração {iteration} (f1={f1}).",
            "mitigations": [
                {
                    "type": "threshold",
                    "recommendation": "[fake] revisar o limiar de decisão do RF.",
                },
                {
                    "type": "feature",
                    "recommendation": "[fake] revisar features temporais/derivadas.",
                },
            ],
            "severity": severity,
        }

    @staticmethod
    def _severity_from_f1(f1: float | None) -> str:
        if f1 is None:
            return "low"
        if f1 < 0.5:
            return "high"
        if f1 < 0.8:
            return "medium"
        return "low"
