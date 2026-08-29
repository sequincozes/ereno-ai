"""Teste manual do agente Analista usando a Groq."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from adversarial_ids.agents.analyst import AnalystAgent


def main() -> None:
    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY não encontrada. Verifique o arquivo .env."
        )

    metrics = {
        "evaluation_type": "adversarial",
        "accuracy": 0.75,
        "precision_attack": 0.70,
        "recall_attack": 0.60,
        "f1_score_attack": 0.65,
        "support_attack": 20,
        "tp": 12,
        "fp": 4,
        "fn": 8,
        "tn": 20,
        "attack_count": 20,
        "normal_count": 24,
        "config_changed": True,
        "degenerate_variant": False,
        "top_feature_importances": [
            {
                "feature": "TrapAreaSum",
                "importance": 0.42,
            },
            {
                "feature": "cbStatusDiff",
                "importance": 0.30,
            },
            {
                "feature": "analogDelta",
                "importance": 0.18,
            },
        ],
    }

    analyst = AnalystAgent()

    result = analyst.analyze(
        iteration=1,
        metrics=metrics,
    )

    print("\nANÁLISE VALIDADA:\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
