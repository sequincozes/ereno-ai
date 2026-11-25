"""Agente Analista (Blue Team).

Recebe métricas produzidas pelo IDS, explica as features mais
influentes e recomenda mitigações defensivas.

A resposta da LLM é validada pelo contrato AnalystOutput e pelas
regras determinísticas presentes em analyst/tools.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

from adversarial_ids.agents.analyst.tools import (
    select_feature_importances,
    severity_from_f1,
    validate_output_against_metrics,
)
from adversarial_ids.config.settings import (
    MODEL_ID,
    PROMPTS_DIR,
    TEMPERATURE,
)
from adversarial_ids.domain import AnalystOutput, Metrics


ANALYST_PROMPT_PATH = PROMPTS_DIR / "analyst.md"


class AnalystAgent:
    """Agente defensivo responsável pela análise das métricas do IDS."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        temperature: float = TEMPERATURE,
        agent: Any | None = None,
        prompt_path: str | Path = ANALYST_PROMPT_PATH,
    ) -> None:
        """Inicializa o Analista.

        O parâmetro ``agent`` permite injetar um agente falso nos testes,
        evitando chamadas reais à API da Groq.
        """

        self.model_id = model_id
        self.temperature = temperature
        self.prompt_path = Path(prompt_path)

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt do Analista não encontrado: {self.prompt_path}"
            )

        self.instructions = self.prompt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not self.instructions:
            raise ValueError("O prompt do Analista está vazio.")

        if agent is not None:
            self.agent = agent
            return

        load_dotenv()

        self.agent = Agent(
            model=Groq(
                id=self.model_id,
                temperature=self.temperature,
            ),
            description=(
                "Analista Blue Team de um experimento acadêmico "
                "sobre robustez de IDS."
            ),
            output_schema=AnalystOutput,
            use_json_mode=True,
            markdown=False,
        )

    def build_prompt(
        self,
        iteration: int,
        metrics: Metrics | dict[str, Any],
        shap_importances: list[dict[str, Any]] | None = None,
    ) -> str:
        """Monta um contexto compacto para reduzir o uso de tokens."""

        if iteration < 0:
            raise ValueError("A iteração não pode ser negativa.")

        metrics_model = Metrics.model_validate(metrics)

        feature_evidence = select_feature_importances(
            metrics=metrics_model,
            shap_importances=shap_importances,
            top_n=5,
        )

        context = {
            "iteration": iteration,
            "model_id": self.model_id,
            "metrics": {
                "accuracy": metrics_model.accuracy,
                "precision_attack": (
                    metrics_model.precision_attack
                ),
                "recall_attack": (
                    metrics_model.recall_attack
                ),
                "f1_score_attack": (
                    metrics_model.f1_score_attack
                ),
                "tp": metrics_model.tp,
                "fp": metrics_model.fp,
                "fn": metrics_model.fn,
                "tn": metrics_model.tn,
                "attack_count": metrics_model.attack_count,
                "normal_count": metrics_model.normal_count,
                "degenerate_variant": (
                    metrics_model.degenerate_variant
                ),
            },
            "feature_evidence": feature_evidence,
            "required_severity": severity_from_f1(
                metrics_model.f1_score_attack
            ),
        }

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            indent=2,
        )

        return (
            f"{self.instructions}\n\n"
            "## Dados da iteração\n\n"
            f"{context_json}\n"
        )

    def analyze(
        self,
        iteration: int,
        metrics: Metrics | dict[str, Any],
        shap_importances: list[dict[str, Any]] | None = None,
    ) -> AnalystOutput:
        """Executa o Analista e devolve uma saída validada."""

        metrics_model = Metrics.model_validate(metrics)

        prompt = self.build_prompt(
            iteration=iteration,
            metrics=metrics_model,
            shap_importances=shap_importances,
        )

        response = self.agent.run(prompt)

        if response is None:
            raise RuntimeError(
                "O agente Analista não retornou uma resposta."
            )

        content = getattr(response, "content", None)

        output = self._parse_response_content(content)

        if output.iteration != iteration:
            raise ValueError(
                "A iteração retornada pelo Analista é incompatível: "
                f"esperado={iteration}, recebido={output.iteration}"
            )

        return validate_output_against_metrics(
            output=output,
            metrics=metrics_model,
            shap_importances=shap_importances,
        )

    @staticmethod
    def _parse_response_content(content: Any) -> AnalystOutput:
        """Converte a resposta Agno em AnalystOutput.

        O Agno pode devolver diretamente um modelo Pydantic, um
        dicionário ou uma string JSON, dependendo do modelo usado.
        """

        if content is None:
            raise RuntimeError(
                "A resposta do Analista não possui conteúdo."
            )

        if isinstance(content, AnalystOutput):
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
                return AnalystOutput.model_validate_json(cleaned)
            except Exception as error:
                raise ValueError(
                    "O Analista retornou um JSON inválido."
                ) from error

        try:
            return AnalystOutput.model_validate(content)
        except Exception as error:
            raise ValueError(
                "Não foi possível converter a resposta em AnalystOutput."
            ) from error
