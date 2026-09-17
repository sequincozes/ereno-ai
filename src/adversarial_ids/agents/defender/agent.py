"""Agente Defensor (Blue Team, épico E5).

Recebe o ``DetectionReport`` de uma execução do IDS e produz um
``DefensePlan`` fundamentado: ações de detecção, contenção e hardening,
cada uma amarrada a evidência real do relatório e a um método de
verificação.

A resposta da LLM é validada pelo contrato ``DefensePlan`` e pelas regras
determinísticas de ``agents/defender/tools.py`` — a LLM nunca escreve um
plano aceito sem que cada evidência bata com o relatório de origem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

from adversarial_ids.agents.defender.tools import (
    legal_techniques_by_bucket,
    parse_defense_plan,
    priority_from_report,
    select_evidence_candidates,
    techniques_by_evidence,
    validate_plan_against_report,
)
from adversarial_ids.config.settings import MODEL_ID, PROMPTS_DIR, TEMPERATURE
from adversarial_ids.domain import DefensePlan
from adversarial_ids.domain.defense_plan import VALIDATION_METRICS
from adversarial_ids.domain.detection_report import DetectionReport

DEFENDER_PROMPT_PATH = PROMPTS_DIR / "defender.md"


class DefenderAgent:
    """Agente defensivo responsável por propor o DefensePlan da iteração."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        temperature: float = TEMPERATURE,
        agent: Any | None = None,
        prompt_path: str | Path = DEFENDER_PROMPT_PATH,
    ) -> None:
        """Inicializa o Defensor.

        O parâmetro ``agent`` permite injetar um agente falso nos testes,
        evitando chamadas reais à API da Groq.
        """

        self.model_id = model_id
        self.temperature = temperature
        self.prompt_path = Path(prompt_path)

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt do Defensor não encontrado: {self.prompt_path}"
            )

        self.instructions = self.prompt_path.read_text(encoding="utf-8").strip()

        if not self.instructions:
            raise ValueError("O prompt do Defensor está vazio.")

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
                "Defensor Blue Team de um experimento acadêmico "
                "sobre robustez de IDS."
            ),
            output_schema=DefensePlan,
            use_json_mode=True,
            markdown=False,
        )

    def build_prompt(
        self,
        report: DetectionReport | dict[str, Any],
        *,
        report_ref: str | None = None,
    ) -> str:
        """Monta um contexto compacto com a evidência citável do relatório.

        ``legal_techniques``, ``techniques_by_evidence`` e ``validation_metrics``
        vêm dos mesmos mapas que o contrato e o portão usam para recusar. O
        prompt não repete essas listas em texto: uma cópia estática envelheceria
        em silêncio no dia em que o catálogo de técnicas crescer.

        ``techniques_by_evidence`` usa o mesmo ``top_n`` de
        ``citable_evidence``: mostrar técnicas ancoradas numa feature que o
        Defensor não pode citar seria oferecer uma escolha que o portão recusa.
        """

        report_model = DetectionReport.model_validate(report)

        context = {
            "model_name": report_model.model_name,
            "split": report_model.split,
            "required_priority": priority_from_report(report_model),
            "detection_report_ref": report_ref,
            "citable_evidence": select_evidence_candidates(report_model, top_n=5),
            "techniques_by_evidence": techniques_by_evidence(report_model, top_n=5),
            "legal_techniques": legal_techniques_by_bucket(),
            "validation_metrics": list(VALIDATION_METRICS),
        }

        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        return f"{self.instructions}\n\n## Dados da execução\n\n{context_json}\n"

    def defend(
        self,
        report: DetectionReport | dict[str, Any],
        *,
        report_ref: str | None = None,
        attack_key: str | None = None,
    ) -> DefensePlan:
        """Executa o Defensor e devolve um DefensePlan validado.

        ``attack_key`` é o ataque-base desta execução. Ele não muda o que o
        portão aceita — alimenta os achados de playbook da avaliação por regras,
        que são conselho — mas é o que permite ao orquestrador persistir um
        relatório de regras que sabe de que cenário IEC-61850 está falando.
        """

        report_model = DetectionReport.model_validate(report)

        prompt = self.build_prompt(report_model, report_ref=report_ref)

        response = self.agent.run(prompt)

        if response is None:
            raise RuntimeError("O agente Defensor não retornou uma resposta.")

        content = getattr(response, "content", None)
        plan = parse_defense_plan(content)

        return validate_plan_against_report(
            plan,
            report_model,
            expected_report_ref=report_ref,
            attack_key=attack_key,
        )
