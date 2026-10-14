"""Protocolo agrupado por seed e IDS congelado.

Quatro conjuntos de seeds, disjuntos e fixados antes de qualquer execução:

- ``train``: traces da config de referência da família — o IDS aprende aqui;
- ``calibration``: traces da config de referência, de seeds que o treino não
  viu — só o tráfego normal deles fixa os limiares por orçamento de alarmes;
- ``search``: onde as estratégias avaliam candidatos (validação);
- ``test``: onde o candidato escolhido é regenerado e medido. Nenhuma
  estratégia vê um número do teste durante a busca.

Cada seed tem o próprio tráfego normal (ver ``traces.py``), então o split por
seed é um split de verdade: nenhuma mensagem do teste aparece no treino.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.experiments.metrics import (
    DEFAULT_BUDGETS,
    OperatingPoint,
    calibrate_threshold,
    run_summary,
)
from adversarial_ids.experiments.traces import TraceStore


@dataclass(frozen=True)
class SeedPlan:
    train: tuple[int, ...] = (101, 102, 103)
    calibration: tuple[int, ...] = (201, 202)
    search: tuple[int, ...] = (301, 302, 303)
    test: tuple[int, ...] = (401, 402, 403, 404, 405)

    def __post_init__(self) -> None:
        groups = [self.train, self.calibration, self.search, self.test]
        flat = [seed for group in groups for seed in group]
        if len(flat) != len(set(flat)):
            raise ValueError(f"Os conjuntos de seeds precisam ser disjuntos: {groups}.")
        if not all(groups):
            raise ValueError("Nenhum conjunto de seeds pode ser vazio.")


@dataclass
class FrozenIds:
    """Um detector treinado uma vez por família e nunca reajustado depois.

    O limiar de cada orçamento também é congelado aqui, na calibração. Avaliar
    um trace não muda nada no detector — é isso que faz de "evasão" uma medida
    contra um alvo fixo, e não contra um detector que se adapta ao candidato.
    """

    attack_key: str
    reference_config: dict[str, Any]
    store: TraceStore
    seeds: SeedPlan = field(default_factory=SeedPlan)
    detector: str = "random_forest"
    protocol_features: str = "deltas"
    budgets: tuple[float, ...] = DEFAULT_BUDGETS

    def __post_init__(self) -> None:
        self.evaluator: IdsEvaluator | None = None
        self.points: tuple[OperatingPoint, ...] = ()
        self.training: dict[str, Any] = {}

    def fit(self) -> "FrozenIds":
        spec = get_attack_spec(self.attack_key)
        train = [self.store.trace(self.attack_key, self.reference_config, s) for s in self.seeds.train]
        self.evaluator = IdsEvaluator(
            target_attack_label=spec.label,
            protocol_features=self.protocol_features,
            detector=self.detector,
        )
        self.training = self.evaluator.fit_traces([str(p) for p in train])

        normal_scores = []
        for seed in self.seeds.calibration:
            path = self.store.trace(self.attack_key, self.reference_config, seed)
            scored = self.evaluator.score_trace(path)
            normal_scores.append(scored["scores"][~scored["is_attack"]])
        pooled = np.concatenate(normal_scores)
        self.points = tuple(calibrate_threshold(pooled, budget) for budget in self.budgets)
        return self

    def evaluate(self, config: dict[str, Any], seeds: Sequence[int]) -> list[dict[str, Any]]:
        """Um resumo por seed (por run) de ``config`` contra o detector congelado."""

        if self.evaluator is None:
            raise RuntimeError("FrozenIds.fit() precisa rodar antes de evaluate().")
        runs = []
        for seed in seeds:
            path = self.store.trace(self.attack_key, config, seed)
            scored = self.evaluator.score_trace(path)
            runs.append(
                {
                    "seed": int(seed),
                    "trace": str(path),
                    **run_summary(
                        scored["is_attack"], scored["scores"], scored["predicted_attack"], self.points
                    ),
                }
            )
        return runs
