"""Evidência do Defensor no E1: um ``DetectionReport`` que muda com o candidato.

O RF é congelado durante a campanha, então as importâncias Gini dele são as
mesmas em toda rodada — um Defensor que só as visse receberia a mesma evidência
a cada candidato, e "o feedback defensivo não ajuda" passaria a medir o desenho
do experimento. Aqui ``top_features`` é o SHAP médio absoluto da classe de
ataque calculado **só sobre as mensagens de ataque que evadiram** no ponto de
operação primário: o que fez o detector chamar aquelas mensagens de normais.
Muda com o candidato porque as mensagens que evadem mudam com ele.

Ver docs/experiments_e1.md, "A evidência do Defensor muda com o candidato".
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.experiments.metrics import counts_at
from adversarial_ids.experiments.protocol import FrozenIds

MISSED_POPULATION = "missed_attack"
ALL_ATTACK_POPULATION = "all_attack"


class EvidenceError(RuntimeError):
    """A evidência não pôde ser calculada (ex.: pacote ``shap`` ausente)."""


def _operating_point(frozen: FrozenIds, budget_per_10k: float):
    for point in frozen.points:
        if point.budget_per_10k == budget_per_10k:
            return point
    raise ValueError(
        f"Orçamento {budget_per_10k} não calibrado; disponíveis: "
        f"{[p.budget_per_10k for p in frozen.points]}."
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def candidate_report(
    frozen: FrozenIds,
    config: dict[str, Any],
    seeds: Sequence[int],
    *,
    budget_per_10k: float = 10.0,
    top_n: int = 15,
    max_samples: int = 200,
) -> DetectionReport:
    """Contagens no ponto primário (somadas nas seeds) + SHAP das mensagens que evadiram.

    Sem falso negativo nenhum, o SHAP é calculado sobre todas as mensagens de
    ataque do candidato — "o que o detector usa para pegá-lo". O ``split`` do
    relatório diz qual população foi explicada, para que ninguém leia um
    ranking como se fosse o outro.
    """

    evaluator = frozen.evaluator
    if evaluator is None:
        raise RuntimeError("FrozenIds.fit() precisa rodar antes de candidate_report().")
    point = _operating_point(frozen, budget_per_10k)

    totals = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    missed: list[pd.DataFrame] = []
    attack: list[pd.DataFrame] = []
    for seed in seeds:
        path = frozen.store.trace(frozen.attack_key, config, seed)
        scored = evaluator.score_trace(path)
        for key, value in counts_at(scored["is_attack"], scored["scores"], point.threshold).items():
            totals[key] += value
        frame = pd.read_csv(path, low_memory=False).drop(columns=[evaluator.label_column])
        is_attack = scored["is_attack"]
        evaded = is_attack & (scored["scores"] < point.threshold)
        missed.append(frame[evaded])
        attack.append(frame[is_attack])

    missed_rows = pd.concat(missed, ignore_index=True)
    if len(missed_rows):
        population, rows = MISSED_POPULATION, missed_rows
    else:
        population, rows = ALL_ATTACK_POPULATION, pd.concat(attack, ignore_index=True)

    top_features = evaluator.get_shap_importances_for_frame(rows, top_n=top_n, max_samples=max_samples)
    if not top_features and len(rows):
        # O best-effort do loop devolve [] em silêncio; aqui isso viraria um
        # relatório sem evidência de feature, e o B4 perderia justamente o que
        # o distingue do B3. Melhor parar.
        raise EvidenceError(
            "SHAP não disponível para a evidência do Defensor — instale o extra "
            "'shap' (uv sync --extra dev --extra dashboard --extra shap --extra xgboost)."
        )

    tp, fn, fp, tn = totals["tp"], totals["fn"], totals["fp"], totals["tn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return DetectionReport(
        model_name=frozen.detector,
        split=f"search_seeds:{'+'.join(str(s) for s in seeds)}:shap={population}",
        accuracy=_ratio(tp + tn, tp + fn + fp + tn),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn),
        top_features=tuple(top_features),
        latency_ms=0.0,
    )
