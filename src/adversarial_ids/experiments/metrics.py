"""Métricas dos experimentos: evasão, AP, orçamento de alarmes e bootstrap por run.

Portado em espírito do protocolo do paper do grayhole
(``explicability-GOOSE-research/experiments/revision_2026``), adaptado à escala
daqui (dezenas de milhares de linhas por trace, não milhões — o
``average_precision_score`` do sklearn sobre os scores basta):

- **Sem limiar:** AP da classe de ataque.
- **Com limiar congelado:** o limiar é escolhido **uma vez**, num conjunto de
  calibração, para um orçamento de alarmes (alarmes por 10 mil mensagens) — e
  reaplicado sem mudança a todo trace avaliado. Reajustar o limiar sobre a
  variante seria deixar a defesa ver o ataque antes de medi-lo.
- **Incerteza:** a unidade de reamostragem é o **run** (o trace de uma seed),
  nunca a mensagem — mensagens de um mesmo trace são correlacionadas. Bootstrap
  percentil, e a diferença pareada entre duas condições só conta como separada
  quando o intervalo exclui zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

ALERTS_PER = 10_000
DEFAULT_BUDGETS: tuple[float, ...] = (1.0, 10.0, 100.0)


@dataclass(frozen=True)
class OperatingPoint:
    """Um limiar congelado e o orçamento que o originou."""

    budget_per_10k: float
    threshold: float
    calibration_alert_rate: float


def calibrate_threshold(normal_scores: np.ndarray, budget_per_10k: float) -> OperatingPoint:
    """Limiar mais permissivo cuja taxa de alarmes no normal cabe no orçamento.

    Calibrado só sobre tráfego **normal** de calibração: o orçamento é de
    alarmes falsos que o operador aceita. Quando nem o maior score cabe (muitos
    empates no topo), devolve-se o limiar acima do máximo — zero alarmes — e a
    taxa alcançada é informada, em vez de relaxar o orçamento em silêncio.
    """

    scores = np.sort(np.asarray(normal_scores, dtype=float))[::-1]
    if scores.size == 0:
        raise ValueError("Calibração sem tráfego normal.")
    allowed = int(np.floor(budget_per_10k / ALERTS_PER * scores.size))
    if allowed <= 0:
        threshold = float(np.nextafter(scores[0], np.inf))
    else:
        # Alarme = score >= limiar. O limiar fica logo acima do (allowed+1)-ésimo
        # maior score, para que no máximo ``allowed`` mensagens o atinjam.
        cut = scores[allowed] if allowed < scores.size else scores[-1]
        threshold = float(np.nextafter(cut, np.inf))
    rate = float((normal_scores >= threshold).mean()) * ALERTS_PER
    return OperatingPoint(budget_per_10k, threshold, rate)


def counts_at(is_attack: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    alarm = np.asarray(scores) >= threshold
    attack = np.asarray(is_attack, dtype=bool)
    return {
        "tp": int((alarm & attack).sum()),
        "fn": int((~alarm & attack).sum()),
        "fp": int((alarm & ~attack).sum()),
        "tn": int((~alarm & ~attack).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    positives, negatives = tp + fn, fp + tn
    recall = tp / positives if positives else float("nan")
    return {
        "recall": recall,
        "evasion": 1.0 - recall if positives else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fpr": fp / negatives if negatives else float("nan"),
        # O que o orçamento limita: alarmes falsos por 10 mil mensagens normais.
        "false_alerts_per_10k_normal": fp / negatives * ALERTS_PER if negatives else float("nan"),
        "alerts_per_10k": (tp + fp) / (positives + negatives) * ALERTS_PER
        if (positives + negatives)
        else float("nan"),
    }


def average_precision(is_attack: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    attack = np.asarray(is_attack, dtype=bool)
    if attack.sum() == 0 or attack.all():
        return float("nan")
    return float(average_precision_score(attack, scores))


def run_summary(
    is_attack: np.ndarray,
    scores: np.ndarray,
    predicted_attack: np.ndarray,
    points: Sequence[OperatingPoint],
) -> dict[str, Any]:
    """Tudo que um trace (um run) contribui para as tabelas."""

    attack = np.asarray(is_attack, dtype=bool)
    argmax = {
        "tp": int((predicted_attack & attack).sum()),
        "fn": int((~predicted_attack & attack).sum()),
        "fp": int((predicted_attack & ~attack).sum()),
        "tn": int((~predicted_attack & ~attack).sum()),
    }
    return {
        "rows": int(attack.size),
        "attack_rows": int(attack.sum()),
        "ap": average_precision(attack, scores),
        "argmax": {**argmax, **rates(argmax)},
        "budgets": {
            str(p.budget_per_10k): {**(c := counts_at(attack, scores, p.threshold)), **rates(c)}
            for p in points
        },
    }


# --------------------------------------------------------------------------- #
# Bootstrap por run                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float | bool]:
        return {"value": self.value, "low": self.low, "high": self.high, "excludes_zero": self.excludes_zero}


def bootstrap_mean(
    values: Sequence[float], *, iterations: int = 1000, seed: int = 42, confidence: float = 0.95
) -> Interval:
    """IC percentil da média de uma métrica por run (ex.: evasão de cada campanha)."""

    data = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(iterations, data.size))
    means = data[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return Interval(float(data.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def paired_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Interval:
    """IC da diferença média ``first - second``, pareada por unidade.

    As duas listas são alinhadas pela unidade (a mesma família e repetição nas
    duas condições). Reamostra-se a unidade, não a mensagem.
    """

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Pareamento exige o mesmo número de unidades ({a.size} != {b.size}).")
    keep = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_mean(list(a[keep] - b[keep]), iterations=iterations, seed=seed, confidence=confidence)
