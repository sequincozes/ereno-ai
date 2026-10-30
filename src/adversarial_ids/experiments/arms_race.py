"""E3 — defesa aplicada e corrida armamentista (RQ3).

Implementa o protocolo fixado em ``docs/experiments_e3.md``. Uma **linhagem** é
(família, braço, repetição): a cada rodada o atacante faz uma campanha do E1
contra o detector da rodada, ``D_r``, congelado durante ela; depois ``D_{r+1}``
é retreinado com a referência + as linhas de ataque das ``k`` melhores
variantes válidas de **todas** as rodadas até ``r``, regeneradas nas seeds de
treino. A rodada 0 é a campanha do E1 do mesmo braço e repetição, lida do
ledger dele — não é rodada de novo.

O pipeline do detector é o do E1 (``protocol_evaluator``); só os dados de
treino mudam, e os limiares são recalibrados no normal de calibração pela mesma
regra. Nada aqui chama LLM nem gera métrica a partir do teste durante a busca.

Dois controles:

- **sem defesa** (``defended=False``): o atacante joga as mesmas rodadas contra
  ``D_0`` — separa "a defesa mudou" de "o atacante teve mais consultas";
- **placebo** (``placebo_detector``): ``D'_r`` recebe o mesmo número de linhas
  de ataque, mas da referência em seeds de treino extras — separa "viu as
  variantes" de "viu mais dado". Só é avaliado, nunca atacado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adversarial_ids.experiments.protocol import FrozenIds, SeedPlan
from adversarial_ids.experiments.search import (
    OBJECTIVE,
    PRIMARY_BUDGET,
    PROPOSALS_PER_CAMPAIGN,
    VALID,
    proposals,
    read_ledger,
    select_best,
    summarize_runs,
)
from adversarial_ids.experiments.traces import TraceStore

ROUNDS = 4
TOP_K = 3
# Seeds de treino extras do placebo: depois das de treino (101–103) e antes das
# de calibração (201+). R = 4, k = 3 e 3 seeds de treino pedem até 36.
PLACEBO_SEED_START = 111
PLACEBO_SEED_LIMIT = 200
# Técnicas do catálogo que correspondem à defesa aplicada (retreinar com as
# variantes = enriquecer o conjunto de treino e retreinar o detector).
RETRAINING_TECHNIQUES = ("detector_retraining", "dataset_enrichment")
IMPORTANCE_SHIFT_TOP = 5


class IncompleteRound(RuntimeError):
    """Uma rodada de que o pedido depende ainda não terminou."""


@dataclass(frozen=True)
class Lineage:
    family: str
    arm: str  # estratégia do atacante do E1: B1, B3 ou B4
    rep: int
    defended: bool = True

    @property
    def name(self) -> str:
        return self.arm if self.defended else f"{self.arm}-nodef"


def top_variants(entries: Sequence[dict[str, Any]], k: int = TOP_K) -> list[dict[str, Any]]:
    """As ``k`` melhores válidas, na ordem de ``select_best`` (a 1ª é o selecionado)."""

    valid = [e for e in proposals(entries) if e["status"] == VALID]
    return sorted(valid, key=lambda e: (-e["objective"], e["search"]["ap"], e["index"]))[:k]


def round_complete(entries: Sequence[dict[str, Any]], budget: int = PROPOSALS_PER_CAMPAIGN) -> bool:
    return len(proposals(entries)) >= budget or any(e.get("status") == "exhausted" for e in entries)


def adversarial_extras(
    selections: Sequence[Sequence[dict[str, Any]]], seeds: Sequence[int]
) -> tuple[tuple[dict[str, Any], int], ...]:
    """(config, seed) das variantes acumuladas: rodada, posição, seed — nessa ordem."""

    return tuple((variant["config"], int(seed)) for chosen in selections for variant in chosen for seed in seeds)


def placebo_seeds(count: int, seeds: SeedPlan, start: int = PLACEBO_SEED_START) -> tuple[int, ...]:
    """``count`` seeds de referência para o placebo, fora de todo conjunto do protocolo."""

    chosen = tuple(range(start, start + count))
    used = {*seeds.train, *seeds.calibration, *seeds.search, *seeds.test}
    if chosen and (chosen[-1] > PLACEBO_SEED_LIMIT or used.intersection(chosen)):
        raise ValueError(f"Seeds do placebo {chosen[0]}–{chosen[-1]} invadem o protocolo {seeds}.")
    return chosen


@dataclass
class DetectorLineage:
    """Os ledgers e os detectores de uma linhagem.

    Treinar ``D_r`` de novo é determinístico (traces em cache, RF com
    ``random_state`` fixo), então nada de modelo é persistido: retomar uma
    rodada retreina o mesmo detector.
    """

    lineage: Lineage
    reference: dict[str, Any]
    store: TraceStore
    e1_root: Path
    e3_root: Path
    k: int = TOP_K
    budget: int = PROPOSALS_PER_CAMPAIGN
    seeds: SeedPlan = field(default_factory=SeedPlan)

    def ledger_path(self, race_round: int) -> Path:
        lin = self.lineage
        if race_round == 0:
            return self.e1_root / "ledger" / lin.family / lin.arm / f"rep{lin.rep}.jsonl"
        return self.e3_root / "ledger" / lin.family / lin.name / f"rep{lin.rep}" / f"round{race_round}.jsonl"

    def entries(self, race_round: int) -> list[dict[str, Any]]:
        return read_ledger(self.ledger_path(race_round))

    def completed_entries(self, race_round: int) -> list[dict[str, Any]]:
        entries = self.entries(race_round)
        if not round_complete(entries, self.budget):
            raise IncompleteRound(
                f"{self.lineage.family}/{self.lineage.name}/rep{self.lineage.rep}: rodada {race_round} "
                f"incompleta ({len(proposals(entries))}/{self.budget} propostas em {self.ledger_path(race_round)})."
            )
        return entries

    def selections(self, upto: int) -> list[list[dict[str, Any]]]:
        """As variantes de treino das rodadas ``0 … upto−1``."""

        return [top_variants(self.completed_entries(r), self.k) for r in range(upto)]

    def _frozen(self, extras, max_rows: int | None = None) -> FrozenIds:
        return FrozenIds(
            self.lineage.family,
            self.reference,
            self.store,
            seeds=self.seeds,
            extra_attack=tuple(extras),
            extra_attack_max_rows=max_rows,
        ).fit()

    def detector(self, index: int) -> FrozenIds:
        """``D_index``. Sem defesa, todo detector é ``D_0``."""

        if index == 0 or not self.lineage.defended:
            return self._frozen(())
        return self._frozen(adversarial_extras(self.selections(index), self.seeds.train))

    def placebo_detector(self, index: int, rows: int) -> FrozenIds:
        """``D'_index``: tantos traces da referência quanto ``D_index`` teve de variantes,
        com as linhas de ataque amostradas até ``rows`` (as que ``D_index`` recebeu)."""

        count = len(adversarial_extras(self.selections(index), self.seeds.train))
        extras = tuple((self.reference, seed) for seed in placebo_seeds(count, self.seeds))
        return self._frozen(extras, max_rows=rows)

    def prior(self, race_round: int, frozen: FrozenIds) -> tuple[dict[str, Any], ...]:
        """Memória do atacante adaptativo: os selecionados das rodadas anteriores,
        re-medidos contra o detector atual nas seeds de busca."""

        items = []
        for r in range(race_round):
            best = select_best(self.completed_entries(r))
            if best is None:
                continue
            now = summarize_runs(frozen.evaluate(best["config"], self.seeds.search))
            items.append(
                {
                    "race_round": r,
                    "index": best["index"],
                    "diff": best["diff"],
                    "objective_then": best["objective"],
                    "objective_now": now[OBJECTIVE],
                }
            )
        return tuple(items)


# --------------------------------------------------------------------------- #
# Matriz cruzada e medidas                                                    #
# --------------------------------------------------------------------------- #
def detector_card(frozen: FrozenIds) -> dict[str, Any]:
    """O que se registra de um detector: treino, limiares e importâncias."""

    training = {k: v for k, v in frozen.training.items() if k != "features"}
    features = frozen.training.get("features") or []
    return {
        "training": training,
        "points": [p.__dict__ for p in frozen.points],
        "importances": frozen.evaluator.get_feature_importances(top_n=len(features) or 15)
        if frozen.evaluator is not None
        else [],
    }


def reference_cost(runs: Sequence[dict[str, Any]], budget: str = PRIMARY_BUDGET) -> dict[str, float]:
    """RQ3d: no ponto primário, contagens somadas nas seeds → precisão, recall, F1.

    Na referência a prevalência é a mesma para todo detector da linhagem, então
    o F1 compara detectores sem a distorção que tem entre variantes.
    """

    totals = {key: sum(int(run["budgets"][budget][key]) for run in runs) for key in ("tp", "fn", "fp", "tn")}
    if any(run["budgets"][budget].get("saturated") for run in runs):
        # Limiar acima do maior score do normal: recall 0 aqui seria do limiar,
        # não do detector (ver metrics.is_saturated).
        nan = float("nan")
        return {**totals, "precision": nan, "recall": nan, "f1": nan,
                "false_alerts_per_10k_normal": nan, "ap": float(np.nanmean([run["ap"] for run in runs])),
                "saturated": True}
    tp, fn, fp, tn = totals["tp"], totals["fn"], totals["fp"], totals["tn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_10k_normal": float(
            np.mean([run["budgets"][budget]["false_alerts_per_10k_normal"] for run in runs])
        ),
        "ap": float(np.nanmean([run["ap"] for run in runs])),
    }


def score_detector(
    frozen: FrozenIds,
    attacks: Sequence[dict[str, Any] | None],
    reference: dict[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Uma linha da matriz (cada ataque contra ``frozen``) + o custo na referência."""

    row = []
    for config in attacks:
        if config is None:
            row.append(None)
            continue
        summary = summarize_runs(frozen.evaluate(config, seeds))
        row.append({k: v for k, v in summary.items() if k != "per_seed"})
    return {"attacks": row, "reference": reference_cost(frozen.evaluate(reference, seeds))}


def _value(cell: dict[str, Any] | None, metric: str) -> float:
    if cell is None:
        return float("nan")
    if metric.startswith("evasion@"):
        return float(cell["evasion"][metric.split("@", 1)[1]])
    if metric.startswith("false_alerts@"):
        return float(cell["false_alerts_per_10k_normal"][metric.split("@", 1)[1]])
    return float(cell[metric])


def saturated_budgets(row: dict[str, Any]) -> set[str]:
    """Orçamentos sem ponto de operação neste detector (``metrics.is_saturated``)."""

    return {str(p["budget_per_10k"]) for p in row.get("points", ()) if p["calibration_alert_rate"] == 0.0}


def _row_value(row: dict[str, Any], cell: dict[str, Any] | None, metric: str) -> float:
    # Matrizes gravadas antes de 22/10/2026 trazem "evasão 1,0" no ponto
    # saturado; a linha do detector guarda os pontos, então a leitura corrige.
    if "@" in metric and metric.split("@", 1)[1] in saturated_budgets(row):
        return float("nan")
    return _value(cell, metric)


def matrix_values(rows: Sequence[dict[str, Any]], metric: str = OBJECTIVE) -> np.ndarray:
    """``E[i][j]``: ``metric`` de ``A_j`` contra ``D_i`` (``nan`` sem selecionado)."""

    return np.array([[_row_value(row, cell, metric) for cell in row["attacks"]] for row in rows], dtype=float)


def trace_alarm_cost(false_alerts: np.ndarray) -> list[float]:
    """RQ3d, secundário: ``F[r+1][r] − F[r][r]``, alarmes falsos/10k no normal
    **do trace da variante** que o retreino acrescentou ao fechar a rodada *r*.

    As features de delta de uma mensagem normal são calculadas contra a
    mensagem anterior, que pode ser de ataque; um detector pode "fechar" a
    variante alarmando também o contexto em volta dela. O custo no normal da
    referência (``reference_cost``) não enxerga isso (piloto de 22/10/2026:
    D0 7% → D1 20% no normal dos traces de A0, placebo 5%).
    """

    return [float(false_alerts[r + 1][r] - false_alerts[r][r]) for r in range(false_alerts.shape[1])]


def race_measures(matrix: np.ndarray) -> dict[str, Any]:
    """RQ3a–c sobre uma matriz ``(R+1) × R``.

    - fechamento ``E[r][r] − E[r+1][r]``: a brecha da rodada que a defesa fechou;
    - esquecimento ``E[R][j] − E[j+1][j]`` (j < R−1): o detector final voltou a
      deixar passar o que já tinha corrigido (positivo = esqueceu);
    - corrida ``E[r][r]``: o que o atacante consegue contra o detector do momento.
    """

    detectors, attacks = matrix.shape
    if detectors != attacks + 1:
        raise ValueError(f"Matriz {matrix.shape}: esperava (R+1) × R.")
    closure = [float(matrix[r][r] - matrix[r + 1][r]) for r in range(attacks)]
    forgetting = [float(matrix[attacks][j] - matrix[j + 1][j]) for j in range(attacks - 1)]
    race = [float(matrix[r][r]) for r in range(attacks)]
    return {
        "closure": closure,
        "closure_mean": _nanmean(closure),
        "forgetting": forgetting,
        "forgetting_mean": _nanmean(forgetting),
        "race": race,
        "race_change": race[-1] - race[0],
    }


def placebo_closure(matrix: np.ndarray, placebo: np.ndarray) -> list[float]:
    """``E[r][r] − E'[r+1][r]``: o fechamento que o placebo ``D'_{r+1}`` daria."""

    return [float(matrix[r][r] - placebo[r][r]) for r in range(matrix.shape[1])]


def _nanmean(values: Sequence[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


# --------------------------------------------------------------------------- #
# O plano do Defensor, medido (não decide nada)                               #
# --------------------------------------------------------------------------- #
def importance_shift(before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]], top: int = IMPORTANCE_SHIFT_TOP) -> list[str]:
    """As ``top`` features cuja importância mais mudou (em módulo) de um detector ao seguinte."""

    old = {item["feature"]: float(item["importance"]) for item in before}
    new = {item["feature"]: float(item["importance"]) for item in after}
    deltas = {f: abs(new.get(f, 0.0) - old.get(f, 0.0)) for f in {*old, *new}}
    return [f for f, _ in sorted(deltas.items(), key=lambda item: (-item[1], item[0]))[:top]]


def plan_measures(plan: dict[str, Any] | None, features: set[str], shifted: Sequence[str]) -> dict[str, Any]:
    """Concordância do plano com a defesa aplicada e com o que o retreino mudou.

    ``features`` separa evidência de feature de evidência de métrica;
    ``overlap`` = fração das features citadas que estão entre as que mais
    mudaram de importância (``None`` quando o plano não cita feature).
    """

    if plan is None:
        return {"accepted": False, "recommends_retraining": None, "overlap": None, "cited_features": []}
    actions = [a for bucket in ("detection_actions", "containment_actions", "hardening_actions") for a in plan.get(bucket, ())]
    techniques = {a["technique"] for a in actions}
    cited = sorted({e["metric_or_feature"] for a in actions for e in a["evidence"]} & features)
    overlap = len(set(cited) & set(shifted)) / len(cited) if cited else None
    return {
        "accepted": True,
        "recommends_retraining": bool(techniques & set(RETRAINING_TECHNIQUES)),
        "techniques": sorted(techniques),
        "cited_features": cited,
        "overlap": overlap,
    }
