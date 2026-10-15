"""E1 — busca e feedback: caixa, validade, estratégias determinísticas e campanha.

Implementa o protocolo fixado em ``docs/experiments_e1.md``. Toda estratégia
propõe uma ``IntentSpec`` — o mesmo contrato do pipeline — e daí em diante o
caminho é um só e determinístico: compilador, caixa de busca, geração nas seeds
de busca, piso de volume, oráculo, detector congelado. Assim a única coisa que
varia entre B1–B4 é **quem propõe**.

- B1 propõe valores sorteados na caixa, como ``target_values``;
- B2 percorre a escada de ``core/feedback_policy.py`` sem olhar métrica;
- B3/B4 (LLM) vivem em ``agents/attacker/`` e satisfazem o mesmo ``Proposer``.

Nada aqui chama LLM nem importa agno. A campanha é um ledger JSONL por
(família, estratégia, repetição), uma linha por proposta: retomar lê o ledger e
continua da proposta seguinte, nunca repete uma já registrada.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from adversarial_ids.config.attack_capabilities import FieldCapability, get_attack_capability
from adversarial_ids.config.attacks_registry import get_attack_spec
from adversarial_ids.core.feedback_policy import escalate_intent
from adversarial_ids.core.intent_compiler import (
    _get_by_path,
    _paired_sibling_path,
    compile_attack_candidate,
)
from adversarial_ids.domain.attack_candidate import AttackCandidate
from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
    TargetValue,
)
from adversarial_ids.experiments.metrics import average_precision
from adversarial_ids.experiments.protocol import FrozenIds, protocol_evaluator
from adversarial_ids.experiments.traces import TraceError

PRIMARY_BUDGET = "10.0"
MAX_FIELDS_PER_CANDIDATE = 4
OPEN_BOUND_FACTOR = 4.0
MIN_ATTACK_ROWS = 50
MIN_ATTACK_PREVALENCE = 0.01
PROPOSALS_PER_CAMPAIGN = 10

# Motivos de invalidade, na ordem em que a validade é checada.
VALID = "valid"
NO_INTENT = "no_intent"  # o propositor não chegou a uma IntentSpec (LLM sem tool, recusa do portão)
COMPILE_FAILED = "compile_failed"
OUT_OF_BOX = "out_of_box"
GENERATION_FAILED = "generation_failed"
VOLUME_FLOOR = "volume_floor"
ORACLE_FAILED = "oracle_failed"


# --------------------------------------------------------------------------- #
# Caixa de busca                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldBox:
    path: str
    value_type: str
    reference: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None

    def contains(self, value: Any) -> bool:
        if self.choices is not None:
            return value in self.choices
        if self.value_type == "integer_list":
            return isinstance(value, list) and all(isinstance(v, int) for v in value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return self.low <= value <= self.high

    def describe(self) -> str:
        if self.choices is not None:
            return f"{self.path}: um de {list(self.choices)} (referência {self.reference!r})"
        return f"{self.path}: [{self.low:g}, {self.high:g}] (referência {self.reference!r})"


def _choices(field: FieldCapability) -> tuple[Any, ...] | None:
    if field.choices is not None:
        return tuple(field.choices)
    if field.value_type == "boolean":
        return (False, True)
    return None


@dataclass(frozen=True)
class SearchBox:
    """O mesmo espaço para todas as estratégias de uma família.

    Os campos editáveis do catálogo e os limites dele; um lado sem limite no
    catálogo vira ``OPEN_BOUND_FACTOR`` × a referência (do ``max`` do par, se
    for par), a menos que ``open_bounds`` dê um limite físico para o campo.
    """

    attack_key: str
    reference: dict[str, Any]
    fields: tuple[FieldBox, ...]
    max_fields: int = MAX_FIELDS_PER_CANDIDATE

    @classmethod
    def for_family(
        cls,
        attack_key: str,
        reference: dict[str, Any],
        *,
        max_fields: int = MAX_FIELDS_PER_CANDIDATE,
        open_bound_factor: float = OPEN_BOUND_FACTOR,
        open_bounds: dict[str, float] | None = None,
    ) -> "SearchBox":
        open_bounds = open_bounds or {}
        boxes = []
        for field_cap in get_attack_capability(attack_key).fields:
            ref_value = _get_by_path(reference, field_cap.path)
            choices = _choices(field_cap)
            if choices is not None or field_cap.value_type == "integer_list":
                boxes.append(FieldBox(field_cap.path, field_cap.value_type, ref_value, choices=choices))
                continue
            scale = ref_value
            sibling = _paired_sibling_path(reference, field_cap.path)
            if sibling is not None:
                pair_parent = field_cap.path.rpartition(".")[0]
                scale = _get_by_path(reference, f"{pair_parent}.max")
            low = field_cap.minimum if field_cap.minimum is not None else min(0, ref_value)
            high = field_cap.maximum
            if high is None:
                high = open_bounds.get(field_cap.path, open_bound_factor * scale)
            boxes.append(
                FieldBox(field_cap.path, field_cap.value_type, ref_value, low=low, high=high)
            )
        return cls(attack_key, reference, tuple(boxes), max_fields)

    def field(self, path: str) -> FieldBox:
        for box in self.fields:
            if box.path == path:
                return box
        raise KeyError(path)

    def violations(self, candidate: AttackCandidate) -> list[str]:
        problems = []
        if len(candidate.diff) > self.max_fields:
            problems.append(f"{len(candidate.diff)} campos alterados (máximo {self.max_fields})")
        for change in candidate.diff:
            try:
                box = self.field(change.path)
            except KeyError:
                problems.append(f"{change.path} não está na caixa")
                continue
            if not box.contains(change.new_value):
                problems.append(f"{change.path}={change.new_value!r} fora de {box.describe()}")
        return problems

    def describe(self) -> list[str]:
        return [box.describe() for box in self.fields]


# --------------------------------------------------------------------------- #
# Propositores                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProposalContext:
    family: str
    rep: int
    index: int
    budget: int
    box: SearchBox
    history: tuple[dict[str, Any], ...]


@dataclass
class Proposal:
    """O que um propositor devolve: uma intenção, ou o motivo de não ter uma.

    ``meta`` vai inteiro para o ledger (tokens, chamadas, plano de defesa lido).
    """

    intent: IntentSpec | None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Proposer(Protocol):
    strategy: str

    def propose(self, context: ProposalContext) -> Proposal | None:
        """``None`` = o propositor esgotou (a escada do B2 acabou)."""


class ProposerUnavailable(RuntimeError):
    """O propositor não pode seguir agora (cota diária): pare e retome depois.

    Nada é registrado para a proposta interrompida — ela não conta no orçamento.
    """


def pinned_intent(family: str, values: dict[str, Any], *, seed: int, note: str) -> IntentSpec:
    """Uma intenção inteiramente ditada: só os campos de ``values``, verbatim."""

    paths = tuple(sorted(values))
    return IntentSpec(
        source_prompt=note,
        objective=IntentObjective.EVADE_DETECTION,
        base_attack=family,
        desired_effect=DesiredEffect.LOWER_RECALL,
        intensity=IntentIntensity.LOW,
        restrictions=IntentRestrictions(
            allowed_fields=paths,
            max_fields_changed=len(paths),
            target_values=tuple(
                TargetValue(path=p, value=tuple(v) if isinstance(v, list) else v)
                for p, v in sorted(values.items())
            ),
        ),
        seed=seed,
    )


class RandomSearch:
    """B1: sorteio uniforme na caixa.

    A RNG de cada proposta é derivada de (família, repetição, índice), não de um
    estado que avança — retomar uma campanha pela proposta 7 sorteia a mesma
    proposta 7 que a execução original sortearia.
    """

    strategy = "B1"
    _MAX_TRIES = 200

    def propose(self, context: ProposalContext) -> Proposal:
        rng = random.Random(f"B1:{context.family}:{context.rep}:{context.index}")
        box = context.box
        for _ in range(self._MAX_TRIES):
            k = rng.randint(1, min(box.max_fields, len(box.fields)))
            chosen = sorted(rng.sample([f.path for f in box.fields], k))
            values = self._draw(rng, box, chosen)
            if values is not None:
                intent = pinned_intent(
                    context.family,
                    values,
                    seed=context.rep,
                    note=f"E1 B1 {context.family} rep{context.rep} proposta {context.index}",
                )
                return Proposal(intent)
        return Proposal(None, error="sorteio sem candidato distinto da referência")

    @staticmethod
    def _draw(rng: random.Random, box: SearchBox, paths: list[str]) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        for path in paths:
            fb = box.field(path)
            if fb.choices is not None:
                options = [c for c in fb.choices if c != fb.reference]
                if not options:
                    return None
                values[path] = rng.choice(options)
            elif fb.value_type == "integer":
                values[path] = rng.randint(int(math.ceil(fb.low)), int(math.floor(fb.high)))
            elif fb.value_type == "number":
                values[path] = round(rng.uniform(fb.low, fb.high), 6)
            else:
                return None  # integer_list: nenhuma família do E1 tem; não sortear às cegas
        # Pares {min, max}: com os dois lados sorteados, ordena; com um só, o
        # irmão é o da referência — e o par tem de continuar um intervalo estrito.
        pairs = [(p, s) for p in paths if (s := _paired_sibling_path(box.reference, p)) is not None]
        for path, (sibling_path, leaf) in pairs:
            if sibling_path in values and leaf == "min":
                low, high = sorted((values[path], values[sibling_path]))
                values[path], values[sibling_path] = low, high
        for path, (sibling_path, leaf) in pairs:
            mine = values[path]
            other = values.get(sibling_path, _get_by_path(box.reference, sibling_path))
            if not (mine < other if leaf == "min" else mine > other):
                return None
        if all(values[p] == box.field(p).reference for p in values):
            return None
        return values


class OpenLoopLadder:
    """B2: a escada da política, em malha aberta.

    Começa em intensidade baixa com 1 campo e sobe um degrau por proposta
    (``escalate_intent``), sem olhar métrica nenhuma. A seed da intenção é a
    repetição: ela embaralha a ordem dos campos no compilador. A escada acaba
    quando não há degrau ou quando o próximo alteraria mais campos que a caixa
    admite — a campanha então termina antes do orçamento, e isso fica registrado.
    """

    strategy = "B2"

    def start(self, family: str, rep: int) -> IntentSpec:
        return IntentSpec(
            source_prompt=f"E1 B2 {family} rep{rep}",
            objective=IntentObjective.EVADE_DETECTION,
            base_attack=family,
            desired_effect=DesiredEffect.LOWER_RECALL,
            intensity=IntentIntensity.LOW,
            restrictions=IntentRestrictions(max_fields_changed=1),
            seed=rep,
        )

    def propose(self, context: ProposalContext) -> Proposal | None:
        intent = self.start(context.family, context.rep)
        step = "início"
        for _ in range(context.index):
            intent, step = escalate_intent(intent)
            if intent is None:
                return None
        if intent.restrictions.max_fields_changed > context.box.max_fields:
            return None
        return Proposal(intent, meta={"ladder_step": step})


# --------------------------------------------------------------------------- #
# Avaliação de um candidato                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Oracle:
    """Um detector que **viu** a variante ainda a separa do normal?

    RF com o mesmo preprocessamento, treinado no trace do candidato na primeira
    seed de busca e avaliado na segunda. Passa com ``AP ≥ p + λ·(1 − p)``. Com
    ``lam=None`` o AP é medido e registrado mas não decide nada — só para o
    piloto e para fixar λ nas referências.
    """

    lam: float | None
    detector: str = "random_forest"
    protocol_features: str = "deltas"

    def measure(self, attack_key: str, train_trace: Path, eval_trace: Path) -> dict[str, Any]:
        evaluator = protocol_evaluator(
            get_attack_spec(attack_key).label, self.detector, self.protocol_features
        )
        evaluator.fit_traces([str(train_trace)])
        scored = evaluator.score_trace(eval_trace)
        prevalence = float(np.mean(scored["is_attack"]))
        ap = average_precision(scored["is_attack"], scored["scores"])
        required = None if self.lam is None else prevalence + self.lam * (1 - prevalence)
        return {
            "ap": ap,
            "prevalence": prevalence,
            "lambda_observed": (ap - prevalence) / (1 - prevalence) if prevalence < 1 else None,
            "required_ap": required,
            "passed": None if required is None else bool(ap >= required),
        }


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": run["seed"],
        "rows": run["rows"],
        "attack_rows": run["attack_rows"],
        "ap": run["ap"],
        "budgets": {
            budget: {
                "evasion": values["evasion"],
                "false_alerts_per_10k_normal": values["false_alerts_per_10k_normal"],
            }
            for budget, values in run["budgets"].items()
        },
    }


def summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Média sobre as seeds: o que entra no objetivo e nas tabelas."""

    budgets = runs[0]["budgets"].keys()
    return {
        "evasion": {b: float(np.mean([r["budgets"][b]["evasion"] for r in runs])) for b in budgets},
        "false_alerts_per_10k_normal": {
            b: float(np.mean([r["budgets"][b]["false_alerts_per_10k_normal"] for r in runs]))
            for b in budgets
        },
        "ap": float(np.nanmean([r["ap"] for r in runs])),
        "per_seed": [_compact_run(r) for r in runs],
    }


@dataclass
class CandidateEvaluator:
    """A cadeia de validade, em ordem; a primeira falha dá o motivo."""

    frozen: FrozenIds
    box: SearchBox
    oracle: Oracle
    min_attack_rows: int = MIN_ATTACK_ROWS
    min_prevalence: float = MIN_ATTACK_PREVALENCE

    def evaluate(self, intent: IntentSpec | None, proposer_error: str | None = None) -> dict[str, Any]:
        if intent is None:
            return {"status": NO_INTENT, "reason": proposer_error or "sem intenção"}
        entry: dict[str, Any] = {"intent": intent.model_dump(mode="json")}
        try:
            candidate = compile_attack_candidate(intent)
        except Exception as error:  # noqa: BLE001 — vira motivo no ledger
            return {**entry, "status": COMPILE_FAILED, "reason": str(error)}
        entry["diff"] = [c.model_dump(mode="json") for c in candidate.diff]
        entry["config"] = candidate.config
        problems = self.box.violations(candidate)
        if problems:
            return {**entry, "status": OUT_OF_BOX, "reason": "; ".join(problems)}
        return {**entry, **self.measure(candidate.config)}

    def measure(self, config: dict[str, Any]) -> dict[str, Any]:
        """Geração → volume → oráculo → métricas de busca, para uma config pronta."""

        seeds = self.frozen.seeds.search
        try:
            runs = self.frozen.evaluate(config, seeds)
        except TraceError as error:
            return {"status": GENERATION_FAILED, "reason": str(error).splitlines()[-1][:400]}
        search = summarize_runs(runs)
        out: dict[str, Any] = {"search": search}
        for run in runs:
            share = run["attack_rows"] / run["rows"] if run["rows"] else 0.0
            if run["attack_rows"] < self.min_attack_rows or share < self.min_prevalence:
                return {
                    **out,
                    "status": VOLUME_FLOOR,
                    "reason": f"seed {run['seed']}: {run['attack_rows']} linhas de ataque "
                    f"({share:.2%}); piso {self.min_attack_rows} e {self.min_prevalence:.0%}",
                }
        store = self.frozen.store
        oracle = self.oracle.measure(
            self.frozen.attack_key,
            store.trace(self.frozen.attack_key, config, seeds[0]),
            store.trace(self.frozen.attack_key, config, seeds[1]),
        )
        out["oracle"] = oracle
        if oracle["passed"] is False:
            return {
                **out,
                "status": ORACLE_FAILED,
                "reason": f"AP do oráculo {oracle['ap']:.3f} < {oracle['required_ap']:.3f}",
            }
        out["objective"] = search["evasion"][PRIMARY_BUDGET]
        return {**out, "status": VALID, "reason": None}


# --------------------------------------------------------------------------- #
# Ledger e campanha                                                           #
# --------------------------------------------------------------------------- #
def read_ledger(path: Path) -> list[dict[str, Any]]:
    """As propostas já registradas. Uma última linha truncada (queda durante a
    escrita) é descartada: aquela proposta não chegou a ser registrada."""

    if not path.exists():
        return []
    entries, lines = [], path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise
            path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return entries


def append_ledger(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"não serializável: {type(value)!r}")


def run_campaign(
    proposer: Proposer,
    evaluator: CandidateEvaluator,
    *,
    family: str,
    rep: int,
    ledger_path: Path,
    budget: int = PROPOSALS_PER_CAMPAIGN,
) -> list[dict[str, Any]]:
    """Roda (ou retoma) uma campanha até o orçamento ou o esgotamento.

    ``ProposerUnavailable`` atravessa: a proposta em curso não é registrada e
    a próxima execução retoma dela.
    """

    entries = read_ledger(ledger_path)
    if entries and entries[-1].get("status") == "exhausted":
        return entries
    while len(entries) < budget:
        index = len(entries)
        context = ProposalContext(family, rep, index, budget, evaluator.box, tuple(entries))
        started = time.monotonic()
        proposal = proposer.propose(context)
        if proposal is None:
            entry = {"index": index, "strategy": proposer.strategy, "status": "exhausted"}
            append_ledger(ledger_path, entry)
            entries.append(entry)
            break
        result = evaluator.evaluate(proposal.intent, proposal.error)
        entry = {
            "index": index,
            "strategy": proposer.strategy,
            "family": family,
            "rep": rep,
            **result,
            "proposer": proposal.meta,
            "elapsed_s": round(time.monotonic() - started, 3),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        append_ledger(ledger_path, entry)
        entries.append(entry)
    return entries


def proposals(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """As propostas de verdade (sem a linha de esgotamento)."""

    return [e for e in entries if e.get("status") != "exhausted"]


def select_best(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Maior evasão de busca; desempate por menor AP, depois a mais antiga."""

    valid = [e for e in proposals(entries) if e["status"] == VALID]
    if not valid:
        return None
    return min(valid, key=lambda e: (-e["objective"], e["search"]["ap"], e["index"]))


def best_so_far_curve(entries: Sequence[dict[str, Any]], budget: int) -> list[float | None]:
    """Melhor evasão de busca até a consulta k (k = 1..budget).

    Uma campanha que esgotou antes do orçamento repete o último valor: a
    comparação é por orçamento consumido, e o B2 não ganha nem perde consultas.
    """

    curve: list[float | None] = []
    best: float | None = None
    real = proposals(entries)
    for k in range(budget):
        if k < len(real) and real[k]["status"] == VALID:
            value = real[k]["objective"]
            best = value if best is None else max(best, value)
        curve.append(best)
    return curve


def campaign_result(
    frozen: FrozenIds, entries: Sequence[dict[str, Any]], *, budget: int = PROPOSALS_PER_CAMPAIGN
) -> dict[str, Any]:
    """O selecionado, regenerado e medido nas seeds de teste, + o custo da campanha."""

    real = proposals(entries)
    best = select_best(entries)
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    for entry in real:
        for key in usage:
            usage[key] += int((entry.get("proposer") or {}).get(key) or 0)
    result: dict[str, Any] = {
        "proposals": len(real),
        "exhausted": len(real) < len(entries),
        "validity": sum(e["status"] == VALID for e in real) / budget,
        "invalid_reasons": _count(e["status"] for e in real if e["status"] != VALID),
        "curve": best_so_far_curve(entries, budget),
        "elapsed_s": round(sum(e.get("elapsed_s", 0.0) for e in real), 1),
        **usage,
        "selected_index": None if best is None else best["index"],
        "selected_diff": None if best is None else best["diff"],
        "search": None if best is None else {k: v for k, v in best["search"].items() if k != "per_seed"},
        "test": None,
    }
    if best is not None:
        result["test"] = summarize_runs(frozen.evaluate(best["config"], frozen.seeds.test))
    return result


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
