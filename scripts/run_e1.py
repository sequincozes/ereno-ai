"""E1 — busca e feedback (RQ1–RQ2), no protocolo de docs/experiments_e1.md.

Três passos, nesta ordem:

    # 1. referências: treina e congela o IDS de cada família, mede B0 e o AP
    #    do oráculo nas referências (é daqui que sai o λ)
    uv run python scripts/run_e1.py references

    # 2. campanhas (retomáveis: rodar de novo continua de onde parou)
    uv run python scripts/run_e1.py campaign --family random_replay --strategy B1
    uv run python scripts/run_e1.py campaign --family random_replay --strategy B3 --reps 1
    uv run python scripts/run_e1.py campaign --pilot --family masquerade_fault --strategy B4

    # 3. agregado: testes por campanha, curvas e as comparações pré-definidas
    uv run python scripts/run_e1.py report

``--pilot`` grava em ``e1/pilot/`` e roda o oráculo sem λ (mede, não decide):
é o piloto do protocolo, que não entra nos resultados. Fora dele, λ = 0,5. B3/B4 precisam de GROQ_API_KEY; em cota diária esgotada a
campanha para sem registrar a proposta em curso (código de saída 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from adversarial_ids.config.settings import MODEL_ID, OUTPUTS_DIR  # noqa: E402
from adversarial_ids.experiments.metrics import (  # noqa: E402
    paired_difference,
    stratified_paired_difference,
)
from adversarial_ids.experiments.protocol import FrozenIds  # noqa: E402
from adversarial_ids.experiments.search import (  # noqa: E402
    PRIMARY_BUDGET,
    PROPOSALS_PER_CAMPAIGN,
    CandidateEvaluator,
    OpenLoopLadder,
    Oracle,
    ProposerUnavailable,
    RandomSearch,
    SearchBox,
    campaign_result,
    run_campaign,
    summarize_runs,
)
from adversarial_ids.experiments.sensitivity import reference_config  # noqa: E402
from adversarial_ids.experiments.traces import TraceStore  # noqa: E402

# injection saiu em 15/10/2026: evasão@10 = 0,98 já na referência (teto) —
# ver "Referências" em docs/experiments_e1.md. Continua medível em `references`.
FAMILIES = ("masquerade_fault", "random_replay", "inverse_replay", "high_stnum")
REFERENCE_FAMILIES = (*FAMILIES, "injection")
LAMBDA = 0.5
STRATEGIES = ("B1", "B2", "B3", "B4")
COMPARISONS = (("B3", "B1"), ("B3", "B2"), ("B4", "B3"), ("B1", "B0"), ("B2", "B0"))
E1_DIR = OUTPUTS_DIR / "experiments" / "e1"


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_ids(family: str, store: TraceStore, root: Path) -> FrozenIds:
    """Treina o IDS congelado (traces em cache: refazer é barato e determinístico)."""

    frozen = FrozenIds(family, reference_config(family), store).fit()
    _dump(
        root / "ids" / f"{family}.json",
        {
            "family": family,
            "detector": frozen.detector,
            "protocol_features": frozen.protocol_features,
            "seeds": frozen.seeds.__dict__,
            "training": frozen.training,
            "points": [p.__dict__ for p in frozen.points],
        },
    )
    return frozen


# Limites físicos dos lados que o catálogo deixa abertos (aprovados em 15/10;
# escalas medidas no benigno das seeds de treino — ver "Caixa de busca" em
# docs/experiments_e1.md). Só o masquerade tem lado aberto entre as famílias do E1.
PHYSICAL_BOUNDS: dict[str, float] = {
    # atraso somado ao timestamp da cópia (não é duração da falha): além do
    # heartbeat (máx. 1,031 s) a cópia cruzaria a próxima mensagem legítima
    "fault.durationMs.min": 1000,
    "fault.durationMs.max": 1000,
    # maior |corrente| do benigno (10,31), das faltas reais simuladas
    "analog.deltaAbs.min": 10.0,
    "analog.deltaAbs.max": 10.0,
    # faltas reais movem as somas 0,71–1,41×; a referência já vai a 2,5
    "trapArea.multiplier.min": 2.5,
    "trapArea.multiplier.max": 2.5,
}


def _open_bounds(items: list[str]) -> dict[str, float]:
    bounds = dict(PHYSICAL_BOUNDS)
    for item in items:
        path, _, value = item.partition("=")
        bounds[path.strip()] = float(value)
    return bounds


def cmd_references(args) -> None:
    store = TraceStore()
    rows = []
    for family in args.families:
        print(f"[{family}] treinando o IDS congelado e medindo a referência...", flush=True)
        frozen = frozen_ids(family, store, E1_DIR)
        reference = reference_config(family)
        box = SearchBox.for_family(family, reference, open_bounds=_open_bounds(args.open_bound))
        measured = CandidateEvaluator(frozen, box, Oracle(lam=None)).measure(reference)
        test = summarize_runs(frozen.evaluate(reference, frozen.seeds.test))
        row = {"family": family, **measured, "test": test, "box": box.describe()}
        _dump(E1_DIR / "references" / f"{family}.json", row)
        oracle = measured.get("oracle") or {}
        rows.append(row)
        print(
            f"  status={measured['status']} oráculo AP={oracle.get('ap', float('nan')):.3f} "
            f"p={oracle.get('prevalence', float('nan')):.3f} λ_obs={oracle.get('lambda_observed')} | "
            f"evasão@10 busca={measured['search']['evasion'][PRIMARY_BUDGET]:.3f} "
            f"teste={test['evasion'][PRIMARY_BUDGET]:.3f} AP teste={test['ap']:.3f}",
            flush=True,
        )
    lams = [r["oracle"]["lambda_observed"] for r in rows if r.get("oracle")]
    if lams:
        print(f"\nPior λ observado nas referências: {min(lams):.3f} — λ fica abaixo disso, com folga.")


def build_proposer(args, family: str, frozen: FrozenIds):
    if args.strategy == "B1":
        return RandomSearch()
    if args.strategy == "B2":
        return OpenLoopLadder()
    from adversarial_ids.agents.attacker.agent import LlmAttacker

    return LlmAttacker(
        family,
        with_defense=args.strategy == "B4",
        model_id=args.model_id,
        frozen=frozen,
        defender_model_id=args.defender_model_id,
        min_interval_s=args.min_interval,
    )


def cmd_campaign(args) -> int:
    root = E1_DIR / "pilot" if args.pilot else E1_DIR
    store = TraceStore()
    for family in args.families:
        frozen = frozen_ids(family, store, root)
        box = SearchBox.for_family(family, reference_config(family), open_bounds=_open_bounds(args.open_bound))
        evaluator = CandidateEvaluator(frozen, box, Oracle(lam=None if args.pilot else args.lam))
        proposer = build_proposer(args, family, frozen)
        for rep in args.reps:
            ledger = root / "ledger" / family / args.strategy / f"rep{rep}.jsonl"
            _dump(ledger.with_name("box.json"), {"lam": evaluator.oracle.lam, "box": box.describe()})
            print(f"[{family} {args.strategy} rep{rep}] ledger {ledger}", flush=True)
            try:
                entries = run_campaign(
                    proposer, evaluator, family=family, rep=rep, ledger_path=ledger, budget=args.budget
                )
            except ProposerUnavailable as error:
                print(f"  parado (retome depois, nada foi perdido): {error}", file=sys.stderr)
                return 3
            result = campaign_result(frozen, entries, budget=args.budget)
            result.update({"family": family, "strategy": args.strategy, "rep": rep, "lam": evaluator.oracle.lam})
            _dump(root / "results" / family / args.strategy / f"rep{rep}.json", result)
            test = result["test"]
            print(
                f"  propostas={result['proposals']} validade={result['validity']:.0%} "
                f"chamadas={result['llm_calls']} tokens={result['input_tokens'] + result['output_tokens']} "
                f"teste evasão@10={test['evasion'][PRIMARY_BUDGET] if test else 'sem candidato válido'}",
                flush=True,
            )
    return 0


def _test_evasion(result: dict | None, budget: str) -> float:
    if not result or not result.get("test"):
        return float("nan")
    return float(result["test"]["evasion"][budget])


def cmd_report(args) -> None:
    root = E1_DIR / "pilot" if args.pilot else E1_DIR
    results: dict[str, dict[str, dict[int, dict]]] = {}
    for path in sorted((root / "results").glob("*/*/rep*.json")):
        data = _load(path)
        results.setdefault(data["family"], {}).setdefault(data["strategy"], {})[data["rep"]] = data
    references = {
        p.stem: _load(p) for p in sorted((E1_DIR / "references").glob("*.json"))
    }
    families = sorted(results)
    report: dict = {"budgets": {}, "per_campaign": results}
    for budget in ("1.0", "10.0", "100.0"):
        def series(family: str, strategy: str, reps: list[int]) -> list[float]:
            if strategy == "B0":
                ref = references.get(family)
                value = float(ref["test"]["evasion"][budget]) if ref else float("nan")
                return [value] * len(reps)
            runs = results[family].get(strategy, {})
            return [_test_evasion(runs.get(r), budget) for r in reps]

        table = {}
        for family in families:
            for strategy in ("B0", *STRATEGIES):
                reps = sorted(results[family].get(strategy, {})) or [0]
                values = [v for v in series(family, strategy, reps) if not np.isnan(v)]
                table.setdefault(family, {})[strategy] = float(np.mean(values)) if values else None
        comparisons = {}
        for a, b in COMPARISONS:
            per_family, first, second = {}, {}, {}
            for family in families:
                # B0 não tem repetições: vale o mesmo número em todas as do outro lado.
                done = [set(results[family].get(s, {})) for s in (a, b) if s != "B0"]
                reps = sorted(set.intersection(*done)) if done else []
                if not reps:
                    continue
                first[family], second[family] = series(family, a, reps), series(family, b, reps)
                per_family[family] = paired_difference(first[family], second[family]).as_dict()
            comparisons[f"{a}-{b}"] = {
                "aggregate": stratified_paired_difference(first, second).as_dict() if first else None,
                "per_family": per_family,
            }
        report["budgets"][budget] = {"mean_test_evasion": table, "comparisons": comparisons}
    _dump(root / "results.json", report)
    primary = report["budgets"][PRIMARY_BUDGET]
    print(f"Evasão de teste média @ {PRIMARY_BUDGET} alarmes/10k:")
    for family, row in primary["mean_test_evasion"].items():
        print(f"  {family:18s} " + "  ".join(f"{s}={v:.3f}" if v is not None else f"{s}=—" for s, v in row.items()))
    for name, comparison in primary["comparisons"].items():
        agg = comparison["aggregate"]
        if agg:
            flag = "separa" if agg["excludes_zero"] else "não separa"
            print(f"  {name}: {agg['value']:+.3f} [{agg['low']:+.3f}, {agg['high']:+.3f}] ({flag})")
    print(f"\n-> {root / 'results.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def families_arg(p):
        p.add_argument("--family", dest="families", action="append", choices=REFERENCE_FAMILIES)
        p.add_argument(
            "--open-bound", action="append", default=[], metavar="CAMINHO=VALOR",
            help="limite físico de um lado aberto do catálogo (substitui o fator 4x)",
        )

    refs = sub.add_parser("references")
    families_arg(refs)

    camp = sub.add_parser("campaign")
    families_arg(camp)
    camp.add_argument("--strategy", choices=STRATEGIES, required=True)
    camp.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    camp.add_argument("--lam", type=float, default=LAMBDA, help="fixado em 0,5 (15/10) a partir das referências")
    camp.add_argument("--pilot", action="store_true")
    camp.add_argument("--budget", type=int, default=PROPOSALS_PER_CAMPAIGN)
    camp.add_argument("--model-id", default=MODEL_ID)
    camp.add_argument("--defender-model-id", default=None)
    camp.add_argument("--min-interval", type=float, default=20.0, help="segundos entre chamadas ao LLM")

    rep = sub.add_parser("report")
    rep.add_argument("--pilot", action="store_true")

    # O console do Windows é cp1252 e não tem "λ"; o log vai para arquivo também.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)
    if getattr(args, "families", "absent") is None:
        args.families = list(REFERENCE_FAMILIES if args.command == "references" else FAMILIES)
    if args.command == "references":
        cmd_references(args)
        return 0
    if args.command == "campaign":
        return cmd_campaign(args)
    cmd_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
