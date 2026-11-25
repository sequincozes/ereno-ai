"""E3 — defesa aplicada e corrida armamentista (RQ3), no protocolo de docs/experiments_e3.md.

Pré-requisito: o E1 completo do braço (a rodada 0 é a campanha do E1, lida do
ledger dele) e, para o braço LLM, o ``run_e1.py report`` depois do B3/B4.

    # 1. rodadas 1…R−1 (retomáveis: rodar de novo continua de onde parou)
    uv run python scripts/run_e3.py campaign --arm B1
    uv run python scripts/run_e3.py campaign --arm B1 --no-defense   # controle
    uv run python scripts/run_e3.py campaign --arm LLM                # B3 ou B4, pela regra do E1

    # 2. matriz cruzada D_i × A_j nas seeds de teste (+ placebo no B1)
    uv run python scripts/run_e3.py matrix --arm B1
    uv run python scripts/run_e3.py matrix --arm B1 --no-defense

    # 3. o plano do Defensor sobre o selecionado de cada rodada (só medido)
    uv run python scripts/run_e3.py plans --arm B1

    # 4. medidas e comparações pré-definidas
    uv run python scripts/run_e3.py report

    # piloto: high_stnum, B1, rep1, R = 2, em e3/pilot/ (não entra nos resultados)
    uv run python scripts/run_e3.py campaign --pilot --arm B1
    uv run python scripts/run_e3.py matrix --pilot --arm B1

Cota diária esgotada (LLM ou Defensor) para sem registrar o passo em curso:
código de saída 3, como o ``run_e1.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from adversarial_ids.config.settings import MODEL_ID, OUTPUTS_DIR  # noqa: E402
from adversarial_ids.experiments.arms_race import (  # noqa: E402
    ROUNDS,
    DetectorLineage,
    IncompleteRound,
    Lineage,
    detector_card,
    importance_shift,
    matrix_values,
    placebo_closure,
    plan_measures,
    race_measures,
    round_complete,
    saturated_budgets,
    score_detector,
    trace_alarm_cost,
)
from adversarial_ids.experiments.metrics import is_saturated, stratified_paired_difference  # noqa: E402
from adversarial_ids.experiments.search import (  # noqa: E402
    OBJECTIVE,
    PRIMARY_BUDGET,
    PROPOSALS_PER_CAMPAIGN,
    CandidateEvaluator,
    Oracle,
    ProposerUnavailable,
    RandomSearch,
    SearchBox,
    campaign_result,
    run_campaign,
    select_best,
)
from adversarial_ids.experiments.sensitivity import reference_config  # noqa: E402
from adversarial_ids.experiments.traces import TraceStore  # noqa: E402
from run_e1 import E1_DIR, FAMILIES, LAMBDA, _dump, _load, _open_bounds  # noqa: E402

E3_DIR = OUTPUTS_DIR / "experiments" / "e3"
LLM_ARMS = ("B3", "B4")
# Fixadas em 20/10/2026, antes do resultado do B3/B4: a família de maior evasão
# suave do B1 no E1 (masquerade_fault, 0,515) e a de maior evasão@10 (high_stnum,
# 0,189). Cabem em ~4–5 dias de cota; as 4 famílias pediriam ~8–9.
LLM_FAMILIES = ("masquerade_fault", "high_stnum")
PILOT = {"families": ["high_stnum"], "reps": [1], "rounds": 2}


def resolve_arm(arm: str) -> str:
    """``LLM`` → B4 se B4−B3 separa a favor do B4 no agregado do E1; senão B3."""

    if arm != "LLM":
        return arm
    path = E1_DIR / "results.json"
    if not path.exists():
        raise SystemExit(f"Sem {path}: rode `run_e1.py report` depois do B3/B4 completos.")
    report = _load(path)
    for strategy in LLM_ARMS:
        for family in FAMILIES:
            reps = (report["per_campaign"].get(family) or {}).get(strategy) or {}
            if len(reps) < 5:
                raise SystemExit(f"E1 incompleto: {family}/{strategy} tem {len(reps)} de 5 campanhas.")
    aggregate = report["metrics"][OBJECTIVE]["comparisons"]["B4-B3"]["aggregate"]
    chosen = "B4" if aggregate["excludes_zero"] and aggregate["value"] > 0 else "B3"
    print(
        f"Braço LLM = {chosen} (E1: B4−B3 = {aggregate['value']:+.3f} "
        f"[{aggregate['low']:+.3f}, {aggregate['high']:+.3f}])",
        flush=True,
    )
    return chosen


def _root(args) -> Path:
    return E3_DIR / "pilot" if args.pilot else E3_DIR


def _lineages(args, arm: str):
    families = args.families or (PILOT["families"] if args.pilot else list(LLM_FAMILIES if arm in LLM_ARMS else FAMILIES))
    reps = args.reps or (PILOT["reps"] if args.pilot else [1, 2, 3, 4, 5])
    for family in families:
        for rep in reps:
            yield Lineage(family, arm, rep, defended=not args.no_defense)


def _rounds(args) -> int:
    return args.rounds or (PILOT["rounds"] if args.pilot else ROUNDS)


def _detector_lineage(lineage: Lineage, store: TraceStore, root: Path, budget: int) -> DetectorLineage:
    return DetectorLineage(lineage, reference_config(lineage.family), store, E1_DIR, root, budget=budget)


def _where(lineage: Lineage) -> Path:
    return Path(lineage.family) / lineage.name / f"rep{lineage.rep}"


def build_proposer(arm: str, family: str, frozen, args):
    if arm == "B1":
        return RandomSearch()
    from adversarial_ids.agents.attacker.agent import LlmAttacker

    return LlmAttacker(
        family,
        with_defense=arm == "B4",
        model_id=args.model_id,
        frozen=frozen,
        defender_model_id=args.defender_model_id,
        min_interval_s=args.min_interval,
    )


def cmd_campaign(args) -> int:
    arm, root, rounds, store = resolve_arm(args.arm), _root(args), _rounds(args), TraceStore()
    if args.no_defense and arm != "B1":
        raise SystemExit("O controle sem defesa só existe no braço B1 (ver docs/experiments_e3.md).")
    for lineage in _lineages(args, arm):
        dl = _detector_lineage(lineage, store, root, args.budget)
        box = SearchBox.for_family(lineage.family, dl.reference, open_bounds=_open_bounds([]))
        try:
            dl.completed_entries(0)
        except IncompleteRound as error:
            print(f"[{lineage.family} {lineage.name} rep{lineage.rep}] pulada: {error}", file=sys.stderr)
            continue
        for race_round in range(1, rounds):
            if round_complete(dl.entries(race_round), args.budget):
                continue
            tag = f"[{lineage.family} {lineage.name} rep{lineage.rep} rodada {race_round}]"
            print(f"{tag} treinando D{race_round}...", flush=True)
            frozen = dl.detector(race_round)
            _dump(root / "ids" / _where(lineage) / f"D{race_round}.json", detector_card(frozen))
            evaluator = CandidateEvaluator(frozen, box, Oracle(lam=args.lam))
            proposer = build_proposer(arm, lineage.family, frozen, args)
            prior = dl.prior(race_round, frozen) if arm in LLM_ARMS else ()
            print(f"{tag} ledger {dl.ledger_path(race_round)}", flush=True)
            try:
                entries = run_campaign(
                    proposer, evaluator, family=lineage.family, rep=lineage.rep,
                    ledger_path=dl.ledger_path(race_round), budget=args.budget,
                    race_round=race_round, prior=prior,
                )
            except ProposerUnavailable as error:
                print(f"  parado (retome depois, nada foi perdido): {error}", file=sys.stderr)
                return 3
            result = campaign_result(frozen, entries, budget=args.budget)
            result.update({"family": lineage.family, "lineage": lineage.name, "rep": lineage.rep, "round": race_round})
            _dump(root / "results" / _where(lineage) / f"round{race_round}.json", result)
            test = result["test"]
            print(
                f"  propostas={result['proposals']} validade={result['validity']:.0%} "
                f"tokens={result['input_tokens'] + result['output_tokens']} "
                + (f"teste contra D{race_round}: evasão suave={test[OBJECTIVE]:.3f}" if test else "teste: sem candidato válido"),
                flush=True,
            )
    return 0


def cmd_matrix(args) -> int:
    arm, root, rounds, store = resolve_arm(args.arm), _root(args), _rounds(args), TraceStore()
    for lineage in _lineages(args, arm):
        out = root / "matrix" / f"{_where(lineage)}.json"
        if out.exists() and not args.force:
            continue
        dl = _detector_lineage(lineage, store, root, args.budget)
        try:
            selected = [select_best(dl.completed_entries(r)) for r in range(rounds)]
        except IncompleteRound as error:
            print(f"[{lineage.family} {lineage.name} rep{lineage.rep}] pulada: {error}", file=sys.stderr)
            continue
        attacks = [None if best is None else best["config"] for best in selected]
        seeds = dl.seeds.test
        indices = range(rounds + 1) if lineage.defended else [0]
        detectors = []
        for index in indices:
            print(f"[{lineage.family} {lineage.name} rep{lineage.rep}] D{index} × A0…A{rounds - 1}", flush=True)
            frozen = dl.detector(index)
            detectors.append({"index": index, **detector_card(frozen), **score_detector(frozen, attacks, dl.reference, seeds)})
        placebo = []
        if lineage.defended and lineage.arm == "B1" and not args.no_placebo:
            for index in range(1, rounds + 1):
                rows = detectors[index]["training"]["attack_only_rows"]
                print(f"[{lineage.family} {lineage.name} rep{lineage.rep}] placebo D'{index} ({rows} linhas)", flush=True)
                frozen = dl.placebo_detector(index, rows)
                placebo.append({"index": index, **detector_card(frozen), **score_detector(frozen, attacks, dl.reference, seeds)})
        _dump(
            out,
            {
                "family": lineage.family, "arm": lineage.arm, "lineage": lineage.name, "rep": lineage.rep,
                "defended": lineage.defended, "rounds": rounds, "seeds": list(seeds),
                "selected": [None if b is None else {"index": b["index"], "diff": b["diff"], "objective": b["objective"]} for b in selected],
                "detectors": detectors, "placebo": placebo,
            },
        )
        print(f"  -> {out}", flush=True)
    return 0


def cmd_plans(args) -> int:
    from adversarial_ids.agents.attacker.agent import DEFENDER_MALFORMED_RETRIES, Pacer, _call
    from adversarial_ids.agents.defender import DefenderAgent
    from adversarial_ids.agents.defender.tools import parse_defense_plan, validate_plan_against_report
    from adversarial_ids.agents.usage import usage_from_response
    from adversarial_ids.experiments.defense_evidence import candidate_report

    arm, root, rounds, store = resolve_arm(args.arm), _root(args), _rounds(args), TraceStore()
    defender, pacer = DefenderAgent(model_id=args.defender_model_id), Pacer(args.min_interval)
    for lineage in _lineages(args, arm):
        if not lineage.defended:
            continue
        dl = _detector_lineage(lineage, store, root, args.budget)
        for race_round in range(rounds):
            out = root / "plans" / _where(lineage) / f"round{race_round}.json"
            if out.exists():
                continue
            try:
                best = select_best(dl.completed_entries(race_round))
            except IncompleteRound as error:
                print(f"  pulada: {error}", file=sys.stderr)
                break
            if best is None:
                _dump(out, {"plan": None, "accepted": False, "note": "rodada sem candidato válido"})
                continue
            frozen = dl.detector(race_round)
            point = next(p for p in frozen.points if str(p.budget_per_10k) == PRIMARY_BUDGET)
            if is_saturated(point):
                # O relatório diria recall 0 por causa do limiar; o Defensor
                # leria isso como evidência. Fica fora da medida, não recusado.
                _dump(out, {"plan": None, "accepted": False, "saturated": True,
                            "note": f"ponto primário ({PRIMARY_BUDGET}/10k) saturado em D{race_round}"})
                continue
            report = candidate_report(frozen, best["config"], dl.seeds.search)
            ref = f"e3://{lineage.family}/{lineage.name}/rep{lineage.rep}/round{race_round}/selected{best['index']}"
            try:
                response, text = _call(defender.agent, defender.build_prompt(report, report_ref=ref), pacer,
                                       minute_retries=4, minute_wait_s=65.0,
                                       malformed_retries=DEFENDER_MALFORMED_RETRIES)
            except ProposerUnavailable as error:
                print(f"  parado (retome depois, nada foi perdido): {error}", file=sys.stderr)
                return 3
            usage = usage_from_response(response) if response is not None else None
            entry = {
                "selected_index": best["index"],
                "report_ref": ref,
                "report_features": [f.feature for f in report.top_features],
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
            }
            try:
                if response is None:
                    raise ValueError(f"chamada falhou: {text}")
                plan = validate_plan_against_report(
                    parse_defense_plan(getattr(response, "content", None), expected_report_ref=ref),
                    report, expected_report_ref=ref, attack_key=lineage.family,
                )
                entry.update({"accepted": True, "plan": plan.model_dump(mode="json")})
            except Exception as error:  # noqa: BLE001 — recusa do portão também é dado
                entry.update({"accepted": False, "plan": None, "note": str(error)[:400]})
            _dump(out, entry)
            print(f"[{lineage.family} {lineage.name} rep{lineage.rep} rodada {race_round}] plano aceito={entry['accepted']}", flush=True)
    return 0


def _paired(values: dict[str, dict[int, tuple[float, float]]]) -> dict | None:
    first = {f: [a for _, (a, _b) in sorted(reps.items())] for f, reps in values.items() if reps}
    second = {f: [b for _, (_a, b) in sorted(reps.items())] for f, reps in values.items() if reps}
    return stratified_paired_difference(first, second).as_dict() if first else None


def cmd_report(args) -> int:
    root = _root(args)
    matrices = [_load(p) for p in sorted((root / "matrix").glob("*/*/rep*.json"))]
    if not matrices:
        raise SystemExit(f"Nenhuma matriz em {root / 'matrix'}: rode `run_e3.py matrix` antes.")
    nodef = {(m["family"], m["arm"], m["rep"]): m for m in matrices if not m["defended"]}
    report: dict = {"metrics": {}, "lineages": {}}
    metrics = (OBJECTIVE, f"evasion@{PRIMARY_BUDGET}")
    for metric in metrics:
        per_arm: dict[str, dict[str, dict]] = {}
        for m in (m for m in matrices if m["defended"]):
            values = matrix_values(m["detectors"], metric)
            measures = race_measures(values)
            last = m["rounds"] - 1
            refs = [_reference(d) for d in m["detectors"]]
            trace_alarms = matrix_values(m["detectors"], f"false_alerts@{PRIMARY_BUDGET}")
            item = {
                "matrix": values.tolist(),
                **measures,
                "reference_recall": [r["recall"] for r in refs],
                "reference_f1": [r["f1"] for r in refs],
                "reference_false_alerts": [r["false_alerts_per_10k_normal"] for r in refs],
                "saturated_budgets": [sorted(saturated_budgets(d)) for d in m["detectors"]],
                "trace_false_alerts": trace_alarms.tolist(),
                "trace_alarm_cost": trace_alarm_cost(trace_alarms),
            }
            if m["placebo"]:
                item["placebo_closure"] = placebo_closure(values, matrix_values(m["placebo"], metric))
                placebo_alarms = matrix_values(m["placebo"], f"false_alerts@{PRIMARY_BUDGET}")
                item["placebo_trace_alarm_cost"] = [
                    float(placebo_alarms[r][r] - trace_alarms[r][r]) for r in range(trace_alarms.shape[1])
                ]
            control = nodef.get((m["family"], m["arm"], m["rep"]))
            if control:
                item["nodefense_race"] = matrix_values(control["detectors"], metric)[0].tolist()
            report["lineages"].setdefault(metric, {})[f"{m['family']}/{m['lineage']}/rep{m['rep']}"] = item

            fam, rep = m["family"], m["rep"]
            arm = per_arm.setdefault(m["arm"], {})

            def put(name, a, b):
                if not (np.isnan(a) or np.isnan(b)):
                    arm.setdefault(name, {}).setdefault(fam, {})[rep] = (a, b)

            put("closure_mean (vs 0)", measures["closure_mean"], 0.0)
            put("forgetting_mean (vs 0)", measures["forgetting_mean"], 0.0)
            put(f"race E[{last}][{last}] - E[0][0]", measures["race"][-1], measures["race"][0])
            if "placebo_closure" in item:
                put("closure - placebo closure", measures["closure_mean"], float(np.nanmean(item["placebo_closure"])))
            if control:
                put(f"E[{last}][{last}] defended - nodefense", measures["race"][-1], item["nodefense_race"][last])
            put("reference recall D_R - D_0", refs[-1]["recall"], refs[0]["recall"])
            put("reference F1 D_R - D_0", refs[-1]["f1"], refs[0]["f1"])
            put("reference false alerts D_R - D_0", refs[-1]["false_alerts_per_10k_normal"], refs[0]["false_alerts_per_10k_normal"])
            # RQ3d secundário (registrado em 22/10/2026, depois do piloto e antes do E3-B1).
            put("trace alarm cost (vs 0)", float(np.nanmean(item["trace_alarm_cost"])), 0.0)
            if "placebo_trace_alarm_cost" in item:
                put("trace alarm cost - placebo", float(np.nanmean(item["trace_alarm_cost"])),
                    float(np.nanmean(item["placebo_trace_alarm_cost"])))
        report["metrics"][metric] = {
            arm: {name: {"aggregate": _paired(values), "units": sum(len(v) for v in values.values())}
                  for name, values in comparisons.items()}
            for arm, comparisons in per_arm.items()
        }
    report["defender"] = _defender_measures(root, matrices)
    _dump(root / "results.json", report)
    for metric in metrics:
        print(f"\n{metric} (teste):")
        for arm, comparisons in report["metrics"][metric].items():
            for name, block in comparisons.items():
                agg = block["aggregate"]
                if agg:
                    flag = "separa" if agg["excludes_zero"] else "não separa"
                    print(f"  {arm} {name}: {agg['value']:+.3f} [{agg['low']:+.3f}, {agg['high']:+.3f}] ({flag}, n={block['units']})")
    defender = report["defender"]
    if defender["plans"]:
        print(f"\nDefensor: {defender['plans']} planos, aceitos {defender['accepted']:.0%}, "
              f"recomendam retreino {defender['recommends_retraining']}, sobreposição média {defender['overlap_mean']}")
    print(f"\n-> {root / 'results.json'}")
    return 0


def _reference(row: dict) -> dict:
    """O custo na referência, com NaN se o ponto primário saturou neste detector."""

    if PRIMARY_BUDGET in saturated_budgets(row):
        nan = float("nan")
        return {**row["reference"], "precision": nan, "recall": nan, "f1": nan, "false_alerts_per_10k_normal": nan}
    return row["reference"]


def _defender_measures(root: Path, matrices: list[dict]) -> dict:
    rows = []
    for m in (m for m in matrices if m["defended"]):
        where = Path(m["family"]) / m["lineage"] / f"rep{m['rep']}"
        for race_round in range(m["rounds"]):
            path = root / "plans" / where / f"round{race_round}.json"
            if not path.exists():
                continue
            entry = _load(path)
            if entry.get("saturated"):
                continue
            before, after = m["detectors"][race_round], m["detectors"][race_round + 1]
            features = {i["feature"] for i in before["importances"]} | set(entry.get("report_features") or ())
            shifted = importance_shift(before["importances"], after["importances"])
            rows.append({"lineage": str(where), "round": race_round, **plan_measures(entry.get("plan"), features, shifted)})
    accepted = [r for r in rows if r["accepted"]]
    overlaps = [r["overlap"] for r in accepted if r["overlap"] is not None]
    return {
        "plans": len(rows),
        "accepted": len(accepted) / len(rows) if rows else None,
        "recommends_retraining": sum(bool(r["recommends_retraining"]) for r in accepted) / len(accepted) if accepted else None,
        "overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, arm=True):
        p.add_argument("--pilot", action="store_true")
        if not arm:
            return
        p.add_argument("--arm", choices=("B1", "B3", "B4", "LLM"), required=True)
        p.add_argument("--family", dest="families", action="append", choices=FAMILIES)
        p.add_argument("--reps", type=int, nargs="+")
        p.add_argument("--rounds", type=int, help=f"rodadas de ataque R (padrão {ROUNDS}; piloto 2)")
        p.add_argument("--budget", type=int, default=PROPOSALS_PER_CAMPAIGN)
        p.add_argument("--no-defense", action="store_true", help="controle: toda rodada contra D0")

    camp = sub.add_parser("campaign")
    common(camp)
    camp.add_argument("--lam", type=float, default=LAMBDA)
    camp.add_argument("--model-id", default=MODEL_ID)
    camp.add_argument("--defender-model-id", default="openai/gpt-oss-20b")
    camp.add_argument("--min-interval", type=float, default=20.0)

    mat = sub.add_parser("matrix")
    common(mat)
    mat.add_argument("--no-placebo", action="store_true")
    mat.add_argument("--force", action="store_true", help="recalcula matrizes já gravadas")

    plans = sub.add_parser("plans")
    common(plans)
    plans.add_argument("--defender-model-id", default="openai/gpt-oss-20b")
    plans.add_argument("--min-interval", type=float, default=20.0)

    rep = sub.add_parser("report")
    common(rep, arm=False)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)
    return {"campaign": cmd_campaign, "matrix": cmd_matrix, "plans": cmd_plans, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
