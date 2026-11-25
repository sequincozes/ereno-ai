"""E1 com orçamento igualado em seis propostas (P0-1 da revisão de 20/11/2026).

O B2 (escada) esgota na 6ª proposta; B1, B3 e B4 usaram até dez. Esta
reanálise refaz a seleção de cada campanha olhando só as seis primeiras
propostas do ledger (inválidas contam), regenera o selecionado nas seeds de
teste quando ele ainda não está em cache e recalcula as diferenças pareadas.
Nenhuma proposta nova, nenhuma chamada a LLM: só o gerador (JAR) e o IDS
congelado do E1, que é retreinado a partir do cache (determinístico).

    # 0. quanto falta gerar (não gera nada)
    uv run python scripts/run_e1_cut6.py plan

    # 1. testa os selecionados no corte (retomável: rodar de novo continua)
    export JAVA_TOOL_OPTIONS=-Xmx2g
    uv run python scripts/run_e1_cut6.py run 2>&1 | tee -a outputs/experiments/e1/cut6/run.log

    # 2. comparações e curva exploratória
    uv run python scripts/run_e1_cut6.py report

Saída em outputs/experiments/e1/cut6/. Os resultados com dez propostas
(e1/results, e1/results.json) não são tocados. A curva de "melhor evasão
suave por número de propostas" usa as seeds de BUSCA e é exploratória: não
foi pré-especificada no protocolo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_e1 import E1_DIR, FAMILIES, METRICS, _dump, _load, _metric, frozen_ids  # noqa: E402

from adversarial_ids.experiments.metrics import (  # noqa: E402
    paired_difference,
    stratified_paired_difference,
)
from adversarial_ids.experiments.search import (  # noqa: E402
    OBJECTIVE,
    PRIMARY_BUDGET,
    best_so_far_curve,
    campaign_result,
    proposals,
    read_ledger,
    select_best,
)
from adversarial_ids.experiments.traces import TraceStore  # noqa: E402

CUT = 6
OUT = E1_DIR / "cut6"
STRATEGIES = ("B1", "B2", "B3", "B4")
# Pré-especificadas por esta revisão: o B2 só entra contra estratégias com o
# mesmo corte; B3−B1 e B4−B3 também são refeitos no corte, para comparação
# com os números de dez propostas.
COMPARISONS = (("B3", "B2"), ("B1", "B2"), ("B3", "B1"), ("B4", "B3"))
REPORT_METRICS = (*METRICS, "ap")


def _ledgers():
    for family in FAMILIES:
        for strategy in STRATEGIES:
            for path in sorted((E1_DIR / "ledger" / family / strategy).glob("rep*.jsonl")):
                yield family, strategy, int(path.stem[3:]), path


def _cut(entries):
    return proposals(entries)[:CUT]


def _full_result(family: str, strategy: str, rep: int) -> dict | None:
    path = E1_DIR / "results" / family / strategy / f"rep{rep}.json"
    return _load(path) if path.exists() else None


def cmd_plan(_args) -> None:
    store = TraceStore()
    total_new = 0
    for family, strategy, rep, path in _ledgers():
        entries = read_ledger(path)
        best6, best10 = select_best(_cut(entries)), select_best(entries)
        if best6 is None:
            print(f"{family:17s} {strategy} rep{rep}: sem candidato válido nas {CUT} primeiras")
            continue
        seeds = (401, 402, 403, 404, 405)
        missing = sum(not store.path_for(family, best6["config"], s).exists() for s in seeds)
        total_new += missing
        same = best10 is not None and best6["index"] == best10["index"]
        before = "igual ao de 10" if same else "antes " + str(best10["index"] if best10 else None)
        print(
            f"{family:17s} {strategy} rep{rep}: selecionado {best6['index']} ({before}) "
            f"busca SE {best6['objective']:.3f} | traces de teste a gerar: {missing}"
        )
    print(f"\nTotal de traces de teste a gerar: {total_new}")


def cmd_run(_args) -> None:
    store = TraceStore()
    frozen_by_family = {}
    for family, strategy, rep, path in _ledgers():
        out = OUT / "results" / family / strategy / f"rep{rep}.json"
        if out.exists():
            continue
        entries = read_ledger(path)
        cut = _cut(entries)
        best6, best10 = select_best(cut), select_best(entries)
        full = _full_result(family, strategy, rep)
        print(f"[{family} {strategy} rep{rep}] selecionado no corte: {best6 and best6['index']}", flush=True)
        if best6 is not None and best10 is not None and best6["index"] == best10["index"] and full:
            # mesmo candidato do resultado de dez propostas: reaproveita o teste
            result = {
                "proposals": len(cut),
                "validity": sum(e["status"] == "valid" for e in cut) / len(cut),
                "curve": best_so_far_curve(entries, CUT),
                "elapsed_s": round(sum(e.get("elapsed_s", 0.0) for e in cut), 1),
                "llm_calls": sum(int((e.get("proposer") or {}).get("llm_calls") or 0) for e in cut),
                "selected_index": best6["index"],
                "selected_diff": best6["diff"],
                "test": full["test"],
                "reused_test_from_10": True,
            }
        else:
            if family not in frozen_by_family:
                print(f"  treinando o IDS congelado de {family} (cache)...", flush=True)
                frozen_by_family[family] = frozen_ids(family, store, OUT)
            result = campaign_result(frozen_by_family[family], cut, budget=CUT)
            result["reused_test_from_10"] = False
        result.update({"family": family, "strategy": strategy, "rep": rep, "cut": CUT})
        _dump(out, result)
        test = result["test"]
        print(
            "  " + (f"teste SE={test[OBJECTIVE]:.3f} evasão@10={test['evasion'][PRIMARY_BUDGET]:.3f} "
                    f"AP={test['ap']:.3f}" if test else "sem candidato válido")
            + (" (reaproveitado)" if result["reused_test_from_10"] else ""),
            flush=True,
        )


def cmd_report(_args) -> None:
    results: dict = {}
    for path in sorted((OUT / "results").glob("*/*/rep*.json")):
        data = _load(path)
        results.setdefault(data["family"], {}).setdefault(data["strategy"], {})[data["rep"]] = data
    families = [f for f in FAMILIES if f in results]
    report: dict = {"cut": CUT, "metrics": {}, "exploratory_curve": {}}

    for metric in REPORT_METRICS:
        table, comparisons = {}, {}
        for family in families:
            for strategy, runs in results[family].items():
                vals = [_metric(r.get("test"), metric) for r in runs.values()]
                vals = [v for v in vals if not np.isnan(v)]
                table.setdefault(family, {})[strategy] = float(np.mean(vals)) if vals else None
        for a, b in COMPARISONS:
            first, second, per_family = {}, {}, {}
            for family in families:
                ra, rb = results[family].get(a, {}), results[family].get(b, {})
                reps = sorted(set(ra) & set(rb))
                if not reps:
                    continue
                first[family] = [_metric(ra[r].get("test"), metric) for r in reps]
                second[family] = [_metric(rb[r].get("test"), metric) for r in reps]
                per_family[family] = paired_difference(first[family], second[family]).as_dict()
            comparisons[f"{a}-{b}"] = {
                "aggregate": stratified_paired_difference(first, second).as_dict() if first else None,
                "per_family": per_family,
            }
        report["metrics"][metric] = {"mean_test": table, "comparisons": comparisons}

    # Curva exploratória: melhor evasão suave nas seeds de BUSCA após k propostas.
    # Antes do primeiro candidato válido, a campanha ainda tem só a referência:
    # vale a evasão suave de busca do B0, o que mantém cada curva (e a média)
    # monótona e sempre sobre as mesmas n campanhas.
    for family, strategy, rep, path in _ledgers():
        floor = _load(E1_DIR / "references" / f"{family}.json")["search"][OBJECTIVE]
        curve = [floor if v is None else max(v, floor) for v in best_so_far_curve(read_ledger(path), 10)]
        report["exploratory_curve"].setdefault(family, {}).setdefault(strategy, []).append(curve)
    for family, by_strategy in report["exploratory_curve"].items():
        for strategy, curves in by_strategy.items():
            arr = np.array(curves, dtype=float)
            by_strategy[strategy] = {"mean": arr.mean(axis=0).tolist(), "n": len(curves)}

    _dump(OUT / "results.json", report)
    for metric in (OBJECTIVE, f"evasion@{PRIMARY_BUDGET}", "ap"):
        block = report["metrics"][metric]
        print(f"\n{metric} no teste, corte em {CUT} propostas (média das campanhas):")
        for family, row in block["mean_test"].items():
            print(f"  {family:18s} " + "  ".join(f"{s}={v:.3f}" if v is not None else f"{s}=—" for s, v in sorted(row.items())))
        for name, comp in block["comparisons"].items():
            agg = comp["aggregate"]
            if agg:
                flag = "separa" if agg["excludes_zero"] else "não separa"
                print(f"  {name}: {agg['value']:+.3f} [{agg['low']:+.3f}, {agg['high']:+.3f}] ({flag})")
    print(f"\n-> {OUT / 'results.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("run")
    sub.add_parser("report")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
