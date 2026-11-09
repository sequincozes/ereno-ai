"""E2 (reduzido) — as variantes que o E1 achou contra o RF transferem para outros detectores?

Mesmo protocolo do E1 (seeds, pipeline, limiares por orçamento calibrados no
normal das seeds de calibração), trocando só o detector do ``FrozenIds``. As
configs avaliadas são as selecionadas por cada campanha do E1 (B1–B4) — a
seleção usou o RF e não é refeita: a pergunta é de *transferência*.

    uv run python scripts/run_e2.py run                       # todos os detectores e famílias
    uv run python scripts/run_e2.py run --detector xgboost --family high_stnum
    uv run python scripts/run_e2.py report

Só lê traces do cache do ``TraceStore``: um trace ausente é erro, nunca uma
chamada ao JAR. Por isso pode rodar ao mesmo tempo que uma campanha do E1/E3
(que usa o runtime do ERENO) — mas cuide da RAM.

O RF entra como controle de integridade: os números dele têm de bater com os
``results/`` do E1 (mesmo detector, mesmas seeds, mesmo pipeline).

Evasão suave (1 − score médio nas mensagens de ataque) só é comparável entre
detectores cujo score é probabilidade (RF, XGBoost, DT). No SVM linear o score
é a função de decisão: a evasão suave sai como NaN e valem AP e evasão por
orçamento, que só dependem da ordem dos scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from adversarial_ids.config.settings import OUTPUTS_DIR  # noqa: E402
from adversarial_ids.experiments.metrics import paired_difference, stratified_paired_difference  # noqa: E402
from adversarial_ids.experiments.protocol import FrozenIds  # noqa: E402
from adversarial_ids.experiments.search import (  # noqa: E402
    OBJECTIVE,
    PRIMARY_BUDGET,
    read_ledger,
    select_best,
    summarize_runs,
)
from adversarial_ids.experiments.sensitivity import reference_config  # noqa: E402
from adversarial_ids.experiments.traces import TraceError, TraceStore  # noqa: E402

FAMILIES = ("masquerade_fault", "random_replay", "inverse_replay", "high_stnum")
STRATEGIES = ("B1", "B2", "B3", "B4")
DETECTORS = ("random_forest", "xgboost", "decision_tree", "svm_linear")
PROBABILISTIC = {"random_forest", "xgboost", "decision_tree"}
E1_DIR = OUTPUTS_DIR / "experiments" / "e1"
E2_DIR = OUTPUTS_DIR / "experiments" / "e2"
METRICS = (OBJECTIVE, "evasion@1.0", "evasion@10.0", "evasion@100.0", "ap")


@dataclass
class CachedOnlyStore(TraceStore):
    """``TraceStore`` que nunca gera: trace fora do cache vira erro."""

    def trace(self, attack_key: str, config: dict[str, Any], seed: int) -> Path:
        target = self.path_for(attack_key, config, seed)
        if not target.exists():
            raise TraceError(f"Trace fora do cache ({attack_key}, seed {seed}): {target}")
        return target


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


def selected_variants(family: str) -> list[dict[str, Any]]:
    """A config selecionada de cada campanha do E1 que tem resultado de teste."""

    out = []
    for strategy in STRATEGIES:
        for result_path in sorted((E1_DIR / "results" / family / strategy).glob("rep*.json")):
            result = _load(result_path)
            if not result.get("test"):
                continue
            ledger = E1_DIR / "ledger" / family / strategy / f"rep{result['rep']}.jsonl"
            best = select_best(read_ledger(ledger))
            if best is None or best["index"] != result["selected_index"]:
                raise RuntimeError(f"Seleção do ledger não bate com o resultado: {result_path}")
            out.append(
                {
                    "strategy": strategy,
                    "rep": int(result["rep"]),
                    "config": best["config"],
                    "e1_test": {k: v for k, v in result["test"].items() if k != "per_seed"},
                }
            )
    return out


def _summary(frozen: FrozenIds, config: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_runs(frozen.evaluate(config, frozen.seeds.test))
    if frozen.detector not in PROBABILISTIC:
        summary[OBJECTIVE] = float("nan")
        summary["attack_score_mean"] = float("nan")
    return summary


def cmd_run(args) -> int:
    store = CachedOnlyStore()
    for family in args.families:
        variants = selected_variants(family)
        for detector in args.detectors:
            out = E2_DIR / family / f"{detector}.json"
            if out.exists() and not args.force:
                print(f"[{family} {detector}] já feito: {out}", flush=True)
                continue
            print(f"[{family} {detector}] treinando nas seeds de treino...", flush=True)
            frozen = FrozenIds(family, reference_config(family), store, detector=detector).fit()
            reference = _summary(frozen, reference_config(family))
            rows = []
            for variant in variants:
                test = _summary(frozen, variant["config"])
                rows.append({k: variant[k] for k in ("strategy", "rep")} | {"test": test})
                print(
                    f"  {variant['strategy']} rep{variant['rep']}: evasão suave={test[OBJECTIVE]:.3f} "
                    f"evasão@10={test['evasion'][PRIMARY_BUDGET]:.3f} AP={test['ap']:.3f}",
                    flush=True,
                )
                if detector == "random_forest":
                    e1 = variant["e1_test"]
                    if not np.isclose(test[OBJECTIVE], e1[OBJECTIVE], atol=1e-9):
                        raise RuntimeError(
                            f"RF não reproduz o E1 ({variant['strategy']} rep{variant['rep']}): "
                            f"{test[OBJECTIVE]} != {e1[OBJECTIVE]}"
                        )
            _dump(
                out,
                {
                    "family": family,
                    "detector": detector,
                    "seeds": frozen.seeds.__dict__,
                    "training": frozen.training,
                    "points": [p.__dict__ for p in frozen.points],
                    "reference": reference,
                    "variants": rows,
                },
            )
            print(
                f"  referência: evasão suave={reference[OBJECTIVE]:.3f} "
                f"evasão@10={reference['evasion'][PRIMARY_BUDGET]:.3f} AP={reference['ap']:.3f} -> {out}",
                flush=True,
            )
    return 0


def _metric(summary: dict | None, metric: str) -> float:
    if not summary:
        return float("nan")
    if metric.startswith("evasion@"):
        return float(summary["evasion"][metric.split("@", 1)[1]])
    return float(summary.get(metric, float("nan")))


def cmd_report(args) -> None:
    data: dict[str, dict[str, Any]] = {}
    for path in sorted(E2_DIR.glob("*/*.json")):
        if path.name == "results.json":
            continue
        d = _load(path)
        data.setdefault(d["family"], {})[d["detector"]] = d
    report: dict[str, Any] = {"metrics": {}}
    for metric in METRICS:
        table: dict[str, Any] = {}
        transfer: dict[str, Any] = {}
        for detector in DETECTORS:
            # Transferência: a variante selecionada contra o RF evade mais que a
            # referência *neste* detector? Pareado por campanha, estratificado por família.
            first, second = {}, {}
            for family, by_det in sorted(data.items()):
                d = by_det.get(detector)
                if not d:
                    continue
                ref = _metric(d["reference"], metric)
                row = table.setdefault(family, {}).setdefault(detector, {"B0": ref})
                for strategy in STRATEGIES:
                    vals = [_metric(v["test"], metric) for v in d["variants"] if v["strategy"] == strategy]
                    vals = [v for v in vals if not np.isnan(v)]
                    row[strategy] = float(np.mean(vals)) if vals else None
                values = [_metric(v["test"], metric) for v in d["variants"]]
                first[family], second[family] = values, [ref] * len(values)
            if first:
                transfer[detector] = {
                    "aggregate": stratified_paired_difference(first, second).as_dict(),
                    "per_family": {f: paired_difference(first[f], second[f]).as_dict() for f in first},
                }
        # A vantagem do LLM (B3−B1 no E1, medida contra o RF) se mantém em cada detector?
        b3_b1: dict[str, Any] = {}
        for detector in DETECTORS:
            first, second = {}, {}
            for family, by_det in sorted(data.items()):
                d = by_det.get(detector)
                if not d:
                    continue
                per = {(v["strategy"], v["rep"]): _metric(v["test"], metric) for v in d["variants"]}
                reps = sorted(r for s, r in per if s == "B3" and ("B1", r) in per)
                if reps:
                    first[family] = [per[("B3", r)] for r in reps]
                    second[family] = [per[("B1", r)] for r in reps]
            if first:
                b3_b1[detector] = stratified_paired_difference(first, second).as_dict()
        report["metrics"][metric] = {
            "mean_test": table,
            "variant_minus_reference": transfer,
            "B3_minus_B1": b3_b1,
        }
    _dump(E2_DIR / "results.json", report)
    for metric in (OBJECTIVE, f"evasion@{PRIMARY_BUDGET}", "ap"):
        block = report["metrics"][metric]
        print(f"\n{metric} no teste (média das campanhas do E1, por estratégia):")
        for family, by_det in block["mean_test"].items():
            for detector, row in by_det.items():
                cells = "  ".join(
                    f"{s}={v:.3f}" if v is not None and not np.isnan(v) else f"{s}=—" for s, v in row.items()
                )
                print(f"  {family:17s} {detector:14s} {cells}")
        print("  variante − referência (todas as campanhas, estratificado por família):")
        for detector, t in block["variant_minus_reference"].items():
            agg = t["aggregate"]
            if agg and not np.isnan(agg["value"]):
                flag = "separa" if agg["excludes_zero"] else "não separa"
                print(f"    {detector:14s} {agg['value']:+.3f} [{agg['low']:+.3f}, {agg['high']:+.3f}] ({flag})")
        print("  B3 − B1 (pareado por repetição, estratificado por família):")
        for detector, agg in block["B3_minus_B1"].items():
            if not np.isnan(agg["value"]):
                flag = "separa" if agg["excludes_zero"] else "não separa"
                print(f"    {detector:14s} {agg['value']:+.3f} [{agg['low']:+.3f}, {agg['high']:+.3f}] ({flag})")
    print(f"\n-> {E2_DIR / 'results.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--family", dest="families", action="append", choices=FAMILIES)
    run.add_argument("--detector", dest="detectors", action="append", choices=DETECTORS)
    run.add_argument("--force", action="store_true", help="refaz pares (família, detector) já gravados")
    sub.add_parser("report")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)
    if args.command == "run":
        args.families = args.families or list(FAMILIES)
        args.detectors = args.detectors or list(DETECTORS)
        return cmd_run(args)
    cmd_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
