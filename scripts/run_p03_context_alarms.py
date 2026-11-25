"""P0-3 da revisão de 20/11/2026: falsos alertas limpos vs. após injeção.

As features de delta de uma mensagem normal (stDiff, sqDiff, tDiff...) são
calculadas contra a mensagem anterior do fluxo, que pode ser injetada. A taxa
de falsos alertas do E1/E3 mistura, então, alarmes em tráfego limpo com
alarmes de contexto do ataque. Este script separa os dois: cada mensagem normal
recebe a distância, em mensagens, até a última injeção anterior a ela, e os
falsos alertas no orçamento primário (10/10k) são contados por estrato.

Estratos (distância d em mensagens desde a última injeção):
    d1      a mensagem imediatamente seguinte a uma injeção
    d2_3    2 ou 3 mensagens depois
    d4_10   4 a 10 mensagens depois
    clean   mais de 10 mensagens depois, ou nenhuma injeção antes dela

Nada é gerado nem treinado com dado novo: só traces já em cache e detectores
retreinados a partir do cache (determinístico). Trace ausente do cache é
registrado como ``missing`` e pulado, nunca gerado. Sem LLM, sem JAR.

    # 1. E1: D0 congelado de cada família × referência e selecionados (seeds de teste)
    uv run python scripts/run_p03_context_alarms.py e1 2>&1 | tee -a outputs/experiments/p03/e1.log

    # 2. E3: cada variante A_r antes (D_r) e depois (D_{r+1}) do retreino
    uv run python scripts/run_p03_context_alarms.py e3 2>&1 | tee -a outputs/experiments/p03/e3.log

    # 3. tabelas
    uv run python scripts/run_p03_context_alarms.py report

``e1`` e ``e3`` são retomáveis: rodar de novo continua de onde parou. Análise
adicionada na revisão, não pré-especificada no protocolo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_e1 import E1_DIR, FAMILIES, _dump  # noqa: E402

from adversarial_ids.config.settings import OUTPUTS_DIR  # noqa: E402
from adversarial_ids.experiments.arms_race import ROUNDS, DetectorLineage, IncompleteRound, Lineage  # noqa: E402
from adversarial_ids.experiments.metrics import ALERTS_PER, is_saturated, stratified_paired_difference  # noqa: E402
from adversarial_ids.experiments.protocol import FrozenIds  # noqa: E402
from adversarial_ids.experiments.search import PRIMARY_BUDGET, read_ledger, select_best  # noqa: E402
from adversarial_ids.experiments.sensitivity import reference_config  # noqa: E402
from adversarial_ids.experiments.traces import TraceStore  # noqa: E402

OUT = OUTPUTS_DIR / "experiments" / "p03"
E3_DIR = OUTPUTS_DIR / "experiments" / "e3"
E1_STRATEGIES = ("B1", "B2", "B3", "B4")
STRATA = ("d1", "d2_3", "d4_10", "clean")


# --------------------------------------------------------------------------- #
# Estratificação                                                              #
# --------------------------------------------------------------------------- #
def distance_since_injection(is_attack: np.ndarray) -> np.ndarray:
    """Mensagens desde a última injeção anterior (na ordem do fluxo); ``inf`` se nenhuma."""

    attack = np.asarray(is_attack, dtype=bool)
    idx = np.arange(attack.size)
    last = np.where(attack, idx, -1)
    last = np.maximum.accumulate(last)
    dist = (idx - last).astype(float)
    dist[last < 0] = np.inf
    return dist


def stratum_masks(dist: np.ndarray, normal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "d1": normal & (dist == 1),
        "d2_3": normal & (dist >= 2) & (dist <= 3),
        "d4_10": normal & (dist >= 4) & (dist <= 10),
        "clean": normal & (dist > 10),
    }


def stratified_counts(frozen: FrozenIds, path: Path) -> dict:
    scored = frozen.evaluator.score_trace(path)
    attack = np.asarray(scored["is_attack"], dtype=bool)
    point = next(p for p in frozen.points if str(p.budget_per_10k) == PRIMARY_BUDGET)
    if is_saturated(point):
        return {"saturated": True}
    alarm = np.asarray(scored["scores"]) >= point.threshold
    masks = stratum_masks(distance_since_injection(attack), ~attack)
    return {
        "saturated": False,
        "attack_rows": int(attack.sum()),
        "strata": {name: {"n": int(m.sum()), "fp": int((alarm & m).sum())} for name, m in masks.items()},
    }


# --------------------------------------------------------------------------- #
# Registro retomável                                                          #
# --------------------------------------------------------------------------- #
def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _measure(frozen, store, family, config, seed, record, out, done) -> None:
    if record["key"] in done:
        return
    path = store.path_for(family, config, seed)
    if not path.exists():
        _append(out, {**record, "missing": True})
        print(f"  {record['key']}: trace fora do cache, pulado", flush=True)
        return
    t0 = time.time()
    _append(out, {**record, **stratified_counts(frozen, path)})
    done.add(record["key"])
    print(f"  {record['key']} ({time.time() - t0:.1f}s)", flush=True)


# --------------------------------------------------------------------------- #
# E1                                                                          #
# --------------------------------------------------------------------------- #
def cmd_e1(args) -> None:
    store, out = TraceStore(), OUT / "e1.jsonl"
    done = _done(out)
    for family in args.families or FAMILIES:
        reference = reference_config(family)
        jobs = [("B0", 0, reference)]
        for strategy in E1_STRATEGIES:
            for ledger in sorted((E1_DIR / "ledger" / family / strategy).glob("rep*.jsonl")):
                best = select_best(read_ledger(ledger))
                if best is not None:
                    jobs.append((strategy, int(ledger.stem[3:]), best["config"]))
        seeds = (401, 402, 403, 404, 405)
        keys = [f"e1/{family}/{s}/rep{r}/seed{seed}" for s, r, _ in jobs for seed in seeds]
        if all(k in done for k in keys):
            continue
        print(f"[{family}] treinando D0 (cache)... {len(jobs)} configs x {len(seeds)} seeds", flush=True)
        frozen = FrozenIds(family, reference, store).fit()
        for strategy, rep, config in jobs:
            for seed in seeds:
                record = {"key": f"e1/{family}/{strategy}/rep{rep}/seed{seed}", "family": family,
                          "strategy": strategy, "rep": rep, "seed": seed}
                _measure(frozen, store, family, config, seed, record, out, done)


# --------------------------------------------------------------------------- #
# E3                                                                          #
# --------------------------------------------------------------------------- #
def cmd_e3(args) -> None:
    store, out = TraceStore(), OUT / "e3.jsonl"
    done = _done(out)
    for family in args.families or FAMILIES:
        for rep in args.reps or (1, 2, 3, 4, 5):
            lineage = Lineage(family, args.arm, rep)
            dl = DetectorLineage(lineage, reference_config(family), store, E1_DIR, E3_DIR)
            try:
                selected = [select_best(dl.completed_entries(r)) for r in range(ROUNDS)]
            except IncompleteRound as error:
                print(f"[{family} {args.arm} rep{rep}] pulada: {error}", flush=True)
                continue
            seeds = dl.seeds.test
            tag = f"e3/{family}/{args.arm}/rep{rep}"
            # D_i mede A_{i-1} (depois do retreino que o incluiu) e A_i (antes);
            # D0 e D_R também medem a referência (custo do retreino no tráfego da família).
            for index in range(ROUNDS + 1):
                todo = []
                if index > 0 and selected[index - 1] is not None:
                    todo.append(("after", index - 1, selected[index - 1]["config"]))
                if index < ROUNDS and selected[index] is not None:
                    todo.append(("before", index, selected[index]["config"]))
                if index in (0, ROUNDS):
                    todo.append(("reference", -1, dl.reference))
                keys = [f"{tag}/D{index}/{when}/A{r}/seed{s}" for when, r, _ in todo for s in seeds]
                if all(k in done for k in keys):
                    continue
                print(f"[{family} {args.arm} rep{rep}] treinando D{index} (cache)...", flush=True)
                frozen = dl.detector(index)
                for when, race_round, config in todo:
                    for seed in seeds:
                        record = {"key": f"{tag}/D{index}/{when}/A{race_round}/seed{seed}", "family": family,
                                  "arm": args.arm, "rep": rep, "detector": index, "when": when,
                                  "attack_round": race_round, "seed": seed}
                        _measure(frozen, store, family, config, seed, record, out, done)


# --------------------------------------------------------------------------- #
# Relatório                                                                   #
# --------------------------------------------------------------------------- #
def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not r.get("missing") and not r.get("saturated")]


def _pool(rows: list[dict]) -> dict:
    """Taxas por 10k normais de cada estrato, com as contagens somadas."""

    out = {}
    total_fp = sum(r["strata"][s]["fp"] for r in rows for s in STRATA)
    for s in STRATA:
        n = sum(r["strata"][s]["n"] for r in rows)
        fp = sum(r["strata"][s]["fp"] for r in rows)
        out[s] = {"n": n, "fp": fp, "per_10k": fp / n * ALERTS_PER if n else None,
                  "share_of_normals": None, "share_of_false_alerts": fp / total_fp if total_fp else None}
    normals = sum(out[s]["n"] for s in STRATA)
    for s in STRATA:
        out[s]["share_of_normals"] = out[s]["n"] / normals if normals else None
    fp_all = sum(out[s]["fp"] for s in STRATA)
    out["all"] = {"n": normals, "fp": fp_all, "per_10k": fp_all / normals * ALERTS_PER if normals else None}
    return out


def _rate(rows: list[dict], stratum: str) -> float:
    n = sum(r["strata"][stratum]["n"] for r in rows)
    fp = sum(r["strata"][stratum]["fp"] for r in rows)
    return fp / n * ALERTS_PER if n else float("nan")


def _fmt(v) -> str:
    return "     —" if v is None else f"{v:6.1f}"


def cmd_report(_args) -> None:
    report: dict = {"budget": PRIMARY_BUDGET, "strata": STRATA, "e1": {}, "e3": {}}

    e1 = _read(OUT / "e1.jsonl")
    print(f"E1: falsos alertas/10k normais no orçamento {PRIMARY_BUDGET}, por estrato (D0, seeds de teste)")
    print(f"  {'família':18s} {'estr.':5s} " + " ".join(f"{s:>6s}" for s in (*STRATA, "all")) + "   %FA em d1  %normais em d1")
    for family in FAMILIES:
        for strategy in ("B0", *E1_STRATEGIES):
            rows = [r for r in e1 if r["family"] == family and r["strategy"] == strategy]
            if not rows:
                continue
            pooled = _pool(rows)
            report["e1"].setdefault(family, {})[strategy] = {"traces": len(rows), **pooled}
            d1 = pooled["d1"]
            print(f"  {family:18s} {strategy:5s} " + " ".join(_fmt(pooled[s]["per_10k"]) for s in (*STRATA, "all"))
                  + f"   {(d1['share_of_false_alerts'] or 0):8.0%}  {(d1['share_of_normals'] or 0):8.0%}")

    e3 = _read(OUT / "e3.jsonl")
    if e3:
        print("\nE3: falsos alertas/10k no normal do trace de A_r, antes (D_r) e depois (D_{r+1}) do retreino")
        print(f"  {'família':18s} {'quando':9s} " + " ".join(f"{s:>6s}" for s in (*STRATA, "all")))
        first: dict = {s: {} for s in (*STRATA, "all")}
        second: dict = {s: {} for s in (*STRATA, "all")}
        for family in FAMILIES:
            fam = [r for r in e3 if r["family"] == family]
            if not fam:
                continue
            for when in ("before", "after"):
                pooled = _pool([r for r in fam if r["when"] == when])
                report["e3"].setdefault(family, {})[when] = pooled
                print(f"  {family:18s} {when:9s} " + " ".join(_fmt(pooled[s]["per_10k"]) for s in (*STRATA, "all")))
            for det in (0, ROUNDS):
                rows = [r for r in fam if r["when"] == "reference" and r["detector"] == det]
                if rows:
                    pooled = _pool(rows)
                    report["e3"][family][f"reference_D{det}"] = pooled
                    print(f"  {family:18s} {'ref D' + str(det):9s} " + " ".join(_fmt(pooled[s]["per_10k"]) for s in (*STRATA, "all")))
            # Pareado por (rep, rodada): depois − antes, no mesmo trace de A_r.
            pairs = sorted({(r["rep"], r["attack_round"]) for r in fam if r["when"] in ("before", "after")})
            for s in (*STRATA, "all"):
                a, b = [], []
                for rep, rnd in pairs:
                    after = [r for r in fam if r["when"] == "after" and r["rep"] == rep and r["attack_round"] == rnd]
                    before = [r for r in fam if r["when"] == "before" and r["rep"] == rep and r["attack_round"] == rnd]
                    if after and before:
                        a.append(_rate_all(after, s))
                        b.append(_rate_all(before, s))
                if a:
                    first[s][family], second[s][family] = a, b
        report["e3"]["after_minus_before"] = {}
        print("\n  depois − antes (pareado por variante, estratificado por família, IC 95% bootstrap):")
        for s in (*STRATA, "all"):
            if not first[s]:
                continue
            diff = stratified_paired_difference(first[s], second[s]).as_dict()
            report["e3"]["after_minus_before"][s] = diff
            flag = "separa" if diff["excludes_zero"] else "não separa"
            print(f"    {s:6s} {diff['value']:+8.1f} [{diff['low']:+8.1f}, {diff['high']:+8.1f}] ({flag})")

    _dump(OUT / "results.json", report)
    print(f"\n-> {OUT / 'results.json'}")


def _rate_all(rows: list[dict], stratum: str) -> float:
    if stratum != "all":
        return _rate(rows, stratum)
    n = sum(r["strata"][s]["n"] for r in rows for s in STRATA)
    fp = sum(r["strata"][s]["fp"] for r in rows for s in STRATA)
    return fp / n * ALERTS_PER if n else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p1 = sub.add_parser("e1")
    p1.add_argument("--families", nargs="*", choices=FAMILIES)
    p3 = sub.add_parser("e3")
    p3.add_argument("--families", nargs="*", choices=FAMILIES)
    p3.add_argument("--reps", nargs="*", type=int)
    p3.add_argument("--arm", default="B1", choices=("B1", "B3"))
    sub.add_parser("report")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parser.parse_args(argv)
    {"e1": cmd_e1, "e3": cmd_e3, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
