"""E6: custo por unidade de trabalho, lido do que o E1 e o E3 já gravaram.

Nada é rodado de novo. Tokens e chamadas vêm do registro de cada chamada nos
results (somado pelo ``campaign_result``); o tempo de uma proposta é o
``elapsed_s`` do ledger (geração + medição + latência do provedor). O retreino
e a matriz cruzada não gravam duração: são estimados pela diferença entre os
horários de gravação dos arquivos (mediana, porque execuções retomadas deixam
intervalos de horas). Saída: ``outputs/experiments/e6/results.json``.

    uv run python scripts/report_e6.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "experiments"
FAMILIES = ("masquerade_fault", "high_stnum", "inverse_replay", "random_replay")
# Intervalos acima disso são pausa (cota, máquina desligada), não trabalho.
MAX_GAP_MIN = 120.0


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _campaigns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [r["input_tokens"] + r["output_tokens"] for r in rows]
    proposals = sum(r["proposals"] for r in rows)
    elapsed = [r["elapsed_s"] for r in rows]
    return {
        "n": len(rows),
        "proposals_per_campaign": st.mean(r["proposals"] for r in rows),
        "llm_calls_per_campaign": st.mean(r["llm_calls"] for r in rows),
        "tokens_per_campaign": st.mean(tokens),
        "tokens_min": min(tokens),
        "tokens_max": max(tokens),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "tokens_total": sum(tokens),
        "minutes_per_campaign": st.mean(elapsed) / 60,
        "seconds_per_proposal": sum(elapsed) / proposals,
        "validity": st.mean(r["validity"] for r in rows if r["validity"] == r["validity"]),
    }


def _gap_median(gaps: list[float]) -> dict[str, Any] | None:
    kept = [g for g in gaps if 0 < g < MAX_GAP_MIN]
    return {"n": len(kept), "median_min": st.median(kept)} if kept else None


def main() -> int:
    out: dict[str, Any] = {"e1": {}, "e3": {}, "e3_retrain": {}, "e3_round_after_retrain": {}, "e3_matrix": {}}

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((ROOT / "e1/results").glob("*/*/rep*.json")):
        d = _load(path)
        by[d["strategy"]].append(d)
    for strategy, rows in sorted(by.items()):
        out["e1"][strategy] = _campaigns(rows)

    by = defaultdict(list)
    for path in sorted((ROOT / "e3/results").glob("*/*/rep*/round*.json")):
        d = _load(path)
        by[d["lineage"]].append(d)
    for lineage, rows in sorted(by.items()):
        out["e3"][lineage] = _campaigns(rows)

    retrain: dict[str, list[float]] = defaultdict(list)
    after: dict[str, list[float]] = defaultdict(list)
    for rep in sorted((ROOT / "e3/ids").glob("*/*/rep*")):
        family, lineage = rep.parent.parent.name, rep.parent.name
        results = ROOT / "e3/results" / family / lineage / rep.name
        for r in (2, 3):  # D1 é treinado a partir do E1: não há arquivo anterior que marque o início
            detector, prev, cur = rep / f"D{r}.json", results / f"round{r - 1}.json", results / f"round{r}.json"
            if detector.exists() and prev.exists():
                retrain[lineage].append((detector.stat().st_mtime - prev.stat().st_mtime) / 60)
            if detector.exists() and cur.exists():
                after[lineage].append((cur.stat().st_mtime - detector.stat().st_mtime) / 60)
    matrix: dict[str, list[float]] = defaultdict(list)
    for lineage_dir in sorted((ROOT / "e3/matrix").glob("*/*")):
        stamps = sorted(p.stat().st_mtime for p in lineage_dir.glob("rep*.json"))
        matrix[lineage_dir.name] += [(b - a) / 60 for a, b in zip(stamps, stamps[1:])]
    for key, source in (("e3_retrain", retrain), ("e3_round_after_retrain", after), ("e3_matrix", matrix)):
        for lineage, gaps in sorted(source.items()):
            out[key][lineage] = _gap_median(gaps)

    plans: dict[str, Any] = {}
    for arm_dir in sorted((ROOT / "e3/plans").glob("*/*")):
        rows = [_load(p) for p in arm_dir.glob("rep*/round*.json")]
        tokens = [r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rows]
        plans[f"{arm_dir.parent.name}/{arm_dir.name}"] = {
            "plans": len(rows),
            "accepted": sum(bool(r.get("accepted")) for r in rows),
            "tokens_per_plan": st.mean(tokens) if tokens else None,
            "tokens_total": sum(tokens),
        }
    out["e3_plans"] = plans

    # Consultas ao Defensor no B4: novas (custam tokens do 20b) vs reusadas.
    fresh = reused = none = accepted = 0
    for path in sorted((ROOT / "e1/ledger").glob("*/B4/rep*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            defense = (json.loads(line).get("proposer") or {}).get("defense")
            if defense is None:
                continue
            if defense.get("source_index") is None:
                none += 1
            elif "reused_from" in defense:
                reused += 1
            else:
                fresh += 1
                accepted += bool(defense.get("accepted"))
    out["e1_b4_defender"] = {"fresh": fresh, "accepted": accepted, "reused": reused, "no_candidate": none}

    llm = [out["e1"].get(s, {}).get("tokens_total", 0) for s in ("B3", "B4")]
    llm += [out["e3"].get("B3", {}).get("tokens_total", 0)]
    llm += [p["tokens_total"] for p in plans.values()]
    out["llm_tokens_total"] = sum(llm)

    target = ROOT / "e6" / "results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("->", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
