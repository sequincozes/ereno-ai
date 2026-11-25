"""E0 — validade do gerador campo a campo, no protocolo por seed.

Para cada família e cada campo editável, gera a variante "só este campo" (nos
sentidos de intensidade que o catálogo admite) e a config de referência nas
mesmas seeds, e classifica o campo como inerte / efetivo / inconclusivo pela
mudança no tráfego gerado. Sem LLM e sem detector.

    uv run python scripts/run_e0_sensitivity.py
    uv run python scripts/run_e0_sensitivity.py --attack masquerade_fault --seeds 501 502

Saída: ``outputs/experiments/e0/sensitivity.json`` e ``sensitivity.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversarial_ids.config.settings import OUTPUTS_DIR  # noqa: E402
from adversarial_ids.experiments.sensitivity import (  # noqa: E402
    attack_features,
    classify,
    content_hash,
    field_variants,
    moved_features,
    noise_floor,
    reference_config,
    separation,
)
from adversarial_ids.experiments.traces import TraceError, TraceStore  # noqa: E402
from adversarial_ids.shared.json_io import save_json  # noqa: E402

FAMILIES = (
    "masquerade_fault",
    "random_replay",
    "inverse_replay",
    "injection",
    "high_stnum",
    "grayhole",
)
OUTPUT_DIR = OUTPUTS_DIR / "experiments" / "e0"


def run_family(attack_key: str, store: TraceStore, seeds: list[int]) -> list[dict]:
    reference = reference_config(attack_key)
    ref_traces = {s: store.trace(attack_key, reference, s) for s in seeds}
    ref_hash = {s: content_hash(p) for s, p in ref_traces.items()}
    ref_attack = {s: attack_features(p)[0] for s, p in ref_traces.items()}
    noise = noise_floor(list(ref_traces.values()))

    rows = []
    for variant in field_variants(attack_key):
        row: dict = {
            "attack": attack_key,
            "field": variant.path,
            "direction": variant.direction,
            "diff": list(variant.diff),
        }
        if variant.config is None:
            rows.append({**row, "status": "compile_failed", "error": variant.error})
            continue

        identical, moved, prevalence, errors = [], [], [], []
        for seed in seeds:
            try:
                path = store.trace(attack_key, variant.config, seed)
            except TraceError as error:
                errors.append(str(error).splitlines()[-1][:300])
                continue
            identical.append(content_hash(path) == ref_hash[seed])
            features, share = attack_features(path)
            prevalence.append(share)
            moved.append(moved_features(separation(features, ref_attack[seed]), noise))

        if errors:
            rows.append({**row, "status": "generation_failed", "error": errors[0]})
            continue
        common = sorted(set.intersection(*(set(m) for m in moved))) if moved else []
        rows.append(
            {
                **row,
                "status": classify(identical, moved),
                "identical_per_seed": identical,
                "moved_per_seed": moved,
                "moved_in_all_seeds": common,
                "attack_prevalence": prevalence,
            }
        )
        print(f"  {attack_key}.{variant.path} [{variant.direction}] -> {rows[-1]['status']}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack", "field", "direction", "status", "changed_to", "moved_in_all_seeds", "min_prevalence"])
        for row in rows:
            writer.writerow(
                [
                    row["attack"],
                    row["field"],
                    row["direction"],
                    row["status"],
                    "; ".join(f"{d['path']}={d['new_value']}" for d in row["diff"]),
                    " ".join(row.get("moved_in_all_seeds", [])),
                    min(row["attack_prevalence"]) if row.get("attack_prevalence") else "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attack", nargs="+", default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[501, 502, 503])
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    store = TraceStore()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict] = []
    for attack_key in args.attack:
        print(f"[E0] {attack_key}", flush=True)
        rows.extend(run_family(attack_key, store, args.seeds))
        save_json(str(out / "sensitivity.json"), {"seeds": args.seeds, "jar_sha256": store.jar_sha, "rows": rows})
        write_csv(out / "sensitivity.csv", rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[E0] {len(rows)} variantes em {time.perf_counter() - started:.0f}s: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
