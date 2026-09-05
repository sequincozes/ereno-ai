"""Gera o histórico golden ``data/iteration_history.json`` (issue #3).

Roda o loop adversarial em **modo cacheado (sem Java)** usando os stubs
determinísticos ``FakeStrategist`` / ``FakeAnalyst``. Serve a dois propósitos:

  1. Provar que o loop roda ponta a ponta sem o JAR do ERENO.
  2. Produzir uma fixture golden estável (timestamps e seeds fixos) que o
     Dashboard (#19) e os testes podem consumir sem chamar a API nem o Java.

Uso:  uv run python scripts/generate_golden_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from adversarial_ids.config.settings import (  # noqa: E402
    ATTACK_JSON_PATH,
    BASELINE_DATASET_PATH,
    GENERATOR_ATTACK_CONFIG_RELATIVE_PATH,
    GENERATOR_OUTPUT_DATASET_PATH,
    GENERATOR_RUNTIME_DIR,
    GENERATOR_RUN_COMMAND,
    GOLDEN_HISTORY_PATH,
    OUTPUTS_DIR,
)
from adversarial_ids.core.generator_runner import GeneratorRunner  # noqa: E402
from adversarial_ids.core.ids_evaluator import IdsEvaluator  # noqa: E402
from adversarial_ids.domain import IterationRecord, Metrics, MasqueradeFaultConfig, StrategistOutput  # noqa: E402
from adversarial_ids.shared.json_io import load_json, save_json  # noqa: E402
from adversarial_ids.shared.json_patch import apply_patch_to_json  # noqa: E402
from adversarial_ids.shared.validator import validate_and_clamp_attack_config  # noqa: E402
from tests.fakes import FakeAnalyst, FakeStrategist  # noqa: E402

GOLDEN_ITERATIONS = 3
# Timestamp base FIXO — mantém o golden byte-estável entre regenerações.
_FIXED_TS = "2026-01-01T00:00:{sec:02d}+00:00"


def _record(
    iteration: int,
    attack_config: dict,
    strategist_output: StrategistOutput | None,
    metrics: dict,
    analyst_output: dict | None,
) -> dict:
    """Valida via schemas de domain/ (#2) e devolve o dict do IterationRecord."""
    record = IterationRecord(
        iteration=iteration,
        attack_config=MasqueradeFaultConfig.model_validate(attack_config),
        strategist_output=strategist_output,
        metrics=Metrics.model_validate(metrics),
        analyst_output=analyst_output,
        timestamp=_FIXED_TS.format(sec=iteration),
    )
    return record.model_dump(mode="json")


def main() -> None:
    baseline_attack = load_json(ATTACK_JSON_PATH)
    attack_config = load_json(ATTACK_JSON_PATH)

    generator = GeneratorRunner(
        runtime_dir=GENERATOR_RUNTIME_DIR,
        attack_config_relative_path=GENERATOR_ATTACK_CONFIG_RELATIVE_PATH,
        output_dataset_path=GENERATOR_OUTPUT_DATASET_PATH,
        run_command=GENERATOR_RUN_COMMAND,
        suggested_config_path=str(OUTPUTS_DIR / "suggested_attack_config.json"),
        cached_dataset_path=BASELINE_DATASET_PATH,  # força modo cacheado
    )
    # Detector e escala fixados explicitamente (E8): o histórico golden precisa
    # ser byte-estável, e ``DETECTOR_MODE``/``DETECTOR_SCALER`` são env vars —
    # herdar o default deixaria o shell de quem regenera decidir qual modelo
    # entra na fixture versionada.
    evaluator = IdsEvaluator(drop_cb_status=False, detector="random_forest", scaler="none")
    strategist = FakeStrategist()
    analyst = FakeAnalyst()

    print(f"[golden] modo cacheado: {generator.is_cached} | seed=42")

    # --- iteração 0: baseline ---
    baseline_dataset = generator.generate_dataset(baseline_attack, iteration=0)
    baseline_metrics = evaluator.train_baseline(baseline_dataset)
    baseline_metrics["config_changed"] = False
    baseline_metrics["attack_count_ratio_vs_baseline"] = 1.0
    baseline_metrics["degenerate_variant"] = False

    history: list[dict] = [
        _record(
            iteration=0,
            attack_config=baseline_attack,
            strategist_output=None,
            metrics=baseline_metrics,
            analyst_output=analyst.analyze(0, baseline_metrics),
        )
    ]

    # --- iterações 1..N: fake strategist → variante → fake analyst ---
    for iteration in range(1, GOLDEN_ITERATIONS + 1):
        strategist_output = strategist.propose(iteration)
        patch = FakeStrategist.to_patch(strategist_output.model_dump())

        candidate = apply_patch_to_json(attack_config, patch)
        attack_config, _ = validate_and_clamp_attack_config(candidate, baseline_attack)

        dataset_path = generator.generate_dataset(attack_config, iteration)
        metrics = evaluator.evaluate_variant(dataset_path)
        metrics["config_changed"] = True
        metrics["attack_count_ratio_vs_baseline"] = 1.0
        metrics["degenerate_variant"] = False

        analyst_output = analyst.analyze(iteration, metrics)

        history.append(
            _record(
                iteration=iteration,
                attack_config=attack_config,
                strategist_output=strategist_output,
                metrics=metrics,
                analyst_output=analyst_output,
            )
        )
        print(f"[golden] iteração {iteration} ok (f1={metrics['f1_score_attack']:.4f})")

    save_json(GOLDEN_HISTORY_PATH, {"history": history})
    print(f"[golden] histórico salvo em: {GOLDEN_HISTORY_PATH} ({len(history)} registros)")


if __name__ == "__main__":
    main()
