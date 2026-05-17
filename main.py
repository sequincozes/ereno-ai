import argparse
import time

from dotenv import load_dotenv

from agents.strategist_agent import StrategistAgent
from configs.settings import (
    ATTACK_CONFIGS_DIR,
    ATTACK_JSON_PATH,
    ERENO_ATTACK_CONFIG_RELATIVE_PATH,
    ERENO_OUTPUT_DATASET_PATH,
    ERENO_PROJECT_PATH,
    ERENO_RUN_COMMAND,
    ITERATION_HISTORY_PATH,
    LLM_RESPONSE_PATH,
    LLM_RESPONSES_DIR,
    METRICS_CSV_PATH,
    MODEL_ID,
    PERFORMANCE_RESULTS_PATH,
    PROMPT_PATH,
    SLEEP_BETWEEN_ITERATIONS_SECONDS,
    SUGGESTED_CONFIG_PATH,
    TEMPERATURE,
    TOTAL_ITERATIONS,
)
from tools.attack_config_validator import validate_and_clamp_attack_config
from tools.ereno_runner import ErenoRunner
from tools.experiment_memory import ExperimentMemory
from tools.ids_evaluator import IdsEvaluator
from tools.json_loader import load_json, load_text, save_json, save_text
from tools.json_patch_applier import apply_patch_to_json
from tools.llm_response_parser import extract_patch_from_llm_response
from tools.results_logger import append_metrics_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial LLM-agent experiment with ERENO and Random Forest."
    )

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Groq model ID to use for this run.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=TOTAL_ITERATIONS,
        help="Number of adversarial iterations to run.",
    )

    return parser.parse_args()


def build_skipped_metrics() -> dict:
    return {
        "evaluation_type": "skipped_no_config_change",
        "dataset_path": None,
        "accuracy": None,
        "precision_masquerade": None,
        "recall_masquerade": None,
        "f1_score_masquerade": None,
        "support_masquerade": None,
        "attack_count": None,
        "normal_count": None,
        "full_attack_count": None,
        "full_normal_count": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
        "attack_label_original": "masquerade_fake_fault",
        "label_column": "class",
        "dataset_rows": None,
        "dataset_columns": None,
        "config_changed": False,
        "attack_count_ratio_vs_baseline": None,
        "degenerate_variant": None,
    }


def main() -> None:
    load_dotenv()

    args = parse_args()
    model_id = args.model_id
    total_iterations = args.iterations

    print("\n==============================")
    print("EXPERIMENT CONFIG")
    print("==============================")
    print(f"Model: {model_id}")
    print(f"Iterations: {total_iterations}")
    print(f"Temperature: {TEMPERATURE}")

    base_prompt = load_text(PROMPT_PATH)

    baseline_attack_json = load_json(ATTACK_JSON_PATH)
    attack_json = load_json(ATTACK_JSON_PATH)

    # Esse arquivo é usado apenas como fallback inicial.
    # Depois do baseline, performance_results passa a ser atualizado pelas métricas reais.
    performance_results = load_json(PERFORMANCE_RESULTS_PATH)

    strategist = StrategistAgent(
        model_id=model_id,
        temperature=TEMPERATURE,
    )

    ereno = ErenoRunner(
        ereno_project_path=ERENO_PROJECT_PATH,
        attack_config_relative_path=ERENO_ATTACK_CONFIG_RELATIVE_PATH,
        output_dataset_path=ERENO_OUTPUT_DATASET_PATH,
        run_command=ERENO_RUN_COMMAND,
        suggested_config_path=str(SUGGESTED_CONFIG_PATH),
    )

    evaluator = IdsEvaluator(
        drop_cb_status=False,
    )

    memory = ExperimentMemory()

    print("\n==============================")
    print("BASELINE TRAINING")
    print("==============================")

    baseline_dataset_path = ereno.generate_dataset(
        attack_config=baseline_attack_json,
        iteration=0,
    )

    baseline_metrics = evaluator.train_baseline(baseline_dataset_path)

    baseline_metrics["config_changed"] = False
    baseline_metrics["attack_count_ratio_vs_baseline"] = 1.0
    baseline_metrics["degenerate_variant"] = False

    performance_results = baseline_metrics

    append_metrics_to_csv(
        csv_path=METRICS_CSV_PATH,
        iteration=0,
        metrics=baseline_metrics,
    )

    print(f"[BASELINE] Modelo treinado no baseline: {baseline_dataset_path}")

    baseline_attack_count = (
        baseline_metrics.get("full_attack_count")
        or baseline_metrics.get("attack_count")
        or baseline_metrics.get("support_masquerade")
    )

    if not baseline_attack_count:
        raise RuntimeError(
            "Não foi possível determinar o total de ataques no baseline."
        )

    print(f"[BASELINE] Total de ataques no baseline completo: {baseline_attack_count}")

    for iteration in range(1, total_iterations + 1):
        print("\n==============================")
        print(f"ITERATION {iteration}")
        print("==============================")

        llm_response = strategist.suggest_changes(
            base_prompt=base_prompt,
            attack_json=attack_json,
            performance_results=performance_results,
            history=memory.get_history(),
        )

        save_text(LLM_RESPONSE_PATH, llm_response)

        iteration_llm_response_path = (
            LLM_RESPONSES_DIR / f"llm_response_iteration_{iteration}.txt"
        )

        save_text(iteration_llm_response_path, llm_response)

        print(f"[LLM] Resposta salva em: {iteration_llm_response_path}")

        patch = extract_patch_from_llm_response(llm_response)

        if patch:
            updated_attack_json = apply_patch_to_json(
                original_json=attack_json,
                patch=patch,
            )
        else:
            print("[MAIN] Nenhum patch válido. Mantendo configuração anterior.")
            updated_attack_json = attack_json

        updated_attack_json, validation_warnings = validate_and_clamp_attack_config(
            candidate_config=updated_attack_json,
            baseline_config=baseline_attack_json,
        )

        for warning in validation_warnings:
            print(f"[VALIDATOR] {warning}")

        config_changed = updated_attack_json != attack_json

        if config_changed:
            print("[MAIN] Configuração alterada nesta iteração.")
        else:
            print("[MAIN] Configuração NÃO mudou nesta iteração. Pulando ERENO.")

        attack_json = updated_attack_json

        iteration_attack_json_path = (
            ATTACK_CONFIGS_DIR / f"attack_config_iteration_{iteration}.json"
        )

        save_json(iteration_attack_json_path, attack_json)

        print(f"[JSON] Configuração da iteração salva em: {iteration_attack_json_path}")

        if not config_changed:
            skipped_metrics = build_skipped_metrics()

            append_metrics_to_csv(
                csv_path=METRICS_CSV_PATH,
                iteration=iteration,
                metrics=skipped_metrics,
            )

            memory.add_iteration(
                iteration=iteration,
                attack_json=attack_json,
                metrics=skipped_metrics,
                llm_response=llm_response,
            )

            memory.save(str(ITERATION_HISTORY_PATH))
            print(f"[MEMORY] Histórico salvo em: {ITERATION_HISTORY_PATH}")

            if SLEEP_BETWEEN_ITERATIONS_SECONDS:
                print(
                    f"[WAIT] Aguardando {SLEEP_BETWEEN_ITERATIONS_SECONDS}s "
                    "antes da próxima iteração..."
                )
                time.sleep(SLEEP_BETWEEN_ITERATIONS_SECONDS)

            continue

        dataset_path = ereno.generate_dataset(
            attack_config=attack_json,
            iteration=iteration,
        )

        metrics = evaluator.evaluate_variant(dataset_path)
        metrics["config_changed"] = True

        current_attack_count = (
            metrics.get("attack_count")
            or metrics.get("support_masquerade")
        )

        if baseline_attack_count and current_attack_count:
            attack_ratio = current_attack_count / baseline_attack_count
        else:
            attack_ratio = None

        metrics["attack_count_ratio_vs_baseline"] = attack_ratio

        if attack_ratio is not None and attack_ratio < 0.5:
            metrics["degenerate_variant"] = True
            print(
                "[VALIDATOR] Variante marcada como degenerada: "
                f"attack_count caiu para {attack_ratio:.2%} do baseline."
            )
        else:
            metrics["degenerate_variant"] = False

        performance_results = metrics

        append_metrics_to_csv(
            csv_path=METRICS_CSV_PATH,
            iteration=iteration,
            metrics=metrics,
        )

        print(f"[CSV] Métricas salvas em: {METRICS_CSV_PATH}")

        memory.add_iteration(
            iteration=iteration,
            attack_json=attack_json,
            metrics=metrics,
            llm_response=llm_response,
        )

        memory.save(str(ITERATION_HISTORY_PATH))
        print(f"[MEMORY] Histórico salvo em: {ITERATION_HISTORY_PATH}")

        if SLEEP_BETWEEN_ITERATIONS_SECONDS:
            print(
                f"[WAIT] Aguardando {SLEEP_BETWEEN_ITERATIONS_SECONDS}s "
                "antes da próxima iteração..."
            )
            time.sleep(SLEEP_BETWEEN_ITERATIONS_SECONDS)


if __name__ == "__main__":
    main()