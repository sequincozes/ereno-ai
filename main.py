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
    SUGGESTED_CONFIG_PATH,
    TEMPERATURE,
    TOTAL_ITERATIONS,
)
from tools.ereno_runner import ErenoRunner
from tools.experiment_memory import ExperimentMemory
from tools.ids_evaluator import IdsEvaluator
from tools.json_loader import load_json, load_text, save_json, save_text
from tools.json_patch_applier import apply_patch_to_json
from tools.llm_response_parser import extract_patch_from_llm_response
from tools.results_logger import append_metrics_to_csv


def main() -> None:
    load_dotenv()

    base_prompt = load_text(PROMPT_PATH)
    attack_json = load_json(ATTACK_JSON_PATH)
    performance_results = load_json(PERFORMANCE_RESULTS_PATH)

    strategist = StrategistAgent(
        model_id=MODEL_ID,
        temperature=TEMPERATURE,
    )

    ereno = ErenoRunner(
        ereno_project_path=ERENO_PROJECT_PATH,
        attack_config_relative_path=ERENO_ATTACK_CONFIG_RELATIVE_PATH,
        output_dataset_path=ERENO_OUTPUT_DATASET_PATH,
        run_command=ERENO_RUN_COMMAND,
        suggested_config_path=str(SUGGESTED_CONFIG_PATH),
    )

    evaluator = IdsEvaluator(drop_cb_status=False)
    memory = ExperimentMemory()

    print("\n==============================")
    print("BASELINE TRAINING")
    print("==============================")

    baseline_dataset_path = ereno.generate_dataset(
        attack_config=attack_json,
        iteration=0,
    )

    baseline_metrics = evaluator.train_baseline(baseline_dataset_path)
    performance_results = baseline_metrics

    append_metrics_to_csv(
        csv_path=METRICS_CSV_PATH,
        iteration=0,
        metrics=baseline_metrics,
    )

    print(f"[BASELINE] Modelo treinado no baseline: {baseline_dataset_path}")

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        print("\n==============================")
        print(f"ITERATION {iteration}")
        print("==============================")

        llm_response = strategist.suggest_changes(
            base_prompt=base_prompt,
            attack_json=attack_json,
            performance_results=performance_results,
            history=memory.get_history()[-1:],
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

        config_changed = updated_attack_json != attack_json

        if config_changed:
            print("[MAIN] Configuração alterada nesta iteração.")
        else:
            print("[MAIN] Configuração NÃO mudou nesta iteração.")

        attack_json = updated_attack_json

        iteration_attack_json_path = (
            ATTACK_CONFIGS_DIR / f"attack_config_iteration_{iteration}.json"
        )
        save_json(iteration_attack_json_path, attack_json)

        print(f"[JSON] Configuração da iteração salva em: {iteration_attack_json_path}")

        dataset_path = ereno.generate_dataset(
            attack_config=attack_json,
            iteration=iteration,
        )

        metrics = evaluator.evaluate_variant(dataset_path)
        metrics["config_changed"] = config_changed

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


if __name__ == "__main__":
    main()