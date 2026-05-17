import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from configs.settings import (
    MODEL_IDS,
    OUTPUTS_DIR,
    SLEEP_BETWEEN_MODELS_SECONDS,
    TOTAL_ITERATIONS,
)


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
RUN_LOG_PATH = OUTPUTS_DIR / "run_all_models_log.txt"


def safe_model_name(model_id: str) -> str:
    return (
        model_id
        .replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
    )


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RUN_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def clean_current_outputs() -> None:
    files_to_remove = [
        OUTPUTS_DIR / "metrics_history.csv",
        OUTPUTS_DIR / "iteration_history.json",
        OUTPUTS_DIR / "attack_config_changes.csv",
        OUTPUTS_DIR / "llm_response.txt",
        OUTPUTS_DIR / "suggested_attack_config.json",
    ]

    for path in files_to_remove:
        if path.exists():
            path.unlink()

    dirs_to_remove = [
        OUTPUTS_DIR / "llm_responses",
        OUTPUTS_DIR / "attack_configs",
    ]

    for path in dirs_to_remove:
        if path.exists():
            shutil.rmtree(path)

    for dataset_file in OUTPUTS_DIR.glob("dataset_iteration_*.csv"):
        dataset_file.unlink()


def restore_baseline_json() -> None:
    source = BASE_DIR / "inputs" / "uc03_masquerade_fault.json"
    target = BASE_DIR / "ERENO-2.0" / "config" / "attacks" / "uc03_masquerade_fault.json"

    if not source.exists():
        raise FileNotFoundError(f"JSON baseline não encontrado: {source}")

    shutil.copy(source, target)


def save_experiment_outputs(model_id: str, status: str) -> None:
    model_dir = EXPERIMENTS_DIR / safe_model_name(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    status_path = model_dir / "status.txt"

    with open(status_path, "w", encoding="utf-8") as file:
        file.write(f"model_id={model_id}\n")
        file.write(f"status={status}\n")
        file.write(f"timestamp={datetime.now().isoformat()}\n")

    files_to_copy = [
        "metrics_history.csv",
        "iteration_history.json",
        "attack_config_changes.csv",
        "llm_response.txt",
        "suggested_attack_config.json",
    ]

    for filename in files_to_copy:
        source = OUTPUTS_DIR / filename

        if source.exists():
            shutil.copy(source, model_dir / filename)

    dirs_to_copy = [
        "llm_responses",
        "attack_configs",
    ]

    for dirname in dirs_to_copy:
        source = OUTPUTS_DIR / dirname
        target = model_dir / dirname

        if source.exists():
            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(source, target)

    datasets_dir = model_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    for dataset_file in OUTPUTS_DIR.glob("dataset_iteration_*.csv"):
        shutil.copy(dataset_file, datasets_dir / dataset_file.name)


def run_compare_configs() -> None:
    compare_script = BASE_DIR / "tools" / "compare_attack_configs.py"

    if not compare_script.exists():
        log("[WARN] tools/compare_attack_configs.py não existe. Pulando comparação.")
        return

    result = subprocess.run(
        ["python", "tools/compare_attack_configs.py"],
        cwd=BASE_DIR,
        text=True,
    )

    if result.returncode != 0:
        log("[WARN] Comparação de configs falhou, mas vou continuar.")


def run_model(model_id: str, iterations: int) -> None:
    log("=" * 80)
    log(f"Iniciando modelo: {model_id}")
    log(f"Iterações: {iterations}")
    log("=" * 80)

    clean_current_outputs()
    restore_baseline_json()

    command = [
        "python",
        "main.py",
        "--model-id",
        model_id,
        "--iterations",
        str(iterations),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            text=True,
        )

        if result.returncode == 0:
            status = "completed"
            log(f"[OK] Modelo finalizado: {model_id}")
        else:
            status = f"failed_return_code_{result.returncode}"
            log(f"[ERRO] Modelo falhou com return code {result.returncode}: {model_id}")

    except KeyboardInterrupt:
        status = "cancelled_by_user"
        log(f"[CANCELADO] Execução interrompida pelo usuário no modelo: {model_id}")

    except Exception as error:
        status = f"exception_{type(error).__name__}"
        log(f"[EXCEPTION] Modelo {model_id} falhou: {error}")

    run_compare_configs()
    save_experiment_outputs(model_id=model_id, status=status)

    log(f"[SAVE] Outputs salvos para modelo: {model_id}")


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    log("Iniciando execução de todos os modelos.")
    log(f"Modelos: {MODEL_IDS}")
    log(f"Total de iterações por modelo: {TOTAL_ITERATIONS}")

    for index, model_id in enumerate(MODEL_IDS, start=1):
        log(f"Modelo {index}/{len(MODEL_IDS)}")

        run_model(
            model_id=model_id,
            iterations=TOTAL_ITERATIONS,
        )

        if index < len(MODEL_IDS):
            log(f"Aguardando {SLEEP_BETWEEN_MODELS_SECONDS}s antes do próximo modelo...")
            time.sleep(SLEEP_BETWEEN_MODELS_SECONDS)

    log("Execução de todos os modelos concluída.")
    log(f"Resultados em: {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()