import argparse
import shutil
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"


def safe_model_name(model_id: str) -> str:
    return (
        model_id
        .replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
    )


def clean_run_outputs() -> None:
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

    shutil.copy(source, target)


def save_experiment_outputs(model_id: str) -> None:
    model_dir = EXPERIMENTS_DIR / safe_model_name(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

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


def run_command(command: list[str]) -> int:
    process = subprocess.run(
        command,
        cwd=BASE_DIR,
        text=True,
    )

    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--iterations", type=int, default=None)

    args = parser.parse_args()

    model_id = args.model_id

    print("\n" + "=" * 80)
    print(f"RODANDO MODELO: {model_id}")
    print("=" * 80)

    clean_run_outputs()
    restore_baseline_json()

    command = [
        "python",
        "main.py",
        "--model-id",
        model_id,
    ]

    if args.iterations is not None:
        command.extend(["--iterations", str(args.iterations)])

    return_code = run_command(command)

    if return_code != 0:
        print(f"[ERRO] main.py falhou para modelo: {model_id}")
    else:
        print(f"[OK] main.py concluído para modelo: {model_id}")

    run_command(["python", "tools/compare_attack_configs.py"])

    save_experiment_outputs(model_id)

    print(f"[OK] Resultados salvos em: {EXPERIMENTS_DIR / safe_model_name(model_id)}")


if __name__ == "__main__":
    main()