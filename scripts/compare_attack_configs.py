import csv
import json
from pathlib import Path
from typing import Any


BASE_CONFIG = Path("inputs/uc03_masquerade_fault.json")
CONFIGS_DIR = Path("outputs/attack_configs")
OUTPUT_CSV = Path("outputs/attack_config_changes.csv")


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def flatten_json(data: Any, parent_key: str = "") -> dict[str, Any]:
    items = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            items.update(flatten_json(value, new_key))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            new_key = f"{parent_key}[{index}]"
            items.update(flatten_json(value, new_key))

    else:
        items[parent_key] = data

    return items


def compare_configs(
    previous_config: dict[str, Any],
    current_config: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    previous_flat = flatten_json(previous_config)
    current_flat = flatten_json(current_config)

    all_keys = sorted(set(previous_flat.keys()) | set(current_flat.keys()))
    changes = []

    for key in all_keys:
        previous_value = previous_flat.get(key)
        current_value = current_flat.get(key)

        if previous_value != current_value:
            changes.append(
                {
                    "iteration": iteration,
                    "field": key,
                    "previous_value": previous_value,
                    "current_value": current_value,
                }
            )

    return changes


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Config original não encontrada: {BASE_CONFIG}")

    if not CONFIGS_DIR.exists():
        raise FileNotFoundError(f"Pasta de configs não encontrada: {CONFIGS_DIR}")

    config_files = sorted(
        CONFIGS_DIR.glob("attack_config_iteration_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )

    if not config_files:
        print("Nenhuma configuração de iteração encontrada.")
        return

    all_changes = []
    previous_config = load_json(BASE_CONFIG)

    for config_file in config_files:
        iteration = int(config_file.stem.split("_")[-1])
        current_config = load_json(config_file)

        changes = compare_configs(
            previous_config=previous_config,
            current_config=current_config,
            iteration=iteration,
        )

        all_changes.extend(changes)
        previous_config = current_config

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "iteration",
            "field",
            "previous_value",
            "current_value",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_changes)

    print(f"CSV de mudanças salvo em: {OUTPUT_CSV}")
    print(f"Total de mudanças encontradas: {len(all_changes)}")

    if len(all_changes) == 0:
        print("Nenhuma mudança detectada entre as configurações.")


if __name__ == "__main__":
    main()
