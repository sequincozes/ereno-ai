import csv
from pathlib import Path
from typing import Any


def append_metrics_to_csv(
    csv_path: str | Path,
    iteration: int,
    metrics: dict[str, Any],
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "iteration",
        "evaluation_type",
        "dataset_path",
        "accuracy",
        "precision_masquerade",
        "recall_masquerade",
        "f1_score_masquerade",
        "support_masquerade",
        "attack_label_original",
        "label_column",
        "dataset_rows",
        "dataset_columns",
    ]

    row = {
        "iteration": iteration,
        "evaluation_type": metrics.get("evaluation_type"),
        "dataset_path": metrics.get("dataset_path"),
        "accuracy": metrics.get("accuracy"),
        "precision_masquerade": metrics.get("precision_masquerade"),
        "recall_masquerade": metrics.get("recall_masquerade"),
        "f1_score_masquerade": metrics.get("f1_score_masquerade"),
        "support_masquerade": metrics.get("support_masquerade"),
        "attack_label_original": metrics.get("attack_label_original"),
        "label_column": metrics.get("label_column"),
        "dataset_rows": metrics.get("dataset_rows"),
        "dataset_columns": metrics.get("dataset_columns"),
    }

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)