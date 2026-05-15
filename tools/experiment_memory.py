from typing import Any

from tools.json_loader import save_json


class ExperimentMemory:
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def add_iteration(
        self,
        iteration: int,
        attack_json: dict[str, Any],
        metrics: dict[str, Any],
        llm_response: str,
    ) -> None:
        self.history.append(
            {
                "iteration": iteration,
                "attack_json": attack_json,
                "metrics": metrics,
                "llm_response": llm_response,
            }
        )

    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    def save(self, path: str) -> None:
        save_json(path, {"history": self.history})