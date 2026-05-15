import json
from pathlib import Path
from typing import Any


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_text(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)

    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)