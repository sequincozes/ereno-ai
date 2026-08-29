from copy import deepcopy
from typing import Any


def apply_patch_to_json(
    original_json: dict[str, Any],
    patch: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Aplica um patch simples no JSON do ataque.
    """

    updated_json = deepcopy(original_json)

    for operation in patch:
        op = operation.get("operation")
        field_path = operation.get("field")
        new_value = operation.get("new_value")

        if op != "replace":
            print(f"[PATCH] Operação ignorada: {op}")
            continue

        if not field_path:
            print("[PATCH] Campo vazio ignorado.")
            continue

        try:
            _set_value_by_path(
                data=updated_json,
                path=field_path,
                value=new_value,
            )
            print(f"[PATCH] Campo atualizado: {field_path} -> {new_value}")
        except Exception as error:
            print(f"[PATCH] Não foi possível aplicar {field_path}: {error}")

    return updated_json


def _set_value_by_path(
    data: dict[str, Any],
    path: str,
    value: Any,
) -> None:
    parts = path.split(".")
    current: Any = data

    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Caminho inválido em: {part}")

        if part not in current:
            raise KeyError(f"Campo não encontrado: {part}")

        current = current[part]

    last_part = parts[-1]

    if not isinstance(current, dict):
        raise TypeError(f"Não é possível alterar campo final: {last_part}")

    if last_part not in current:
        raise KeyError(f"Campo final não encontrado: {last_part}")

    current[last_part] = value