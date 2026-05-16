import json
import re
from typing import Any


def extract_patch_from_llm_response(llm_response: str) -> list[dict[str, Any]]:
    """
    Extrai alterações da resposta da LLM.

    Aceita:
    1. JSON com chave "patch";
    2. JSON achatado com campos do ataque como chaves:
       {
         "fault.prob": 0.5,
         "analog.deltaAbs.max": 1.5
       }
    3. Texto do tipo:
       - `fault.prob`: 0,4
    4. Blocos do tipo:
       {
         "field": "fault.prob",
         "current_value": 0.4,
         "type": "float"
       }
    """

    if not llm_response or not llm_response.strip():
        print("[PARSER] Resposta vazia.")
        return []

    lower_response = llm_response.lower()

    error_signals = [
        "rate limit",
        "rate_limit_exceeded",
        "request entity too large",
        "request_too_large",
        "error calling groq",
        "error in agent run",
        '"error"',
        "tokens per day",
        "tokens per minute",
        "invalid_request_error",
    ]

    if any(signal in lower_response for signal in error_signals):
        print("[PARSER] Resposta parece erro da API. Patch ignorado.")
        return []

    json_patch = _try_extract_json_patch(llm_response)
    if json_patch:
        return json_patch

    flat_json_patch = _try_extract_flat_json_patch(llm_response)
    if flat_json_patch:
        print(f"[PARSER] Patch extraído de JSON achatado com {len(flat_json_patch)} alteração(ões).")
        return flat_json_patch

    field_objects_patch = _try_extract_field_objects_patch(llm_response)
    if field_objects_patch:
        print(
            f"[PARSER] Patch extraído de objetos field/current_value com "
            f"{len(field_objects_patch)} alteração(ões)."
        )
        return field_objects_patch

    textual_patch = _try_extract_textual_patch(llm_response)
    if textual_patch:
        print(f"[PARSER] Patch textual extraído com {len(textual_patch)} alteração(ões).")
        return textual_patch

    print("[PARSER] Nenhum patch válido encontrado.")
    return []


def _extract_json_candidates(llm_response: str) -> list[str]:
    candidates: list[str] = []

    fenced_blocks = re.findall(
        r"```json\s*(.*?)\s*```",
        llm_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.extend(fenced_blocks)

    object_matches = re.findall(
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        llm_response,
        flags=re.DOTALL,
    )
    candidates.extend(object_matches)

    object_match = re.search(r"\{.*\}", llm_response, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    candidates.append(llm_response.strip())

    return candidates


def _try_extract_json_patch(llm_response: str) -> list[dict[str, Any]]:
    for candidate in _extract_json_candidates(llm_response):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        if "error" in parsed:
            print("[PARSER] JSON contém erro. Patch ignorado.")
            return []

        patch = parsed.get("patch")

        if not isinstance(patch, list):
            continue

        valid_patch = []

        for item in patch:
            if not isinstance(item, dict):
                continue

            operation = item.get("operation")
            field = item.get("field")

            if operation != "replace":
                continue

            if not field or not _looks_like_field_path(field):
                continue

            valid_patch.append(item)

        print(f"[PARSER] Patch JSON extraído com {len(valid_patch)} alteração(ões).")
        return valid_patch

    return []


def _try_extract_flat_json_patch(llm_response: str) -> list[dict[str, Any]]:
    """
    Extrai JSON achatado como:

    {
      "fault.prob": 0.5,
      "fault.durationMs.min": 500,
      "sqnumMode": "fast"
    }
    """

    patch = []

    for candidate in _extract_json_candidates(llm_response):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        for field, value in parsed.items():
            if not isinstance(field, str):
                continue

            if not _looks_like_field_path(field):
                continue

            patch.append(
                {
                    "operation": "replace",
                    "field": field,
                    "old_value": None,
                    "new_value": value,
                    "reason": "Alteração extraída automaticamente de JSON achatado da LLM.",
                }
            )

        if patch:
            return patch

    return []


def _try_extract_field_objects_patch(llm_response: str) -> list[dict[str, Any]]:
    patch = []

    object_blocks = re.findall(
        r"\{[^{}]*\"field\"\s*:\s*\"[^\"]+\"[^{}]*\}",
        llm_response,
        flags=re.DOTALL,
    )

    for block in object_blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        field = parsed.get("field")

        if not field or not isinstance(field, str):
            continue

        if not _looks_like_field_path(field):
            continue

        if "new_value" in parsed:
            value = parsed["new_value"]
        elif "current_value" in parsed:
            value = parsed["current_value"]
        elif "value" in parsed:
            value = parsed["value"]
        else:
            continue

        patch.append(
            {
                "operation": "replace",
                "field": field,
                "old_value": None,
                "new_value": value,
                "reason": "Alteração extraída automaticamente de bloco field/current_value da LLM.",
            }
        )

    return patch


def _try_extract_textual_patch(llm_response: str) -> list[dict[str, Any]]:
    patch = []

    pattern = re.compile(
        r"[-*]?\s*[`\"]?([A-Za-z0-9_.]+)[`\"]?\s*:\s*[`\"]?([^`\"\n\r,}]+)[`\"]?\s*(?:,|$|\n)",
        flags=re.MULTILINE,
    )

    matches = pattern.findall(llm_response)

    for field, raw_value in matches:
        field = field.strip()
        raw_value = raw_value.strip().strip(".").strip()

        if not _looks_like_field_path(field):
            continue

        value = _parse_value(raw_value)

        patch.append(
            {
                "operation": "replace",
                "field": field,
                "old_value": None,
                "new_value": value,
                "reason": "Alteração extraída automaticamente da resposta textual da LLM.",
            }
        )

    return patch


def _looks_like_field_path(field: str) -> bool:
    ignored_fields = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1-score",
        "support",
        "current_value",
        "type",
        "attackType",
        "enabled",
    }

    if field in ignored_fields or field.lower() in ignored_fields:
        return False

    allowed_roots = [
        "fault",
        "trapArea",
        "analog",
        "sqnumMode",
        "ttlValues",
        "cbStatus",
        "incrementStNumOnFault",
    ]

    return any(field == root or field.startswith(f"{root}.") for root in allowed_roots)


def _parse_value(value: str) -> Any:
    value = value.strip()
    value = value.replace(",", ".")

    value = re.sub(
        r"^(alterar|mudar|trocar|reduzir|aumentar)\s+para\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"^para\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )

    quoted = re.search(r'["\']([^"\']+)["\']', value)
    if quoted:
        return quoted.group(1).strip()

    bool_match = re.search(r"\b(true|false|verdadeiro|falso)\b", value, flags=re.IGNORECASE)
    if bool_match:
        bool_value = bool_match.group(1).lower()
        return bool_value in ["true", "verdadeiro"]

    number_match = re.search(r"-?\d+(?:\.\d+)?", value)
    if number_match:
        number_text = number_match.group(0)

        if "." in number_text:
            return float(number_text)

        return int(number_text)

    word_match = re.search(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", value)
    if word_match:
        return word_match.group(0)

    return value