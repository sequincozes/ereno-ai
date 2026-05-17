import json
import re
from typing import Any


def extract_patch_from_llm_response(llm_response: str) -> list[dict[str, Any]]:
    if not llm_response or not llm_response.strip():
        print("[PARSER] Resposta vazia.")
        return []

    lower_response = llm_response.lower()

    error_or_refusal_signals = [
        "i’m sorry",
        "i'm sorry",
        "i cannot help",
        "i can’t help",
        "não posso ajudar",
        "nao posso ajudar",
        "request entity too large",
        "request_too_large",
        "rate limit",
        "rate_limit_exceeded",
        "tokens per minute",
        "tokens per day",
        "invalid_request_error",
        "error calling groq",
    ]

    if any(signal in lower_response for signal in error_or_refusal_signals):
        print("[PARSER] Resposta parece recusa ou erro da API. Patch ignorado.")
        return []

    section_patch = _try_extract_alteracoes_aplicaveis(llm_response)
    if section_patch:
        print(f"[PARSER] Patch extraído de ALTERACOES_APLICAVEIS com {len(section_patch)} alteração(ões).")
        return section_patch

    json_patch = _try_extract_json_patch(llm_response)
    if json_patch:
        return json_patch

    flat_json_patch = _try_extract_flat_json_patch(llm_response)
    if flat_json_patch:
        print(f"[PARSER] Patch extraído de JSON achatado com {len(flat_json_patch)} alteração(ões).")
        return flat_json_patch

    field_objects_patch = _try_extract_field_objects_patch(llm_response)
    if field_objects_patch:
        print(f"[PARSER] Patch extraído de field/current_value com {len(field_objects_patch)} alteração(ões).")
        return field_objects_patch

    textual_patch = _try_extract_textual_patch(llm_response)
    if textual_patch:
        print(f"[PARSER] Patch textual extraído com {len(textual_patch)} alteração(ões).")
        return textual_patch

    print("[PARSER] Nenhum patch válido encontrado.")
    return []


def _try_extract_alteracoes_aplicaveis(llm_response: str) -> list[dict[str, Any]]:
    match = re.search(
        r"ALTERACOES_APLICAVEIS\s*(.*)",
        llm_response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return []

    section = match.group(1).strip()
    lines = section.splitlines()

    patch: list[dict[str, Any]] = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("```"):
            continue

        if ":" not in line:
            continue

        field, raw_value = line.split(":", 1)
        field = field.strip().strip("`").strip()
        raw_value = raw_value.strip().strip("`").strip()

        if not _looks_like_field_path(field):
            continue

        patch.append(
            {
                "operation": "replace",
                "field": field,
                "old_value": None,
                "new_value": _parse_value(raw_value),
                "reason": "Alteração extraída da seção ALTERACOES_APLICAVEIS.",
            }
        )

    return patch


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

        patch = parsed.get("patch")

        if not isinstance(patch, list):
            continue

        valid_patch = []

        for item in patch:
            if not isinstance(item, dict):
                continue

            if item.get("operation") != "replace":
                continue

            field = item.get("field")

            if not field or not _looks_like_field_path(field):
                continue

            valid_patch.append(item)

        print(f"[PARSER] Patch JSON extraído com {len(valid_patch)} alteração(ões).")
        return valid_patch

    return []


def _try_extract_flat_json_patch(llm_response: str) -> list[dict[str, Any]]:
    for candidate in _extract_json_candidates(llm_response):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        patch = []

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
                    "reason": "Alteração extraída de JSON achatado.",
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
                "reason": "Alteração extraída de objeto field/current_value.",
            }
        )

    return patch


def _try_extract_textual_patch(llm_response: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"[-*]?\s*[`\"]?([A-Za-z0-9_.]+)[`\"]?\s*:\s*[`\"]?([^`\"\n\r,}]+)[`\"]?\s*(?:,|$|\n)",
        flags=re.MULTILINE,
    )

    patch = []

    for field, raw_value in pattern.findall(llm_response):
        field = field.strip()
        raw_value = raw_value.strip().strip(".").strip()

        if not _looks_like_field_path(field):
            continue

        patch.append(
            {
                "operation": "replace",
                "field": field,
                "old_value": None,
                "new_value": _parse_value(raw_value),
                "reason": "Alteração extraída de texto livre.",
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
        "ttlMsValues",
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
        return bool_match.group(1).lower() in ["true", "verdadeiro"]

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