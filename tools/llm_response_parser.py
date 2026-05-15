import json
import re
from typing import Any


def extract_updated_attack_json(
    llm_response: str,
    fallback_attack_json: dict[str, Any],
) -> dict[str, Any]:
    """
    Extrai o JSON atualizado da resposta da LLM.

    Segurança:
    - Não aceita respostas de erro da API.
    - Não aceita qualquer JSON genérico.
    - Só aceita JSON com a chave 'updated_attack_json'.
    - Só aplica se o JSON atualizado preservar campos essenciais do ataque.
    """

    if not llm_response or not llm_response.strip():
        print("[PARSER] Resposta vazia. Usando JSON anterior.")
        return fallback_attack_json

    lower_response = llm_response.lower()

    error_signals = [
        "rate limit",
        "rate_limit_exceeded",
        "error calling groq",
        "error in agent run",
        '"error"',
        "tokens per day",
    ]

    if any(signal in lower_response for signal in error_signals):
        print("[PARSER] Resposta parece ser erro da API. Usando JSON anterior.")
        return fallback_attack_json

    json_candidates: list[str] = []

    fenced_blocks = re.findall(
        r"```json\s*(.*?)\s*```",
        llm_response,
        flags=re.DOTALL | re.IGNORECASE,
    )

    json_candidates.extend(fenced_blocks)

    object_match = re.search(r"\{.*\}", llm_response, flags=re.DOTALL)
    if object_match:
        json_candidates.append(object_match.group(0))

    json_candidates.append(llm_response.strip())

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            continue

        if "error" in parsed:
            print("[PARSER] JSON contém erro. Usando JSON anterior.")
            return fallback_attack_json

        if "updated_attack_json" not in parsed:
            print("[PARSER] JSON não contém updated_attack_json. Ignorando candidato.")
            continue

        updated = parsed["updated_attack_json"]

        if not isinstance(updated, dict):
            print("[PARSER] updated_attack_json não é objeto JSON. Ignorando candidato.")
            continue

        if not _looks_like_valid_attack_json(updated, fallback_attack_json):
            print("[PARSER] updated_attack_json não parece preservar o JSON do ataque. Usando anterior.")
            return fallback_attack_json

        print("[PARSER] Novo JSON de ataque extraído com segurança.")
        return updated

    print("[PARSER] Nenhum JSON atualizado válido encontrado. Usando JSON anterior.")
    return fallback_attack_json


def _looks_like_valid_attack_json(
    updated: dict[str, Any],
    original: dict[str, Any],
) -> bool:
    """
    Validação conservadora.

    O JSON atualizado precisa preservar chaves essenciais do JSON original.
    Isso evita salvar respostas de erro ou objetos incompletos no ERENO.
    """

    if not original:
        return False

    original_keys = set(original.keys())
    updated_keys = set(updated.keys())

    # Exige que a maior parte das chaves originais continue presente.
    common_keys = original_keys.intersection(updated_keys)

    if len(original_keys) > 0:
        preservation_ratio = len(common_keys) / len(original_keys)
        if preservation_ratio < 0.7:
            print(f"[PARSER] Poucas chaves preservadas: {preservation_ratio:.2f}")
            return False

    # Bloqueia JSONs claramente vindos de erro/API.
    forbidden_keys = {"error", "message", "code", "type"}

    if forbidden_keys.intersection(updated_keys):
        print("[PARSER] Chaves de erro detectadas no JSON atualizado.")
        return False

    return True