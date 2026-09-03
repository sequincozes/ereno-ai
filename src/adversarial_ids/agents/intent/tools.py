"""Ferramenta de tool calling estruturado do IntentAgent (Agno).

A LLM chama ``submit_intent_spec`` para propor os campos estruturados de uma
intenção. Deliberadamente **não** recebe ``source_prompt``: o texto original
do usuário é anexado pelo ``IntentAgent`` depois da chamada, nunca
reconstituído pela LLM (evita paráfrase/perda do prompt original no
artefato persistido).

A ferramenta valida a forma dos dados (``IntentRestrictions``, enums) antes de
devolvê-los — mas a allowlist de capacidades por ataque
(``validate_intent_capability``) roda fora daqui, no ``IntentAgent``, mantendo
o guardrail central fora do alcance da LLM.
"""

from __future__ import annotations

import json
from typing import Any

from agno.tools import tool
from pydantic import ValidationError

from adversarial_ids.domain.intent_spec import (
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
)

_VALID_OBJECTIVES = {objective.value for objective in IntentObjective}
_VALID_EFFECTS = {effect.value for effect in DesiredEffect}
_VALID_INTENSITIES = {intensity.value for intensity in IntentIntensity}


@tool(stop_after_tool_call=True)
def submit_intent_spec(
    objective: str,
    base_attack: str,
    desired_effect: str,
    intensity: str = "medium",
    allowed_fields: list[str] | None = None,
    forbidden_fields: list[str] | None = None,
    max_fields_changed: int = 3,
    seed: int = 42,
) -> str:
    """Submete os campos estruturados de uma intenção interpretada do prompt.

    Use esta ferramenta para registrar a interpretação final do pedido do
    usuário. A ferramenta valida os parâmetros e retorna o resultado
    validado ou uma mensagem de erro acionável.

    Args:
        objective: Objetivo de alto nível ("assess_ids_robustness" ou
            "evade_detection").
        base_attack: Chave do ataque ERENO a customizar. Use uma das chaves
            listadas na seção "Catálogo de capacidades" das instruções (ex.:
            "masquerade_fault", "flooding", "grayhole").
        desired_effect: Efeito mensurável desejado (ex.: "lower_recall").
        intensity: Intensidade da alteração ("low", "medium" ou "high").
        allowed_fields: Se o prompt restringir explicitamente quais campos
            podem mudar, liste os caminhos dot aqui. Deixe vazio/None quando
            o prompt não impuser essa restrição.
        forbidden_fields: Caminhos dot que o prompt proíbe alterar.
        max_fields_changed: Máximo de campos que o compilador pode alterar
            (1 a 16; o teto efetivo é o número de campos do ataque escolhido
            capazes do efeito pedido, que pode ser bem menor).
        seed: Semente determinística para reprodutibilidade (0 a 4294967295).
    """
    errors: list[str] = []

    if objective not in _VALID_OBJECTIVES:
        errors.append(
            f"objective inválido: {objective!r}. Use um de {sorted(_VALID_OBJECTIVES)}."
        )
    if desired_effect not in _VALID_EFFECTS:
        errors.append(
            f"desired_effect inválido: {desired_effect!r}. "
            f"Use um de {sorted(_VALID_EFFECTS)}."
        )
    if intensity not in _VALID_INTENSITIES:
        errors.append(
            f"intensity inválida: {intensity!r}. Use um de {sorted(_VALID_INTENSITIES)}."
        )
    if not isinstance(base_attack, str) or not base_attack.strip():
        errors.append("base_attack não pode ser vazio.")

    if errors:
        return json.dumps({"errors": errors})

    try:
        restrictions = IntentRestrictions(
            allowed_fields=tuple(allowed_fields) if allowed_fields else None,
            forbidden_fields=tuple(forbidden_fields or ()),
            max_fields_changed=max_fields_changed,
        )
    except ValidationError as exc:
        return json.dumps({"errors": [str(exc)]})

    payload: dict[str, Any] = {
        "objective": objective,
        "base_attack": base_attack.strip(),
        "desired_effect": desired_effect,
        "intensity": intensity,
        "restrictions": json.loads(restrictions.model_dump_json()),
        "seed": seed,
    }
    return json.dumps(payload)
