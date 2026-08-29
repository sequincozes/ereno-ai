"""strategist/personas.py — personas de ataque (conservadora vs. agressiva).

Define as personas que controlam o comportamento do Estrategista:
temperatura, agressividade das alterações e instruções injetadas no prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Persona:
    """Perfil de ataque do Estrategista (Red Team).

    Atributos:
        name: Identificador da persona ("conservative" ou "aggressive").
        temperature: Temperatura da LLM (conservative=0.1, aggressive=0.4).
        max_changes_per_iteration: Número máximo de campos alterados por
            iteração (conservative=1, aggressive=3).
        max_change_ratio: Razão máxima de alteração relativa ao valor
            atual — ex.: 0.15 significa que o valor só pode variar até 15%
            do original (conservative). Para aggressive, 0.50 = até 50%.
        description: Descrição legível do perfil em português.
        system_prompt_extra: Instruções extras injetadas no prompt do agente.
    """

    name: Literal["conservative", "aggressive"]
    temperature: float
    max_changes_per_iteration: int
    max_change_ratio: float
    description: str
    system_prompt_extra: str


CONSERVATIVE = Persona(
    name="conservative",
    temperature=0.1,
    max_changes_per_iteration=1,
    max_change_ratio=0.15,
    description=(
        "Perfil conservador: alterações mínimas e graduais, priorizando "
        "plausibilidade e evitando variantes degeneradas. Ideal para testar "
        "a sensibilidade do IDS a perturbações sutis."
    ),
    system_prompt_extra=(
        "Você está no perfil CONSERVATIVE.\n"
        "- Altere NO MÁXIMO 1 campo por iteração.\n"
        "- Cada alteração deve ser de no máximo 15% do valor original.\n"
        "- Priorize mudanças sutis que preservem a plausibilidade do ataque.\n"
        "- Evite zerar ou saturar parâmetros; prefira ajustes finos.\n"
        "- Se o ataque já estiver difícil de detectar, seja ainda mais cauteloso."
    ),
)

AGGRESSIVE = Persona(
    name="aggressive",
    temperature=0.4,
    max_changes_per_iteration=3,
    max_change_ratio=0.50,
    description=(
        "Perfil agressivo: alterações mais ousadas e em maior quantidade, "
        "explorando combinações de parâmetros para derrubar a detecção. "
        "Ideal para stress-test do IDS."
    ),
    system_prompt_extra=(
        "Você está no perfil AGGRESSIVE.\n"
        "- Altere até 3 campos por iteração.\n"
        "- Cada alteração pode variar até 50% do valor original.\n"
        "- Explore combinações de parâmetros para maximizar a evasão.\n"
        "- Não tenha medo de mudanças significativas, mas evite variantes "
        "degeneradas (não zere fault.prob, analog.deltaAbs ou trapArea.spikeProb).\n"
        "- Se o F1 do ataque ainda estiver alto, seja mais ousado."
    ),
)

_PERSONAS: dict[str, Persona] = {
    "conservative": CONSERVATIVE,
    "aggressive": AGGRESSIVE,
}


def get_persona(name: str) -> Persona:
    """Retorna a persona pelo nome.

    Args:
        name: Nome da persona ("conservative" ou "aggressive").

    Returns:
        Instância de ``Persona`` correspondente.

    Raises:
        KeyError: Se o nome não corresponder a uma persona conhecida.
    """
    if name not in _PERSONAS:
        raise KeyError(
            f"Persona desconhecida: '{name}'. "
            f"Use uma das seguintes: {list(_PERSONAS.keys())}."
        )
    return _PERSONAS[name]