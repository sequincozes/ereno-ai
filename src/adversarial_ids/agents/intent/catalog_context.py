"""Injeta o catálogo real de capacidades no prompt do IntentAgent.

Antes de o catálogo crescer além de ``masquerade_fault``, ``prompts/intent.md``
podia listar o único ataque habilitado como texto estático. Com 10 ataques
registrados isso deixou de ser possível sem duplicar (e eventualmente
divergir de) ``config/attack_capabilities.py`` — o mesmo problema que
``agents/defender/tools.py::select_evidence_candidates`` resolve para o
Defensor: o texto que a LLM lê e a regra que valida a saída dela precisam vir
da mesma fonte, ou os dois lados divergem silenciosamente.

Precedente do lado Red: ``agents/orchestrator/live.py::_attack_context``
injeta o contexto de **um** ataque por execução (o ataque já foi escolhido via
``--attack`` antes do Estrategista rodar). Aqui é o oposto — a função do
IntentAgent é *escolher* o ataque a partir do prompt, então o contexto precisa
ser o catálogo inteiro, não um recorte.
"""

from __future__ import annotations

from adversarial_ids.config.attack_capabilities import AttackCapability, ATTACK_CAPABILITY_CATALOG
from adversarial_ids.config.attacks_registry import get_attack_spec, list_attack_keys
from adversarial_ids.config.settings import PROMPTS_DIR

_INTENT_PROMPT_PATH = PROMPTS_DIR / "intent.md"


def _render_capability(capability: AttackCapability) -> str:
    spec = get_attack_spec(capability.attack_key)
    objectives = ", ".join(sorted(o.value for o in capability.supported_objectives))
    effects = ", ".join(sorted(e.value for e in capability.supported_effects))

    lines = [
        f"### `{capability.attack_key}` — `{capability.capability_id}`",
        f"- Descrição: {spec.description}",
        f"- Classe rotulada no dataset: `{spec.label}`",
        f"- Objetivos suportados: {objectives}",
        f"- Efeitos suportados: {effects}",
        "- Campos (caminho — tipo — efeitos):",
    ]
    for field in capability.fields:
        field_effects = ", ".join(sorted(e.value for e in field.effects))
        lines.append(f"  - `{field.path}` — {field.value_type} — {field_effects} — {field.description}")
    return "\n".join(lines)


def build_capability_context(
    catalog: dict[str, AttackCapability] | None = None,
) -> str:
    """Bloco Markdown com o catálogo real, ordenado por ``list_attack_keys()``.

    Derivado de ``ATTACK_CAPABILITY_CATALOG`` — a mesma allowlist que
    ``resolve_candidate_paths`` aplica no portão determinístico — para que o
    texto que a LLM lê nunca diga mais (nem menos) do que o portão realmente
    aceita. Ordenado pela ordem de registro em ``attacks_registry.py`` para
    que o prompt seja estável entre execuções.
    """

    catalog = catalog if catalog is not None else ATTACK_CAPABILITY_CATALOG
    by_attack_key = {capability.attack_key: capability for capability in catalog.values()}

    sections = [
        "## Catálogo de capacidades (allowlist determinística)",
        "",
        "Só os ataques abaixo têm capacidade intent-driven habilitada. Propor "
        "`base_attack` fora desta lista faz o portão determinístico rejeitar a "
        "intenção antes do compilador — nunca invente um ataque ou um caminho "
        "de campo que não esteja listado aqui.",
        "",
    ]
    for attack_key in list_attack_keys():
        capability = by_attack_key.get(attack_key)
        if capability is None:
            continue
        sections.append(_render_capability(capability))
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def build_intent_instructions(
    prompt_path=_INTENT_PROMPT_PATH,
    catalog: dict[str, AttackCapability] | None = None,
) -> str:
    """``prompts/intent.md`` + o bloco do catálogo, na ordem lida pela LLM."""

    base = prompt_path.read_text(encoding="utf-8")
    return f"{base}\n\n{build_capability_context(catalog)}"
