"""O que o injetor adaptativo do E1 lê, e como os erros do provedor são lidos.

Sem agno aqui: a renderização do contexto e a classificação de erro são
testáveis sem LLM. Ver docs/experiments_e1.md, "O que o LLM vê".
"""

from __future__ import annotations

from typing import Any, Sequence

from adversarial_ids.agents.intent.catalog_context import build_capability_context
from adversarial_ids.config.attack_capabilities import get_attack_capability
from adversarial_ids.config.settings import PROMPTS_DIR
from adversarial_ids.domain.defense_plan import DefensePlan
from adversarial_ids.experiments.search import PRIMARY_BUDGET, VALID, ProposalContext

ATTACKER_PROMPT_PATH = PROMPTS_DIR / "attacker.md"


def build_attacker_instructions(family: str, prompt_path=ATTACKER_PROMPT_PATH) -> str:
    """``prompts/attacker.md`` + o catálogo **só da família** da campanha."""

    capability = get_attack_capability(family)
    catalog = build_capability_context({capability.capability_id: capability})
    return f"{prompt_path.read_text(encoding='utf-8').strip()}\n\n{catalog}"


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else repr(value)


def render_entry(entry: dict[str, Any]) -> str:
    changes = ", ".join(
        f"{c['path']}={_fmt(c['new_value'])}" for c in entry.get("diff") or ()
    ) or "—"
    line = f"#{entry['index'] + 1} [{entry['status']}] {changes}"
    search = entry.get("search")
    if entry["status"] == VALID and search:
        evasion = " / ".join(f"{search['evasion'][b]:.3f}" for b in sorted(search["evasion"], key=float))
        line += f" | evasão@1/10/100 = {evasion} | AP = {search['ap']:.3f}"
    elif entry.get("reason"):
        line += f" | motivo: {str(entry['reason'])[:240]}"
    return line


def render_plan(plan: DefensePlan) -> str:
    lines = [f"Prioridade: {plan.priority}"]
    for bucket, actions in (
        ("detecção", plan.detection_actions),
        ("contenção", plan.containment_actions),
        ("hardening", plan.hardening_actions),
    ):
        for action in actions:
            evidence = ", ".join(f"{e.metric_or_feature}={_fmt(e.value)}" for e in action.evidence)
            test = action.validation_test
            lines.append(
                f"- [{bucket}] {action.technique.value if hasattr(action.technique, 'value') else action.technique}: "
                f"{action.description[:200]} (evidência: {evidence}; valida por "
                f"{test.metric} {test.direction} {test.target:g})"
            )
    return "\n".join(lines)


def build_round_message(
    context: ProposalContext, *, plan_block: str | None = None
) -> str:
    """A mensagem da rodada: campanha, caixa, histórico e (B4) o plano."""

    remaining = context.budget - context.index
    history: Sequence[dict[str, Any]] = [e for e in context.history if e.get("status") != "exhausted"]
    sections = [
        "## Campanha",
        f"Família: `{context.family}` · proposta {context.index + 1} de {context.budget} "
        f"(restam {remaining}, contando esta). Objetivo: evasão no orçamento {PRIMARY_BUDGET} "
        "alarmes falsos/10k.",
        "",
        f"## Caixa de busca (no máximo {context.box.max_fields} campos por proposta)",
        *(f"- {line}" for line in context.box.describe()),
        "",
        "## Histórico desta campanha",
        *([render_entry(e) for e in history] or ["(nenhuma proposta ainda)"]),
    ]
    if plan_block is not None:
        sections += ["", "## Último plano do Defensor (sobre o último candidato válido)", plan_block]
    sections += ["", "Proponha a próxima variante chamando `submit_intent_spec`."]
    return "\n".join(sections)


# --------------------------------------------------------------------------- #
# Erros do provedor                                                           #
# --------------------------------------------------------------------------- #
DAILY = "daily"
MINUTE = "minute"
TOO_LARGE = "too_large"

_DAILY_MARKERS = ("per day", "(tpd)", "(rpd)", "tokens per day", "requests per day")
_TOO_LARGE_MARKERS = ("request too large", "request_too_large", "413")
_MINUTE_MARKERS = ("rate_limit_exceeded", "rate limit", "429", "(tpm)", "(rpm)")


def classify_provider_error(text: str | None) -> str | None:
    """413 = prompt grande, 429 por minuto = espaçar, 429 por dia = parar.

    Os três aparecem de fora como "a LLM não chamou a ferramenta"; ler o erro
    antes de contar a proposta como inválida é o que impede uma cota esgotada
    de virar um ponto de dado. ``None`` = não é limite do provedor.
    """

    if not text:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _DAILY_MARKERS):
        return DAILY
    if any(marker in lowered for marker in _TOO_LARGE_MARKERS):
        return TOO_LARGE
    if any(marker in lowered for marker in _MINUTE_MARKERS):
        return MINUTE
    return None
