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
from adversarial_ids.domain.intent_spec import DesiredEffect

# Só três conjuntos de efeitos existem nos 103 campos do catálogo (verificado:
# a invariante 4 garante que as três alavancas de evasão sempre andam juntas).
# Enumerar os nomes longos em cada campo custava ~1,9k tokens por chamada — um
# rótulo curto com legenda única corta a maior redundância do prompt sem perder
# nada, o que abre espaço para as faixas de `_render_domain` (Fase 1) caberem no
# TPM de 8.000 da conta. Ver `docs/target_values.md` e [[groq-modelos-e-tpm]].
_EVASION = frozenset(
    {DesiredEffect.LOWER_F1, DesiredEffect.LOWER_RECALL, DesiredEffect.MIMIC_NORMAL_TRAFFIC}
)
_ACTIVITY = frozenset({DesiredEffect.INCREASE_ATTACK_ACTIVITY})

_INTENT_PROMPT_PATH = PROMPTS_DIR / "intent.md"


def _render_domain(field) -> str:
    """Faixa aceita do campo, curta o bastante para caber no prompt.

    Existe por causa de ``target_values``: para ditar um valor, a LLM precisa
    saber onde ele pode cair, e o portão determinístico recusa o que estiver
    fora (``attack_capabilities.validate_target_values``). Mantido telegráfico
    de propósito — o catálogo inteiro já custa ~6,7k tokens por chamada contra
    um TPM de 8.000 na conta do projeto (ver ``docs/pilot_e0.md``), então cada
    campo paga no máximo um fragmento entre colchetes.
    """

    if field.choices is not None:
        return " [" + "|".join(repr(choice) for choice in field.choices) + "]"
    if field.minimum is None and field.maximum is None:
        return ""
    low = "" if field.minimum is None else f"{field.minimum:g}"
    high = "" if field.maximum is None else f"{field.maximum:g}"
    return f" [{low}..{high}]"


def _render_effects(field) -> str:
    """Rótulo curto dos efeitos do campo, resolvido pela legenda do catálogo.

    Colapsa a lista de nomes (``lower_f1, lower_recall, ...``) num dos três
    casos que de fato ocorrem — ``evasão``, ``atividade`` ou os dois. A legenda
    no topo do catálogo mapeia cada rótulo de volta aos nomes reais dos efeitos
    que o portão aceita, então o texto não diz menos do que dizia; só diz mais
    curto.
    """

    effects = frozenset(field.effects)
    if effects == _EVASION:
        return "ev"
    if effects == _ACTIVITY:
        return "at"
    if effects == _EVASION | _ACTIVITY:
        return "ev+at"
    # Um quarto conjunto surgiria só se o catálogo mudasse; então o nome longo
    # é o correto — melhor um campo verboso que um rótulo que mente.
    return ", ".join(sorted(e.value for e in field.effects))


#: Nome curto do tipo. Mesma troca dos rótulos de efeito: a legenda do
#: catálogo devolve o nome inteiro, e o portão não lê o texto do prompt.
_SHORT_TYPE = {
    "number": "num",
    "integer": "int",
    "boolean": "bool",
    "string": "str",
    "integer_list": "int[]",
}


def _field_line(field) -> str:
    kind = _SHORT_TYPE.get(field.value_type, field.value_type)
    return (
        f"  - `{field.path}` {kind}{_render_domain(field)} · "
        f"{_render_effects(field)} — {field.description}"
    )


def _signature(field) -> tuple[str, ...]:
    """Tudo que a linha do campo diz. Dois campos com a mesma assinatura são a
    mesma entrada do catálogo aparecendo em ataques diferentes — e só então
    podem ser descritos uma vez. Efeitos e faixa entram na assinatura porque
    um mesmo caminho pode carregar efeitos diferentes em outro ataque, e aí o
    texto compartilhado mentiria sobre o portão."""

    return (
        field.path,
        field.value_type,
        _render_domain(field),
        _render_effects(field),
        field.description,
    )


def _shared_fields(catalog: dict[str, AttackCapability]) -> dict[tuple[str, ...], object]:
    """Campos que aparecem, idênticos, em mais de um ataque.

    As quatro variantes de ``delayed_replay`` e os dois replays repetem oito
    campos palavra por palavra. Descrevê-los uma vez num glossário e deixar
    cada ataque citar só o caminho corta ~2,7k caracteres do prompt sem tirar
    nada do que o portão aceita — a mesma troca que ``_render_effects`` já faz
    com os nomes de efeito.
    """

    seen: dict[tuple[str, ...], list] = {}
    for capability in catalog.values():
        for field in capability.fields:
            seen.setdefault(_signature(field), []).append(field)
    return {sig: fields[0] for sig, fields in seen.items() if len(fields) > 1}


def _render_shared(shared: dict[tuple[str, ...], object]) -> str:
    lines = [
        "### Campos compartilhados",
        "",
        "Os ataques abaixo citam estes caminhos pelo nome; a linha completa "
        "(tipo, faixa e efeitos) é esta aqui e vale igual em todos eles.",
        "",
    ]
    lines.extend(_field_line(field) for field in shared.values())
    return "\n".join(lines)


def _render_capability(
    capability: AttackCapability,
    shared: dict[tuple[str, ...], object] | None = None,
) -> str:
    spec = get_attack_spec(capability.attack_key)
    objectives = ", ".join(sorted(o.value for o in capability.supported_objectives))
    shared = shared or {}

    lines = [
        f"### `{capability.attack_key}` — `{capability.capability_id}`",
        f"- {spec.description} Classe no dataset: `{spec.label}`. "
        f"Objetivos: {objectives}.",
    ]

    common = [f for f in capability.fields if _signature(f) in shared]
    own = [f for f in capability.fields if _signature(f) not in shared]

    if common:
        paths = ", ".join(f"`{field.path}`" for field in common)
        lines.append(f"- Campos compartilhados (ver a seção acima): {paths}.")
    if own:
        lines.append("- Campos próprios:")
        lines.extend(_field_line(field) for field in own)
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
        "Legenda de cada campo (`caminho tipo[faixa] · efeitos — significado`): "
        "tipo é `num`/`int`/`bool`/`str`/`int[]`; `[a..b]` é a faixa aceita (um "
        "lado vazio = sem limite daquele lado) e `[x|y]` são os únicos valores "
        "aceitos — um valor fixado fora disso é rejeitado; e os efeitos são `ev` "
        "(o campo serve a `lower_f1`, `lower_recall` e `mimic_normal_traffic`), "
        "`at` (serve a `increase_attack_activity`) ou `ev+at` (serve aos dois).",
        "",
    ]

    shared = _shared_fields(catalog)
    if shared:
        sections.append(_render_shared(shared))
        sections.append("")

    for attack_key in list_attack_keys():
        capability = by_attack_key.get(attack_key)
        if capability is None:
            continue
        sections.append(_render_capability(capability, shared))
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def build_intent_instructions(
    prompt_path=_INTENT_PROMPT_PATH,
    catalog: dict[str, AttackCapability] | None = None,
) -> str:
    """``prompts/intent.md`` + o bloco do catálogo, na ordem lida pela LLM."""

    base = prompt_path.read_text(encoding="utf-8")
    return f"{base}\n\n{build_capability_context(catalog)}"
