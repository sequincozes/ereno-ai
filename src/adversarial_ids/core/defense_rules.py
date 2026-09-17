"""Avaliação por regras do DefensePlan (épico E5, janela D47-54).

Último componente da janela "Defesa acionável": o catálogo
(``config/defense_techniques.py``) dizia o que cada evidência recomenda, mas
ninguém o consultava — a recomendação da LLM e a evidência que ela citava
podiam não ter relação nenhuma e o plano passava.

Este módulo é a consulta. Ele não repete nada do portão de evidência
(``agents/defender/tools.py::validate_plan_against_report``), que responde
"esse número existe mesmo?"; aqui a pergunta é a seguinte, e é outra:
**a técnica proposta é resposta ao que a ação citou?**

## O que "sustentado" quer dizer, exatamente

Uma ação está sustentada quando sua ``technique`` aparece no catálogo para
*pelo menos uma* das evidências que ela cita. Pelo menos uma, e não todas, de
propósito: um plano sério cita o recall **e** a feature que o derrubou, e exigir
que a técnica respondesse às duas rejeitaria justamente a ação mais bem
fundamentada do plano.

A avaliação é pura e determinística — nenhuma LLM, mesma disciplina do E10. Ela
também nunca levanta exceção por conteúdo: produz o veredito, e quem recusa é o
portão. Um plano ruim precisa ser *legível* antes de ser recusado.

## O eixo do playbook é conselho, nunca recusa

``attack_key`` liga a segunda metade da avaliação: o playbook IEC-61850 da
família daquele ataque. Ele fala do cenário, não desta rodada — um playbook de
replay pede validação de sequência mesmo sob um modo de features em que nenhum
delta chega ao relatório. Por isso ele aconselha e não bloqueia: cobrar que todo
plano esgote o playbook transformaria a resposta certa para *esta* execução em
erro. Ver ``docs/defender_validation.md``.
"""

from __future__ import annotations

from typing import Any

from adversarial_ids.config.defense_techniques import (
    DefensePlaybook,
    playbook_for_attack,
    techniques_for_evidence,
)
from adversarial_ids.domain.defense_plan import (
    DEFENSE_BUCKETS,
    DefenseAction,
    DefensePlan,
)
from adversarial_ids.domain.defense_rule_report import (
    DefenseRuleReport,
    RuleFinding,
    severity_of,
)
from adversarial_ids.domain.detection_report import DetectionReport


# Evidência descritiva: os campos de texto do relatório. Derivado do contrato em
# vez de listado à mão — no dia em que o DetectionReport ganhar outro campo de
# texto, ele entra aqui sozinho, e não como um achado "feature desconhecida".
_DESCRIPTIVE_EVIDENCE: frozenset[str] = frozenset(
    name
    for name, field in DetectionReport.model_fields.items()
    if field.annotation is str
)


def _iter_buckets(
    plan: DefensePlan,
) -> tuple[tuple[str, tuple[DefenseAction, ...]], ...]:
    """Os três baldes do plano, pelos nomes que o vocabulário declara.

    ``DefenseBucket`` nomeia cada balde pelo campo que o carrega justamente para
    que ninguém precise repetir a tripla em mais um lugar — este módulo seria o
    terceiro.
    """

    return tuple((bucket, getattr(plan, bucket)) for bucket in DEFENSE_BUCKETS)


def _finding(rule: str, message: str, **fields: Any) -> RuleFinding:
    """Monta um achado já com a severidade declarada para aquela regra.

    A severidade nunca é escrita à mão em lugar nenhum deste módulo: sai sempre
    de ``severity_of``, que é o mesmo mapa que o portão consulta para decidir o
    que recusa.
    """

    return RuleFinding.model_validate(
        {"rule": rule, "severity": severity_of(rule), "message": message, **fields}
    )


def _evaluate_action(
    action: DefenseAction,
    *,
    bucket: str,
    index: int,
) -> tuple[bool, list[RuleFinding]]:
    """Avalia uma ação: ela está sustentada, e o que mais há a dizer sobre ela."""

    findings: list[RuleFinding] = []
    location = {"bucket": bucket, "action_index": index, "technique": action.technique}

    supported: set[str] = set()
    descriptive: list[str] = []
    uncatalogued: list[str] = []

    for evidence in action.evidence:
        key = evidence.metric_or_feature
        techniques = techniques_for_evidence(key)

        if techniques:
            supported.update(techniques)
            continue

        # Sem técnica catalogada, e os dois motivos possíveis pedem conselhos
        # diferentes: uma chave descritiva é evidência legítima que apenas não se
        # remedia, enquanto uma feature desconhecida é buraco do catálogo.
        if key in _DESCRIPTIVE_EVIDENCE:
            descriptive.append(key)
        else:
            uncatalogued.append(key)

    if descriptive:
        findings.append(
            _finding(
                "evidence_not_actionable",
                f"Evidência descritiva não sustenta técnica nenhuma: "
                f"{', '.join(descriptive)}. Ela identifica a execução, mas não "
                "há o que remediar nela.",
                subjects=tuple(descriptive),
                **location,
            )
        )

    if uncatalogued:
        findings.append(
            _finding(
                "uncatalogued_feature",
                f"Feature fora do catálogo feature→técnica: "
                f"{', '.join(uncatalogued)}. A evidência é real, mas nenhuma "
                "técnica se ancora nela até o catálogo cobri-la.",
                subjects=tuple(uncatalogued),
                **location,
            )
        )

    grounded = action.technique in supported
    if not grounded:
        cited = ", ".join(
            evidence.metric_or_feature for evidence in action.evidence
        )
        alternatives = (
            ", ".join(sorted(supported))
            if supported
            else "nenhuma — toda a evidência citada é descritiva ou desconhecida"
        )
        findings.append(
            _finding(
                "technique_not_grounded",
                f"A técnica {action.technique!r} não é resposta catalogada à "
                f"evidência citada ({cited}). O que essa evidência sustenta: "
                f"{alternatives}.",
                subjects=(action.technique,),
                **location,
            )
        )

    return grounded, findings


def _evaluate_playbook(
    plan: DefensePlan,
    report: DetectionReport,
    playbook: DefensePlaybook,
) -> list[RuleFinding]:
    """Compara o plano com o playbook da família, sem nunca bloquear."""

    findings: list[RuleFinding] = []

    used = {action.technique for _, actions in _iter_buckets(plan) for action in actions}
    missing = tuple(
        technique for technique in playbook.techniques if technique not in used
    )
    if missing:
        findings.append(
            _finding(
                "playbook_technique_missing",
                f"O playbook {playbook.key!r} ({playbook.title}) prescreve "
                f"técnicas que o plano não propõe: {', '.join(missing)}.",
                subjects=missing,
            )
        )

    # Só cobra a assinatura que o detector *de fato* destacou nesta execução.
    # Cobrar uma feature ausente de `top_features` seria cobrar do plano algo que
    # o relatório não tinha como mostrar — e, sob o modo de features default,
    # seria o caso de quase todo o playbook de replay.
    surfaced = {feature.feature for feature in report.top_features}
    cited = {
        evidence.metric_or_feature
        for _, actions in _iter_buckets(plan)
        for action in actions
        for evidence in action.evidence
    }
    ignored = tuple(
        feature
        for feature in playbook.signature_features
        if feature in surfaced and feature not in cited
    )
    if ignored:
        findings.append(
            _finding(
                "playbook_signature_ignored",
                f"O relatório destacou features-assinatura de "
                f"{playbook.key!r} que nenhuma ação cita: {', '.join(ignored)}.",
                subjects=ignored,
            )
        )

    return findings


def evaluate_plan_rules(
    plan: DefensePlan | dict[str, Any],
    report: DetectionReport | dict[str, Any],
    *,
    attack_key: str | None = None,
    detection_report_ref: str | None = None,
) -> DefenseRuleReport:
    """Avalia o plano contra o catálogo de defesa e devolve o veredito.

    Determinística e sem efeito colateral: não levanta exceção por plano ruim —
    devolve um ``DefenseRuleReport`` em que ``is_grounded`` é falso e os achados
    dizem por quê. Quem recusa é ``validate_plan_against_report``.

    ``attack_key`` liga o eixo do playbook (só conselhos). Omitido, o relatório
    sai com ``playbook_key=None``, dizendo que aquele eixo não foi avaliado em
    vez de deixar a ausência de achados parecer aprovação.
    """

    plan_model = DefensePlan.model_validate(plan)
    report_model = DetectionReport.model_validate(report)

    findings: list[RuleFinding] = []
    total = 0
    grounded_count = 0

    for bucket, actions in _iter_buckets(plan_model):
        for index, action in enumerate(actions):
            total += 1
            grounded, action_findings = _evaluate_action(
                action, bucket=bucket, index=index
            )
            grounded_count += int(grounded)
            findings.extend(action_findings)

    playbook_key: str | None = None
    if attack_key is not None:
        playbook = playbook_for_attack(attack_key)
        playbook_key = playbook.key
        findings.extend(_evaluate_playbook(plan_model, report_model, playbook))

    return DefenseRuleReport(
        detection_report_ref=detection_report_ref or plan_model.detection_report_ref,
        attack_key=attack_key,
        playbook_key=playbook_key,
        total_actions=total,
        grounded_actions=grounded_count,
        findings=tuple(findings),
    )
