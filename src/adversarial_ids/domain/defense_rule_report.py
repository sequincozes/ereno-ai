"""DefenseRuleReport — o veredito da avaliação por regras do DefensePlan.

Fecha o épico E5 (janela D47-54, "Defesa acionável"). O portão de evidência
(``agents/defender/tools.py::validate_plan_against_report``) já provava que o
plano **não mente**: a métrica existe, o valor bate, o teste de validação cobra
melhora. O que faltava é a outra metade da frase do gate de saída — "100% das
recomendações ligadas a evidência" —, porque *citar* recall e *responder* a ele
não são a mesma coisa. Um plano que cita ``recall`` e propõe
``goose_authentication`` passava inteiro: o balde é legal, a evidência é real, o
alvo é remedível. Autenticar publisher é uma boa ideia, mas não é o que um
recall baixo pede, e nada no contrato dizia isso.

Este relatório é a resposta: para cada ação, a técnica escolhida está ou não
catalogada como resposta a alguma evidência que a própria ação cita
(``config/defense_techniques.py``). ``grounded_actions``/``total_actions`` é
literalmente o número que a janela cobra.

## Por que um relatório, e não só uma exceção

Duas severidades, e a distinção é deliberada:

``blocking``
    A recomendação não responde à evidência que ela mesma cita. Isso é defeito
    do plano, e o portão recusa — mesma disciplina do E5: um plano rejeitado
    falha o ``LoopStage`` e para a execução, em vez de degradar para um plano
    fraco persistido como se fosse bom.

``advisory``
    O plano é válido, mas o catálogo sabe de algo que ele não aproveitou: uma
    técnica que o playbook IEC-61850 daquela família prescreve e ninguém propôs,
    ou uma feature-assinatura que o detector de fato destacou e nenhuma ação
    citou. Recusar isso seria exigir que a LLM esgotasse o playbook a cada
    rodada; silenciar seria jogar fora a única leitura que o catálogo tem do
    cenário. Vira achado registrado, não erro.

A parte aconselhável só existe quando o chamador informa ``attack_key`` — sem
saber que ataque gerou a execução não há playbook a cobrar, e o relatório diz
isso com ``playbook_key=None`` em vez de fingir cobertura total.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.defense_plan import DefenseBucket, DefenseTechnique

RuleSeverity = Literal["blocking", "advisory"]

DefenseRuleId = Literal[
    # Blocking: a técnica não é resposta catalogada a nenhuma evidência citada.
    "technique_not_grounded",
    # Advisory: a evidência citada é descritiva (model_name, split) — ela
    # localiza a execução, mas não há o que remediar nela.
    "evidence_not_actionable",
    # Advisory: a evidência é uma feature que o catálogo ainda não conhece.
    # Lacuna do catálogo, não defeito do plano.
    "uncatalogued_feature",
    # Advisory: o playbook da família prescreve uma técnica que o plano não usa.
    "playbook_technique_missing",
    # Advisory: o relatório destacou uma feature-assinatura do playbook e
    # nenhuma ação a citou.
    "playbook_signature_ignored",
]
DEFENSE_RULE_IDS: tuple[str, ...] = get_args(DefenseRuleId)

# Que regra bloqueia e que regra apenas aconselha. Mora aqui, junto do
# vocabulário, pelo mesmo motivo que ``_BUCKETS_BY_TECHNIQUE`` mora em
# ``defense_plan``: o core avalia e o portão recusa a partir do mesmo mapa,
# então os dois não têm como divergir sobre o que é motivo de recusa.
_SEVERITY_BY_RULE: dict[str, str] = {
    "technique_not_grounded": "blocking",
    "evidence_not_actionable": "advisory",
    "uncatalogued_feature": "advisory",
    "playbook_technique_missing": "advisory",
    "playbook_signature_ignored": "advisory",
}

if set(_SEVERITY_BY_RULE) != set(DEFENSE_RULE_IDS):  # pragma: no cover
    raise RuntimeError(
        "_SEVERITY_BY_RULE está fora de sincronia com DefenseRuleId: "
        f"faltando={sorted(set(DEFENSE_RULE_IDS) - set(_SEVERITY_BY_RULE))}, "
        f"sobrando={sorted(set(_SEVERITY_BY_RULE) - set(DEFENSE_RULE_IDS))}."
    )


def severity_of(rule: str) -> str:
    """Severidade declarada de uma regra."""

    if rule not in _SEVERITY_BY_RULE:
        raise ValueError(
            f"Regra desconhecida: {rule!r}. Esperado uma de {DEFENSE_RULE_IDS}."
        )

    return _SEVERITY_BY_RULE[rule]


def blocking_rules() -> tuple[str, ...]:
    """Regras cuja violação recusa o plano, na ordem de declaração."""

    return tuple(
        rule for rule in DEFENSE_RULE_IDS if _SEVERITY_BY_RULE[rule] == "blocking"
    )


class RuleFinding(BaseModel):
    """Um achado da avaliação, localizado na ação que o provocou.

    ``bucket``/``action_index`` são nulos nos achados de plano (os do playbook),
    que não pertencem a ação nenhuma — o playbook cobra do plano inteiro. Manter
    o mesmo tipo para os dois evita duas listas que depois divergem.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: DefenseRuleId
    severity: RuleSeverity
    bucket: DefenseBucket | None = None
    action_index: int | None = Field(default=None, ge=0)
    technique: DefenseTechnique | None = None
    # As chaves (features, métricas ou técnicas) sobre as quais o achado fala.
    # Separado de ``message`` de propósito: a mensagem é para quem lê,
    # ``subjects`` é para quem agrega — contar achados por feature ao longo de
    # uma campanha não deveria exigir parsing de texto em português.
    subjects: tuple[str, ...] = ()
    message: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _severity_matches_the_rule(self) -> "RuleFinding":
        """Um achado bloqueante rotulado `advisory` escaparia do portão."""

        expected = severity_of(self.rule)
        if self.severity != expected:
            raise ValueError(
                f"Severidade incoerente para {self.rule!r}: "
                f"esperado={expected!r}, recebido={self.severity!r}."
            )

        return self

    @model_validator(mode="after")
    def _action_findings_carry_their_location(self) -> "RuleFinding":
        """Ou o achado localiza a ação (balde *e* índice), ou é de plano."""

        if (self.bucket is None) != (self.action_index is None):
            raise ValueError(
                "Achado de ação precisa de bucket e action_index juntos; "
                f"recebido bucket={self.bucket!r}, "
                f"action_index={self.action_index!r}."
            )

        return self


class DefenseRuleReport(BaseModel):
    """Quanto do plano a evidência sustenta, e o que o catálogo ainda cobra."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    # Referência da execução, como no DefensePlan: o arquivo sozinho precisa
    # dizer de que rodada ele saiu.
    detection_report_ref: str | None = None
    attack_key: str | None = None
    playbook_key: str | None = None
    total_actions: int = Field(ge=1)
    grounded_actions: int = Field(ge=0)
    findings: tuple[RuleFinding, ...] = ()

    @model_validator(mode="after")
    def _grounded_actions_fit_the_plan(self) -> "DefenseRuleReport":
        if self.grounded_actions > self.total_actions:
            raise ValueError(
                f"grounded_actions={self.grounded_actions} excede "
                f"total_actions={self.total_actions}."
            )

        return self

    @model_validator(mode="after")
    def _ungrounded_actions_left_a_trail(self) -> "DefenseRuleReport":
        """A contagem e os achados contam a mesma história, ou o relatório mente.

        Sem isto, um relatório com ``grounded_actions`` inflado e nenhum achado
        passaria — e é exatamente esse número que a janela D47-54 cobra.
        """

        ungrounded = self.total_actions - self.grounded_actions
        blocking = sum(
            1 for finding in self.findings if finding.rule == "technique_not_grounded"
        )
        if blocking != ungrounded:
            raise ValueError(
                f"{ungrounded} ação(ões) sem lastro, mas "
                f"{blocking} achado(s) 'technique_not_grounded'."
            )

        return self

    @property
    def grounded_fraction(self) -> float:
        """Fração das ações cuja técnica responde à evidência citada."""

        return self.grounded_actions / self.total_actions

    @property
    def blocking_findings(self) -> tuple[RuleFinding, ...]:
        """Achados que recusam o plano, na ordem em que foram produzidos."""

        return tuple(
            finding for finding in self.findings if finding.severity == "blocking"
        )

    @property
    def is_grounded(self) -> bool:
        """Verdadeiro quando nada bloqueia — o gate "100%" da janela D47-54."""

        return not self.blocking_findings
