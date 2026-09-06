"""DefensePlan — recomendação defensiva rastreável até evidência.

Contrato congelado (ação 72h #2), completado na janela D47-54 ("Defesa
acionável"). Endereça o gap P0 "defesa é recomendação curta" do diagnóstico:
toda ação exige evidência (métrica ou feature concreta), uma **técnica
defensiva nomeada** e um **teste de validação executável** — nunca uma
alegação solta. O contrato não tem campo de execução: a validação/aceite do
plano é explícita ("sem aplicar defesas automaticamente"), então não há como
uma ``DefenseAction`` disparar nada sozinha.

``schema_version`` foi de 1 para 2 nesta janela. A v1 amarrava cada ação a um
``validation_method: str`` de texto livre — qualquer string não vazia passava,
inclusive "revalidar". O gate de saída da janela é "100% das recomendações
ligadas a evidência **e teste de validação**", e metade dele não era
verificável por tipo nenhum. A v2 troca aquele campo por ``ValidationTest``,
que nomeia a métrica a remedir, a direção esperada e o alvo numérico. A prosa
do método não se perdeu: virou ``ValidationTest.procedure``.

Assim como ``DetectorKey`` mora em ``detector_manifest`` e não em
``core/detectors.py``, o vocabulário de técnicas (``DefenseTechnique``) mora
aqui: o catálogo feature→técnica em ``config/defense_techniques.py`` e o
prompt do Defensor derivam dele, e não o contrário. O domínio não importa
``config`` — é o config que se declara em sincronia com ``DEFENSE_TECHNIQUES``.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Os três baldes do plano, nomeados pelo campo que os carrega em DefensePlan —
# assim uma mensagem de erro cita exatamente o campo que o autor precisa mexer.
DefenseBucket = Literal[
    "detection_actions",
    "containment_actions",
    "hardening_actions",
]
DEFENSE_BUCKETS: tuple[str, ...] = get_args(DefenseBucket)

DefenseTechnique = Literal[
    # Detecção — melhoram a capacidade de perceber o ataque.
    "physical_consistency_check",
    "goose_sequence_validation",
    "goose_timing_analysis",
    "detector_threshold_tuning",
    "detector_retraining",
    "feature_engineering",
    "class_rebalancing",
    # Contenção — limitam o impacto de um ataque que passou.
    "operator_alerting",
    "device_quarantine",
    "network_segmentation",
    "traffic_rate_limiting",
    # Hardening — mudanças estruturais de longo prazo.
    "goose_authentication",
    "publisher_binding",
    "dataset_enrichment",
    "continuous_monitoring",
]
DEFENSE_TECHNIQUES: tuple[str, ...] = get_args(DefenseTechnique)

# Em que baldes cada técnica é legítima. Várias são genuinamente de duplo uso
# (uma allowlist de publisher bloqueia o intruso *e* é controle estrutural), e
# forçar um balde único rejeitaria plano correto. O que este mapa impede é o
# erro que a LLM realmente comete: propor `network_segmentation` como ação de
# *detecção*, onde ela não mede nada.
_BUCKETS_BY_TECHNIQUE: dict[str, frozenset[str]] = {
    # Confrontar a grandeza elétrica declarada com a física do evento: uma falta
    # real move as três fases de forma correlacionada e casa com a operação do
    # disjuntor. É a única técnica que responde às 18 features analógicas, que
    # são quase todo o espaço de features que sobrevive ao preprocessador.
    "physical_consistency_check": frozenset(
        {"detection_actions", "hardening_actions"}
    ),
    "goose_sequence_validation": frozenset({"detection_actions", "hardening_actions"}),
    "goose_timing_analysis": frozenset({"detection_actions"}),
    "detector_threshold_tuning": frozenset({"detection_actions"}),
    "detector_retraining": frozenset({"detection_actions"}),
    "feature_engineering": frozenset({"detection_actions"}),
    "class_rebalancing": frozenset({"detection_actions"}),
    "operator_alerting": frozenset({"containment_actions"}),
    "device_quarantine": frozenset({"containment_actions"}),
    "network_segmentation": frozenset({"containment_actions", "hardening_actions"}),
    "traffic_rate_limiting": frozenset({"containment_actions", "hardening_actions"}),
    "goose_authentication": frozenset({"hardening_actions"}),
    "publisher_binding": frozenset({"containment_actions", "hardening_actions"}),
    "dataset_enrichment": frozenset({"hardening_actions"}),
    "continuous_monitoring": frozenset({"hardening_actions"}),
}

if set(_BUCKETS_BY_TECHNIQUE) != set(DEFENSE_TECHNIQUES):  # pragma: no cover
    raise RuntimeError(
        "_BUCKETS_BY_TECHNIQUE está fora de sincronia com DefenseTechnique: "
        f"faltando={sorted(set(DEFENSE_TECHNIQUES) - set(_BUCKETS_BY_TECHNIQUE))}, "
        f"sobrando={sorted(set(_BUCKETS_BY_TECHNIQUE) - set(DEFENSE_TECHNIQUES))}."
    )

_UNKNOWN_BUCKETS = {
    bucket
    for buckets in _BUCKETS_BY_TECHNIQUE.values()
    for bucket in buckets
    if bucket not in DEFENSE_BUCKETS
}
if _UNKNOWN_BUCKETS:  # pragma: no cover
    raise RuntimeError(f"Balde inexistente em _BUCKETS_BY_TECHNIQUE: {_UNKNOWN_BUCKETS}")


def techniques_for_bucket(bucket: str) -> tuple[str, ...]:
    """Técnicas legítimas num balde, em ordem estável.

    O prompt do Defensor monta sua lista de opções a partir daqui, e o
    validador do contrato recusa a partir do mesmo mapa — os dois lados não
    têm como divergir sobre o que é uma técnica legal, que é a mesma
    disciplina de ``select_evidence_candidates`` para evidência.
    """

    if bucket not in DEFENSE_BUCKETS:
        raise ValueError(
            f"Balde desconhecido: {bucket!r}. Esperado um de {DEFENSE_BUCKETS}."
        )

    return tuple(
        technique
        for technique in DEFENSE_TECHNIQUES
        if bucket in _BUCKETS_BY_TECHNIQUE[technique]
    )


# Métricas que um teste de validação pode remedir. É um subconjunto estrito da
# allowlist de evidência: `model_name` e `split` são texto (não há "aumentar o
# split"), e importância de feature não é resultado defensivo — provar que uma
# feature ficou mais importante não prova que o ataque passou a ser detectado.
# `agents/defender/tools.py` testa que todas estas chaves existem mesmo num
# DetectionReport, para as duas listas não se separarem no tempo.
ValidationMetric = Literal[
    "accuracy",
    "precision",
    "recall",
    "f1",
    "latency_ms",
    "confusion_matrix.tp",
    "confusion_matrix.fp",
    "confusion_matrix.fn",
    "confusion_matrix.tn",
]
VALIDATION_METRICS: tuple[str, ...] = get_args(ValidationMetric)

# `increase`/`decrease` cobram *mudança* em relação ao valor medido hoje;
# `at_least`/`at_most` cobram um piso/teto absoluto. A distinção importa: "subir
# o recall" e "não deixar a precisão cair abaixo de 0.90" são exigências
# diferentes, e um plano sério costuma ter as duas ao mesmo tempo.
ValidationDirection = Literal["increase", "decrease", "at_least", "at_most"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_or_feature: str = Field(min_length=1)
    value: float | int | str
    detection_report_ref: str | None = None


class ValidationTest(BaseModel):
    """Como saber se a ação funcionou — em termos remediáveis, não em prosa.

    Substitui o ``validation_method: str`` da v1. Um teste precisa dizer o que
    remedir (``metric``), o que se espera dele (``direction`` + ``target``) e
    sobre que dados (``split``); ``procedure`` continua carregando o "como" em
    linguagem natural, que é útil para quem executa mas não é verificável.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: ValidationMetric
    direction: ValidationDirection
    # `allow_inf_nan=False`: um alvo NaN passa em qualquer comparação numérica
    # e transforma o teste em algo infalsificável — exatamente o que a v2 veio
    # eliminar. Sem limite superior porque a métrica pode ser uma contagem da
    # matriz de confusão ou latência em ms, não só uma taxa em [0, 1].
    target: float = Field(ge=0.0, allow_inf_nan=False)
    split: str | None = None
    procedure: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _rate_metrics_stay_within_their_range(self) -> "ValidationTest":
        """Um alvo de 1.5 para o F1 é uma promessa impossível, não uma meta."""

        if self.metric in ("accuracy", "precision", "recall", "f1") and self.target > 1.0:
            raise ValueError(
                f"Alvo fora do domínio de {self.metric}: {self.target} "
                "(métricas de taxa vivem em [0, 1])."
            )

        return self


class DefenseAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1, max_length=1000)
    technique: DefenseTechnique
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    validation_test: ValidationTest


class DefensePlan(BaseModel):
    """Plano defensivo fundamentado; nunca aplicado automaticamente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    priority: Literal["low", "medium", "high", "critical"]
    # A v1 só tinha a referência dentro de cada Evidence, e opcional. O plano é
    # persistido como `defense_plan.json` avulso pelo orquestrador: sem uma
    # referência no topo, o arquivo sozinho não diz de que execução ele saiu.
    detection_report_ref: str | None = None
    detection_actions: tuple[DefenseAction, ...] = ()
    containment_actions: tuple[DefenseAction, ...] = ()
    hardening_actions: tuple[DefenseAction, ...] = ()

    def _buckets(self) -> tuple[tuple[str, tuple[DefenseAction, ...]], ...]:
        return (
            ("detection_actions", self.detection_actions),
            ("containment_actions", self.containment_actions),
            ("hardening_actions", self.hardening_actions),
        )

    @model_validator(mode="after")
    def _has_at_least_one_action(self) -> "DefensePlan":
        if not (
            self.detection_actions
            or self.containment_actions
            or self.hardening_actions
        ):
            raise ValueError(
                "DefensePlan precisa de ao menos uma ação de detecção, "
                "contenção ou hardening."
            )
        return self

    @model_validator(mode="after")
    def _technique_belongs_to_its_bucket(self) -> "DefensePlan":
        """Segmentar a rede não detecta nada; retreinar não contém nada."""

        for bucket, actions in self._buckets():
            for action in actions:
                if bucket not in _BUCKETS_BY_TECHNIQUE[action.technique]:
                    permitidos = ", ".join(
                        sorted(_BUCKETS_BY_TECHNIQUE[action.technique])
                    )
                    raise ValueError(
                        f"Técnica {action.technique!r} não pertence a {bucket}; "
                        f"baldes válidos para ela: {permitidos}."
                    )

        return self

    @model_validator(mode="after")
    def _evidence_refs_agree_with_the_plan_ref(self) -> "DefensePlan":
        """Duas referências no mesmo plano apontando para execuções diferentes
        significam que alguém montou o plano com evidência de outra rodada."""

        if self.detection_report_ref is None:
            return self

        for bucket, actions in self._buckets():
            for action in actions:
                for evidence in action.evidence:
                    ref = evidence.detection_report_ref
                    if ref is not None and ref != self.detection_report_ref:
                        raise ValueError(
                            f"Evidência de {evidence.metric_or_feature!r} em "
                            f"{bucket} referencia {ref!r}, mas o plano declara "
                            f"{self.detection_report_ref!r}."
                        )

        return self
