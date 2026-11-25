"""DetectorComparison — o relatório comparativo do gate D36-46 (épico E8).

O E8 tornou RF/DT/SVM *comparáveis* (mesmo split, mesmo preparo, mesmo formato
de ``DetectionReport``); este contrato é o que efetivamente **os compara**. É a
segunda metade do gate de saída da janela — "mesmo split/protocolo; relatório
comparativo; fallback RF preservado" — e do critério de pronto do backlog,
"comparação justa e relatório único".

Três propriedades que o contrato carrega e que a prosa sozinha não garantiria:

1. **Ranking auditável.** ``ranking`` não é uma lista que o produtor afirma
   estar ordenada: um validador recomputa a ordem a partir de ``runs`` e da
   ``ranking_metric``, então um relatório com ranking inventado não valida.

2. **Falha isolada não derruba os demais.** Cada detector é uma linha com
   ``status`` próprio; um SVM que estourar memória vira ``failed`` com a causa
   registrada e os outros três continuam no relatório — exigência literal da
   tabela de aceite da camada Detecção.

3. **"Mesmo protocolo" é verificado, não prometido.** ``protocol_consistent``
   compara o espaço de features que cada detector realmente treinou. Importa
   porque a garantia tem um furo conhecido: com ``FEATURE_SELECTION_MODE=
   mutual_info`` a seleção é ajustada sobre o X já escalado, e
   ``mutual_info_classif`` não é estritamente invariante a escala — uma árvore
   (``scaler="none"``) e um SVM (``scaler="standard"``) podem, em princípio,
   acabar com conjuntos de features diferentes. Em vez de deixar isso como uma
   nota de rodapé em ``docs/detectors.md``, o relatório detecta e diz.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adversarial_ids.domain.detection_report import DetectionReport
from adversarial_ids.domain.detector_manifest import DetectorKey, ImportanceKind

# Métrica que ordena o ranking. As três que a janela D36-46 pede
# ("ranking por F1/recall/latência"); as outras duas sempre entram como
# critério de desempate, então o relatório nunca depende de uma só.
RankingMetric = Literal["f1", "recall", "latency_ms"]


class DetectorRun(BaseModel):
    """Uma linha do comparativo: um detector, sob o protocolo comum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detector: DetectorKey
    status: Literal["succeeded", "failed"]
    # Preenchidos só quando succeeded (ver validador de coerência abaixo).
    report: DetectionReport | None = None
    resolved_scaler: Literal["none", "standard"] | None = None
    importance_kind: ImportanceKind | None = None
    trained_rows: int | None = Field(default=None, gt=0)
    trained_features: tuple[str, ...] = ()
    # Preenchido só quando failed. Mesma disciplina de ``LoopStage.error``: a
    # causa fica no artefato, não só no stdout de quem rodou.
    error: str | None = None

    @model_validator(mode="after")
    def _status_matches_the_payload(self) -> "DetectorRun":
        if self.status == "succeeded":
            if self.report is None:
                raise ValueError("status='succeeded' exige um DetectionReport.")
            if self.error is not None:
                raise ValueError("status='succeeded' não pode carregar error.")
            if not self.trained_features:
                raise ValueError("status='succeeded' exige trained_features.")
        else:
            if self.report is not None:
                raise ValueError("status='failed' não pode carregar um DetectionReport.")
            if not self.error:
                raise ValueError("status='failed' exige a causa em error.")
        return self

    @model_validator(mode="after")
    def _report_names_this_detector(self) -> "DetectorRun":
        # Sem isto, um relatório poderia atribuir os números de um detector à
        # linha de outro — exatamente o erro que o E8 corrigiu em
        # build_detection_report (model_name era a constante "random_forest").
        if self.report is not None and self.report.model_name != self.detector:
            raise ValueError(
                f"o DetectionReport da linha {self.detector!r} diz "
                f"model_name={self.report.model_name!r}."
            )
        return self


class DetectorComparison(BaseModel):
    """Relatório comparativo de N detectores sobre o mesmo par de datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    baseline_dataset: str = Field(min_length=1)
    variant_dataset: str = Field(min_length=1)
    # Mesma string de ``DetectionReport.split``: é a prova, no artefato, de que
    # todos os detectores foram medidos sob a mesma partição.
    split: str = Field(min_length=1)

    ranking_metric: RankingMetric = "f1"
    runs: tuple[DetectorRun, ...] = Field(min_length=1)
    # Chaves dos detectores que rodaram, do melhor para o pior. Só os
    # bem-sucedidos: um detector que falhou não tem posição no ranking.
    ranking: tuple[str, ...] = ()

    protocol_consistent: bool = True
    protocol_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _each_detector_appears_once(self) -> "DetectorComparison":
        keys = [run.detector for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"detector repetido em runs: {sorted(keys)!r}.")
        return self

    @model_validator(mode="after")
    def _every_report_shares_the_declared_split(self) -> "DetectorComparison":
        divergent = {
            run.detector: run.report.split
            for run in self.runs
            if run.report is not None and run.report.split != self.split
        }
        if divergent:
            raise ValueError(
                f"split declarado {self.split!r} não bate com o de {divergent!r} — "
                "sem partição comum não há comparação."
            )
        return self

    @model_validator(mode="after")
    def _ranking_is_the_recomputed_order(self) -> "DetectorComparison":
        expected = rank_runs(self.runs, metric=self.ranking_metric)
        if self.ranking != expected:
            raise ValueError(
                f"ranking={self.ranking!r} não corresponde à ordem recomputada "
                f"por {self.ranking_metric!r} ({expected!r})."
            )
        return self

    @model_validator(mode="after")
    def _protocol_flag_matches_the_feature_spaces(self) -> "DetectorComparison":
        spaces = {run.trained_features for run in self.runs if run.status == "succeeded"}
        if self.protocol_consistent and len(spaces) > 1:
            raise ValueError(
                "protocol_consistent=True mas os detectores treinaram sobre espaços "
                f"de features diferentes ({len(spaces)} conjuntos distintos)."
            )
        if not self.protocol_consistent and not self.protocol_notes:
            raise ValueError(
                "protocol_consistent=False exige protocol_notes explicando a divergência."
            )
        return self

    @property
    def winner(self) -> str | None:
        """Melhor detector pela ``ranking_metric``, ou ``None`` se nenhum rodou."""
        return self.ranking[0] if self.ranking else None


def rank_runs(runs: tuple[DetectorRun, ...], *, metric: RankingMetric = "f1") -> tuple[str, ...]:
    """Ordena os detectores bem-sucedidos, do melhor para o pior.

    As três métricas que a janela D36-46 pede entram sempre: a escolhida
    manda, as outras duas desempatam. F1 e recall são "maior é melhor",
    latência é "menor é melhor". O último critério é o nome do detector — um
    empate exato nas três (comum entre RF e DT num dataset fácil, onde os dois
    acertam tudo) não pode depender da ordem de iteração de quem construiu a
    lista, ou o mesmo experimento produziria rankings diferentes.
    """

    def sort_key(run: DetectorRun) -> tuple[float, float, float, str]:
        assert run.report is not None  # garantido pelo filtro abaixo
        f1, recall, latency = -run.report.f1, -run.report.recall, run.report.latency_ms
        if metric == "recall":
            primary: tuple[float, float, float] = (recall, f1, latency)
        elif metric == "latency_ms":
            primary = (latency, f1, recall)
        else:
            primary = (f1, recall, latency)
        return (*primary, run.detector)

    succeeded = [run for run in runs if run.status == "succeeded" and run.report is not None]
    return tuple(run.detector for run in sorted(succeeded, key=sort_key))
