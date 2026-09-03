"""build_detection_report — DetectionReport a partir do IdsEvaluator (E3/E4).

Nenhuma lógica de detecção nova nesta entrega: reaproveita o mesmo Random
Forest do loop legado (comparar RF/DT/SVM sob o mesmo protocolo é o épico
E8, D36-46). Este módulo só traduz o dict solto que ``IdsEvaluator`` produz
para o contrato ``DetectionReport`` e mede a latência da chamada de
avaliação — o resto (accuracy/precision/recall/f1, matriz, importâncias) já
vem calculado pelo avaliador existente.
"""

from __future__ import annotations

import time

from adversarial_ids.core.ids_evaluator import IdsEvaluator
from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.domain.metrics import FeatureImportance


class DetectionReportError(ValueError):
    """A avaliação não produziu uma matriz de confusão binária utilizável."""


def build_detection_report(
    evaluator: IdsEvaluator,
    dataset_path: str,
    *,
    split: str,
    model_name: str = "random_forest",
) -> DetectionReport:
    """Avalia ``dataset_path`` no modelo já treinado e empacota o resultado.

    Requer ``evaluator`` já treinado (``train_baseline`` chamado antes) e um
    ``dataset_path`` com exatamente duas classes no split de teste (ataque
    vs. normal) — é o que o gate do E4 (``build_dataset_bundle``) garante
    antes deste ponto. Se ainda assim ``tp/fp/fn/tn`` vier incompleto (mais
    de duas classes efetivas no split), levanta ``DetectionReportError`` em
    vez de produzir um relatório com a matriz de confusão faltando.
    """

    start = time.perf_counter()
    metrics = evaluator.evaluate_variant(dataset_path)
    latency_ms = (time.perf_counter() - start) * 1000.0

    tp, fp, fn, tn = metrics.get("tp"), metrics.get("fp"), metrics.get("fn"), metrics.get("tn")
    if None in (tp, fp, fn, tn):
        raise DetectionReportError(
            "A avaliação não produziu uma matriz de confusão binária "
            f"({dataset_path}): tp/fp/fn/tn incompletos. O DetectionReport "
            "exige exatamente duas classes efetivas (ataque vs. normal) no "
            "split de teste."
        )

    top_features = tuple(
        FeatureImportance.model_validate(item)
        for item in metrics.get("top_feature_importances", [])[:15]
    )

    return DetectionReport(
        model_name=model_name,
        split=split,
        accuracy=metrics["accuracy"],
        precision=metrics["precision_attack"],
        recall=metrics["recall_attack"],
        f1=metrics["f1_score_attack"],
        confusion_matrix=ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn),
        top_features=top_features,
        latency_ms=latency_ms,
    )
