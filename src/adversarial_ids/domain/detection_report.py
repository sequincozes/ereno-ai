"""DetectionReport — saída padronizada de um detector sobre um DatasetBundle.

Contrato congelado (ação 72h #2). Formato único de saída para qualquer
detector (RF, DT, SVM, ...), exigido pela etapa D36-46 ("mesmo split/protocolo
para todos os detectores"). Reaproveita ``FeatureImportance`` do módulo
``metrics`` legado — a métrica em si é a mesma noção, só o formato do
relatório muda.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adversarial_ids.domain.metrics import FeatureImportance


class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)


class DetectionReport(BaseModel):
    """Resultado de um detector, sob o mesmo protocolo de teste dos demais."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model_name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    confusion_matrix: ConfusionMatrix
    top_features: tuple[FeatureImportance, ...] = ()
    latency_ms: float = Field(ge=0.0)
