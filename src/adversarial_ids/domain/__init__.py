"""domain — modelos tipados (Pydantic) compartilhados entre os agentes.

Contratos de I/O congelados na Fase 0 — a fronteira exata entre os membros.

Schemas base (issue #2, M3):
  - attack_configs/*  — um schema por ataque ERENO registrado (ver
                        ``domain/attack_configs/__init__.py``)
  - Metrics           — saída da avaliação do IDS (Random Forest)
  - IterationRecord   — unidade do histórico; costura os 4 schemas

Schemas dos agentes (co-definidos pelos donos):
  - #4  StrategistOutput  (M1)  — congelado; plugado em IterationRecord.strategist_output
  - #5  AnalystOutput     (M2)  — a plugar em IterationRecord.analyst_output
"""

from adversarial_ids.domain.analyst_output import (
    AnalystOutput,
    DeceptiveFeature,
    Mitigation,
)
from adversarial_ids.domain.attack_candidate import AttackCandidate, FieldChange
from adversarial_ids.domain.attack_configs import (
    AnalogConfig,
    CONFIG_MODEL_BY_ATTACK,
    DelayedReplayBackoffConfig,
    DelayedReplayBatchDumpConfig,
    DelayedReplayConfig,
    DelayedReplayDoubleDropConfig,
    FaultConfig,
    FloodingConfig,
    GrayholeConfig,
    HighStNumInjectionConfig,
    InjectionConfig,
    InverseReplayConfig,
    MasqueradeFaultConfig,
    RandomReplayConfig,
    TrapAreaConfig,
    config_model_for,
)
from adversarial_ids.domain.dataset_bundle import DatasetBundle
from adversarial_ids.domain.defense_plan import DefenseAction, DefensePlan, Evidence
from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.domain.feedback_decision import (
    FeedbackDecision,
    FeedbackStopReason,
    RoundOutcome,
)
from adversarial_ids.domain.feature_manifest import (
    DroppedColumn,
    FeatureManifest,
    ScalerStat,
)
from adversarial_ids.domain.selection_manifest import SelectionManifest
from adversarial_ids.domain.iteration_record import IterationRecord
from adversarial_ids.domain.intent_spec import (
    MAX_FIELDS_CHANGED_CEILING,
    DesiredEffect,
    IntentIntensity,
    IntentObjective,
    IntentRestrictions,
    IntentSpec,
)
from adversarial_ids.domain.loop_record import LoopRecord, LoopStage, LoopStageStatus
from adversarial_ids.domain.metrics import FeatureImportance, Metrics

from adversarial_ids.domain.strategist_output import (  # M1 · issue #4
    Change,
    StrategistOutput,
)

__all__ = [
    "AnalystOutput",
    "DeceptiveFeature",
    "Mitigation",
    "MasqueradeFaultConfig",
    "FaultConfig",
    "AnalogConfig",
    "TrapAreaConfig",
    "RandomReplayConfig",
    "InverseReplayConfig",
    "InjectionConfig",
    "HighStNumInjectionConfig",
    "FloodingConfig",
    "GrayholeConfig",
    "DelayedReplayConfig",
    "DelayedReplayBackoffConfig",
    "DelayedReplayBatchDumpConfig",
    "DelayedReplayDoubleDropConfig",
    "CONFIG_MODEL_BY_ATTACK",
    "config_model_for",
    "Change",
    "StrategistOutput",
    "Metrics",
    "FeatureImportance",
    "IterationRecord",
    "IntentSpec",
    "IntentRestrictions",
    "MAX_FIELDS_CHANGED_CEILING",
    "IntentObjective",
    "DesiredEffect",
    "IntentIntensity",
    "AttackCandidate",
    "FieldChange",
    "DatasetBundle",
    "DetectionReport",
    "ConfusionMatrix",
    "DefensePlan",
    "DefenseAction",
    "Evidence",
    "LoopRecord",
    "LoopStage",
    "LoopStageStatus",
    "FeedbackDecision",
    "FeedbackStopReason",
    "RoundOutcome",
    "FeatureManifest",
    "DroppedColumn",
    "ScalerStat",
    "SelectionManifest",
]
