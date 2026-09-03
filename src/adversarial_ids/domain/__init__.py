"""domain — modelos tipados (Pydantic) compartilhados entre os agentes.

Contratos de I/O congelados na Fase 0 — a fronteira exata entre os membros.

Schemas base (issue #2, M3):
  - AttackConfig      — configuração do ataque sintético (12 campos editáveis)
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
from adversarial_ids.domain.attack_config import (
    AnalogConfig,
    AttackConfig,
    DeltaAbs,
    DurationMs,
    FaultConfig,
    Multiplier,
    TrapAreaConfig,
)
from adversarial_ids.domain.dataset_bundle import DatasetBundle
from adversarial_ids.domain.defense_plan import DefenseAction, DefensePlan, Evidence
from adversarial_ids.domain.detection_report import ConfusionMatrix, DetectionReport
from adversarial_ids.domain.iteration_record import IterationRecord
from adversarial_ids.domain.intent_spec import (
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
    "AttackConfig",
    "FaultConfig",
    "DurationMs",
    "AnalogConfig",
    "DeltaAbs",
    "TrapAreaConfig",
    "Multiplier",
    "Change",
    "StrategistOutput",
    "Metrics",
    "FeatureImportance",
    "IterationRecord",
    "IntentSpec",
    "IntentRestrictions",
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
]
