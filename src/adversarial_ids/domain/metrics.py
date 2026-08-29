from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeatureImportance(BaseModel):
    """Uma feature e sua importância no Random Forest.

    Espelha ``ids_evaluator._get_feature_importances`` — lista de dicts
    ``{"feature": str, "importance": float}``.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    importance: float


class Metrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    # --- identificação da avaliação ---
    evaluation_type: str | None = None
    dataset_path: str | None = None

    # --- desempenho do RF sobre a classe de ataque (masquerade) ---
    accuracy: float | None = None
    precision_attack: float | None = None
    recall_attack: float | None = None
    f1_score_attack: float | None = None
    support_attack: int | None = None

    # --- rótulo de ataque ---
    attack_label_encoded: int | None = None
    attack_label_original: str | None = None

    # --- contagens ---
    attack_count: int | None = None
    normal_count: int | None = None
    full_attack_count: int | None = None
    full_normal_count: int | None = None

    # --- matriz de confusão (None quando não é binária) ---
    tp: int | None = None
    fp: int | None = None
    fn: int | None = None
    tn: int | None = None

    # --- metadados do dataset / features ---
    label_column: str | None = None
    class_mapping: dict[int, str] = Field(default_factory=dict)
    dataset_rows: int | None = None
    dataset_columns: int | None = None
    removed_columns: list[str] = Field(default_factory=list)
    used_features: list[str] | None = None
    top_feature_importances: list[FeatureImportance] = Field(default_factory=list)

    # --- anotações adicionadas pelo loop (cli.py) ---
    config_changed: bool | None = None
    attack_count_ratio_vs_baseline: float | None = None
    degenerate_variant: bool | None = None
