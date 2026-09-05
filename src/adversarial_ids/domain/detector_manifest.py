"""DetectorManifest — estado do detector treinado (épico E8).

Terceiro manifest da mesma família, e o último elo da cadeia que o E6 e o E7
deixaram anunciada: ``FeatureManifest`` (E6) registra *como as features foram
preparadas*, ``SelectionManifest`` (E7) *quais sobreviveram e sobre quantas
linhas o treino aconteceu*, e este registra *qual detector aprendeu sobre esse
espaço, com quais hiperparâmetros*. Os três são persistidos lado a lado no
estágio DETECTOR do pipeline intent-driven (``feature_manifest.json``,
``selection_manifest.json``, ``detector_manifest.json``).

Por que um manifest e não apenas ``DetectionReport.model_name``: o relatório
responde "quão bem o detector foi"; o manifest responde "que detector era, e
sob que preparo de dados" — que é justamente o que a etapa D36-46 exige para
comparar RF/DT/SVM **sob o mesmo protocolo**. Sem ele, dois relatórios com
``model_name`` diferente não provam que rodaram sob o mesmo split, a mesma
escala e o mesmo espaço de features; com ele, a comparação é auditável a
partir dos artefatos em disco, sem reexecutar nada.

``importance_kind`` existe porque ``DetectionReport.top_features`` **não
significa a mesma coisa** entre detectores: importância Gini (RF/DT) e módulo
do coeficiente linear (SVM linear) são escalas diferentes, e o SVM RBF não
expõe importância nenhuma. Comparar ranking de features entre detectores só é
honesto depois de olhar este campo.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Vocabulário de detectores. Mora aqui, e não em ``core/detectors.py``, porque
# o domínio não depende do core — é o core que valida seu registro contra este
# contrato na importação, nunca o contrário. Consequência prática: a CLI lista
# as opções de ``--detector`` sem arrastar o scikit-learn para o import.
DetectorKey = Literal["random_forest", "decision_tree", "svm_linear", "svm_rbf"]

DETECTOR_KEYS: tuple[str, ...] = get_args(DetectorKey)

ImportanceKind = Literal["gini", "linear_coef", "none"]

# Qual tipo de importância cada detector é capaz de produzir. Um par
# (detector, importance_kind) fora desta tabela é um manifest que descreve um
# detector que não existe — daí ser um validador de consistência, não uma
# escolha configurável.
_IMPORTANCE_KIND_BY_DETECTOR: dict[str, str] = {
    "random_forest": "gini",
    "decision_tree": "gini",
    "svm_linear": "linear_coef",
    "svm_rbf": "none",
}

# Só modelos baseados em árvore são explicáveis por ``shap.TreeExplainer`` —
# o caminho best-effort que ``IdsEvaluator.get_shap_importances`` usa.
_TREE_DETECTORS = frozenset({"random_forest", "decision_tree"})


class DetectorManifest(BaseModel):
    """Detector treinado + o preparo de dados sob o qual ele treinou."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1

    detector: DetectorKey
    # Mesmo valor que vai para ``DetectionReport.model_name`` — é o que amarra
    # um relatório ao manifest do detector que o produziu.
    model_name: str = Field(min_length=1)
    # Hiperparâmetros efetivamente passados ao estimador do scikit-learn
    # (defaults do registro + overrides do chamador, já resolvidos).
    hyperparameters: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    random_state: int = 42

    # -- Preparo de dados sob o qual este detector treinou ------------------ #
    # ``requires_scaling`` é a recomendação do detector; ``resolved_scaler`` é
    # o que o ``FeaturePreprocessor`` de fato aplicou. Os dois discordarem é
    # legítimo (o chamador pode forçar) e por isso não é erro — mas fica
    # registrado, porque um SVM sem escala explica sozinho uma métrica ruim.
    requires_scaling: bool = False
    resolved_scaler: Literal["none", "standard"] = "none"

    # -- Capacidades de explicação ------------------------------------------ #
    importance_kind: ImportanceKind
    supports_shap: bool = False

    # -- Prova do que foi treinado ------------------------------------------ #
    # Linhas de TREINO efetivamente usadas em ``fit`` — já depois da seleção
    # de features e do undersampling (E7); nunca o dataset inteiro nem a
    # partição de teste. Mesma disciplina de ``FeatureManifest.fitted_rows``.
    trained_rows: int = Field(gt=0)
    trained_features: tuple[str, ...] = Field(min_length=1)
    train_duration_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _importance_kind_matches_detector(self) -> "DetectorManifest":
        expected = _IMPORTANCE_KIND_BY_DETECTOR[self.detector]
        if self.importance_kind != expected:
            raise ValueError(
                f"detector={self.detector!r} produz importance_kind={expected!r}, "
                f"não {self.importance_kind!r}."
            )
        return self

    @model_validator(mode="after")
    def _only_tree_detectors_support_shap(self) -> "DetectorManifest":
        if self.supports_shap and self.detector not in _TREE_DETECTORS:
            raise ValueError(
                f"supports_shap=True só vale para detectores de árvore "
                f"({sorted(_TREE_DETECTORS)!r}); detector={self.detector!r} não é um."
            )
        return self

    # Nota: um terceiro validador ("importance_kind='none' é incompatível com
    # supports_shap=True") seria código morto — os dois acima já rejeitam todas
    # as combinações que chegariam nele, porque só ``svm_rbf`` tem
    # importance_kind='none' e ele já não é detector de árvore. Verificado
    # exaustivamente sobre o produto detector × importance_kind × supports_shap.
