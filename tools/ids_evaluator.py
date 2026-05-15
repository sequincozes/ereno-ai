from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class IdsEvaluator:
    """
    Avaliador do IDS.

    Treina o Random Forest apenas no baseline e testa cada variante gerada
    sem retreinar o modelo. Também remove features constantes e features
    temporais/sequenciais que podem causar vazamento ou facilitar demais
    a classificação.
    """

    COLUMNS_TO_ALWAYS_DROP = [
        # Temporais / sequência / derivados muito fortes
        "Time",
        "t",
        "GooseTimestamp",
        "receivedTimestamp",
        "timestampDiff",
        "tDiff",
        "timeFromLastChange",
        "delay",
        "SqNum",
        "StNum",
        "sqDiff",
        "stDiff",

        # Metadados/constantes comuns do protocolo
        "frameLen",
        "ethDst",
        "ethSrc",
        "ethType",
        "gooseTimeAllowedtoLive",
        "gooseAppid",
        "gooseLen",
        "TPID",
        "gocbRef",
        "datSet",
        "goID",
        "test",
        "confRev",
        "ndsCom",
        "numDatSetEntries",
        "APDUSize",
        "protocol",
        "gooseLengthDiff",
        "apduSizeDiff",
        "frameLengthDiff",
        "e2eLatency",
    ]

    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        n_estimators: int = 100,
        drop_cb_status: bool = False,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.drop_cb_status = drop_cb_status

        self.model: RandomForestClassifier | None = None
        self.label_column: str | None = None
        self.feature_columns: list[str] | None = None
        self.feature_encoders: dict[str, dict[str, int]] = {}

        self.label_encoder: LabelEncoder | None = None
        self.class_mapping: dict[int, str] = {}
        self.attack_label: int | None = None

        self.removed_columns: list[str] = []

    def train_baseline(self, dataset_path: str) -> dict[str, Any]:
        print(f"[IDS] Treinando modelo baseline com: {dataset_path}")

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("Dataset baseline vazio.")

        self.label_column = self._find_label_column(df)
        print(f"[IDS] Coluna de classe detectada: {self.label_column}")

        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]

        X_prepared = self._fit_transform_features(X)
        y_encoded = self._fit_transform_labels(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded if len(set(y_encoded)) > 1 else None,
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        metrics = self._compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            dataset_path=dataset_path,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            evaluation_type="baseline_internal_test",
        )

        print("[IDS] Métricas do baseline:")
        print(metrics)

        return metrics

    def evaluate_variant(self, dataset_path: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Modelo ainda não treinado. Execute train_baseline() primeiro.")

        if self.label_column is None:
            raise RuntimeError("Coluna de classe ainda não definida.")

        print(f"[IDS] Testando variante com modelo baseline: {dataset_path}")

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("Dataset variante vazio.")

        if self.label_column not in df.columns:
            raise ValueError(f"Coluna de classe '{self.label_column}' não existe no dataset variante.")

        X = df.drop(columns=[self.label_column])
        y = df[self.label_column]

        X_prepared = self._transform_features(X)
        y_encoded = self._transform_labels(y)

        y_pred = self.model.predict(X_prepared)

        metrics = self._compute_metrics(
            y_true=y_encoded,
            y_pred=y_pred,
            dataset_path=dataset_path,
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            evaluation_type="variant_external_test",
        )

        print("[IDS] Métricas da variante:")
        print(metrics)

        return metrics

    def _find_label_column(self, df: pd.DataFrame) -> str:
        possible_names = [
            "class",
            "label",
            "classe",
            "target",
            "attack",
            "is_attack",
            "category",
            "type",
        ]

        lower_columns = {column.lower(): column for column in df.columns}

        for name in possible_names:
            if name.lower() in lower_columns:
                return lower_columns[name.lower()]

        return df.columns[-1]

    def _fit_transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        constant_cols = [
            col for col in X.columns
            if X[col].nunique(dropna=False) <= 1
        ]

        drop_cols = set(self.COLUMNS_TO_ALWAYS_DROP + constant_cols)

        if self.drop_cb_status:
            drop_cols.update(["cbStatus", "cbStatusDiff"])

        self.removed_columns = sorted([col for col in drop_cols if col in X.columns])

        print(f"[IDS] Colunas removidas: {self.removed_columns}")

        X = X.drop(columns=self.removed_columns, errors="ignore")
        X = X.dropna(axis=1, how="all")

        self.feature_columns = list(X.columns)

        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)
            else:
                X[column] = X[column].fillna("missing").astype(str)
                unique_values = sorted(X[column].unique())
                mapping = {value: index for index, value in enumerate(unique_values)}
                self.feature_encoders[column] = mapping
                X[column] = X[column].map(mapping).fillna(-1).astype(int)

        print(f"[IDS] Features usadas no modelo: {self.feature_columns}")

        return X

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError("Features do baseline ainda não foram definidas.")

        X = X.copy()
        X = X.drop(columns=self.removed_columns, errors="ignore")

        for column in self.feature_columns:
            if column not in X.columns:
                X[column] = 0

        X = X[self.feature_columns]

        for column in X.columns:
            if column in self.feature_encoders:
                mapping = self.feature_encoders[column]
                X[column] = (
                    X[column]
                    .fillna("missing")
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )
            else:
                X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)

        return X

    def _fit_transform_labels(self, y: pd.Series) -> pd.Series:
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y.astype(str))

        self.class_mapping = {
            int(index): str(label)
            for index, label in enumerate(self.label_encoder.classes_)
        }

        self.attack_label = self._detect_attack_label(self.class_mapping)

        print(f"[IDS] Mapeamento de classes: {self.class_mapping}")
        print(f"[IDS] Classe de ataque: {self.attack_label}")

        return pd.Series(y_encoded)

    def _transform_labels(self, y: pd.Series) -> pd.Series:
        if self.label_encoder is None:
            raise RuntimeError("LabelEncoder ainda não treinado.")

        known_labels = set(self.label_encoder.classes_)

        encoded_values = []
        for value in y.astype(str):
            if value in known_labels:
                encoded_values.append(int(self.label_encoder.transform([value])[0]))
            else:
                encoded_values.append(-1)

        return pd.Series(encoded_values)

    def _detect_attack_label(self, class_mapping: dict[int, str]) -> int:
        for encoded_label, original_label in class_mapping.items():
            normalized = original_label.lower()

            if (
                "masquerade" in normalized
                or "attack" in normalized
                or "malicious" in normalized
                or normalized in ["1", "true", "ataque"]
            ):
                return encoded_label

        return max(class_mapping.keys())

    def _get_feature_importances(self) -> list[dict[str, Any]]:
        if self.model is None or self.feature_columns is None:
            return []

        importances = self.model.feature_importances_

        ranking = sorted(
            [
                {
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(self.feature_columns, importances)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )

        return ranking[:15]

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        dataset_path: str,
        dataset_rows: int,
        dataset_columns: int,
        evaluation_type: str,
    ) -> dict[str, Any]:
        if self.attack_label is None:
            raise RuntimeError("Classe de ataque não definida.")

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[self.attack_label],
            average=None,
            zero_division=0,
        )

        return {
            "evaluation_type": evaluation_type,
            "dataset_path": dataset_path,
            "accuracy": float(accuracy),
            "attack_label_encoded": int(self.attack_label),
            "attack_label_original": self.class_mapping.get(int(self.attack_label)),
            "precision_masquerade": float(precision[0]),
            "recall_masquerade": float(recall[0]),
            "f1_score_masquerade": float(f1[0]),
            "support_masquerade": int(support[0]),
            "label_column": self.label_column,
            "class_mapping": self.class_mapping,
            "dataset_rows": int(dataset_rows),
            "dataset_columns": int(dataset_columns),
            "removed_columns": self.removed_columns,
            "used_features": self.feature_columns,
            "top_feature_importances": self._get_feature_importances(),
        }