import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder


BASELINE_PATH = "outputs/baseline_masquerade_dataset.csv"
ADVERSARIAL_PATH = "outputs/adversarial_masquerade_01_dataset.csv"

LABEL_COL = "class"

DROP_COLS = [
    LABEL_COL,

    # Temporais / sequência / derivados fortes
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


def prepare_features_train(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    y = df[LABEL_COL].astype(str)

    feature_columns = list(X.columns)
    encoders = {}

    for col in X.columns:
        if X[col].dtype == "object":
            values = sorted(X[col].fillna("missing").astype(str).unique())
            mapping = {value: idx for idx, value in enumerate(values)}
            encoders[col] = mapping
            X[col] = X[col].fillna("missing").astype(str).map(mapping).fillna(-1).astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, feature_columns, encoders, label_encoder


def prepare_features_test(
    df: pd.DataFrame,
    feature_columns: list[str],
    encoders: dict,
    label_encoder: LabelEncoder,
):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    y = df[LABEL_COL].astype(str)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    for col in X.columns:
        if col in encoders:
            mapping = encoders[col]
            X[col] = X[col].fillna("missing").astype(str).map(mapping).fillna(-1).astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y_encoded = label_encoder.transform(y)

    return X, y_encoded


def print_dataset_info(name: str, df: pd.DataFrame):
    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)
    print("Linhas:", len(df))
    print("Colunas:", len(df.columns))
    print("\nDistribuição da classe:")
    print(df[LABEL_COL].value_counts())


def main():
    baseline_df = pd.read_csv(BASELINE_PATH)
    adversarial_df = pd.read_csv(ADVERSARIAL_PATH)

    print_dataset_info("BASELINE DATASET", baseline_df)
    print_dataset_info("ADVERSARIAL DATASET", adversarial_df)

    X_train, y_train, feature_columns, encoders, label_encoder = prepare_features_train(
        baseline_df
    )

    X_adv, y_adv = prepare_features_test(
        adversarial_df,
        feature_columns,
        encoders,
        label_encoder,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTreinando Random Forest no baseline inteiro...")
    model.fit(X_train, y_train)

    print("Testando no dataset adversarial...")
    y_pred = model.predict(X_adv)

    classes = list(label_encoder.classes_)
    attack_label = list(label_encoder.classes_).index("masquerade_fake_fault")

    accuracy = accuracy_score(y_adv, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_adv,
        y_pred,
        labels=[attack_label],
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_adv, y_pred)

    print("\n" + "=" * 40)
    print("RESULTADOS ADVERSARIAIS")
    print("=" * 40)

    print("Classes:", classes)
    print("\nAccuracy:", accuracy)

    print("\nMatriz de confusão:")
    print(cm)

    print("\nRelatório completo:")
    print(classification_report(y_adv, y_pred, target_names=classes))

    print("\nMétricas da classe masquerade_fake_fault:")
    print("Precision:", float(precision[0]))
    print("Recall:", float(recall[0]))
    print("F1-score:", float(f1[0]))
    print("Support:", int(support[0]))

    if len(classes) == 2:
        attack_index = attack_label
        normal_index = 1 - attack_index

        tp = int(cm[attack_index][attack_index])
        fn = int(cm[attack_index][normal_index])
        fp = int(cm[normal_index][attack_index])
        tn = int(cm[normal_index][normal_index])

        print("\nTP:", tp)
        print("FN:", fn)
        print("FP:", fp)
        print("TN:", tn)

    importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 20 features:")
    print(importances.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
