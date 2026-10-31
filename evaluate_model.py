import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


DATA_PATH = os.path.join("Data", "lung_cancer.csv")
MODEL_PATH = os.path.join("models", "gaussian_nb.pkl")
METADATA_PATH = os.path.join("models", "gaussian_nb_metadata.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().dropna()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].map({"F": 0, "M": 1})

    X = df.drop(columns=["LUNG_CANCER"])
    y = df["LUNG_CANCER"].astype(int)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    return X, y


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy:   {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision:  {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:   {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["NO (0)", "YES (1)"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    evaluate()
