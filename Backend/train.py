# ==========================================================
# LungDetect AI
# File: train.py
# ==========================================================

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import joblib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ==========================================================
# Create folders
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# Load processed data
# ==========================================================

print("=" * 50)
print("Loading Processed Dataset")
print("=" * 50)

X_train = joblib.load(DATA_DIR / "X_train.pkl")
X_test = joblib.load(DATA_DIR / "X_test.pkl")
y_train = joblib.load(DATA_DIR / "y_train.pkl")
y_test = joblib.load(DATA_DIR / "y_test.pkl")

print("Dataset Loaded Successfully.\n")

# ==========================================================
# Models
# ==========================================================

models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        random_state=42
    ),
}

model_files = {
    "Gaussian Naive Bayes": MODELS_DIR / "gaussian_nb.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
}

# ==========================================================
# Training
# ==========================================================

results = []

best_model = None
best_name = ""
best_score = (-1, -1)
best_accuracy = 0

print("=" * 50)
print("Training Models")
print("=" * 50)

max_train_samples = 100000

if len(X_train) > max_train_samples:
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(X_train), size=max_train_samples, replace=False)
    X_train_fit = X_train[train_indices]
    y_train_fit = y_train.iloc[train_indices] if hasattr(y_train, "iloc") else y_train[train_indices]
    print(f"Training on a sample of {max_train_samples} rows to keep runtime manageable.")
else:
    X_train_fit = X_train
    y_train_fit = y_train

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_fit, y_train_fit)
    joblib.dump(model, model_files[name])

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(y_test, prediction, zero_division=0)

    recall = recall_score(y_test, prediction, zero_division=0)

    f1 = f1_score(y_test, prediction, zero_division=0)

    auc = roc_auc_score(y_test, probability)

    cv_X = X_train_fit
    cv_y = y_train_fit
    cv_splits = 3

    if len(X_train_fit) > 10000:
        cv_X = X_train_fit[:10000]
        cv_y = y_train_fit[:10000]
        cv_splits = 2

    cv = cross_val_score(
        model,
        cv_X,
        cv_y,
        cv=cv_splits,
        scoring="accuracy",
        n_jobs=1
    ).mean()

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        auc,
        cv
    ])

    print("Accuracy :", round(accuracy,4))
    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))
    print("ROC AUC  :", round(auc,4))
    print("CV Score :", round(cv,4))

    # Save best model

    selection_score = (f1, auc)

    if selection_score > best_score:

        best_accuracy = accuracy
        best_model = model
        best_name = name
        best_score = selection_score

# ==========================================================
# Save best model
# ==========================================================

joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
joblib.dump({"name": best_name}, MODELS_DIR / "best_model_metadata.pkl")

print("\n")
print("=" * 50)
print("Best Model")
print("=" * 50)

print(best_name)
print("Accuracy:", round(best_accuracy,4))

# ==========================================================
# Comparison Table
# ==========================================================

columns = [

    "Model",

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "ROC_AUC",

    "CV"

]

result_df = pd.DataFrame(
    results,
    columns=columns
)

result_df = result_df.sort_values(
    by=["F1", "ROC_AUC"],
    ascending=False
)

print("\n")
print(result_df)

result_df.to_csv(
    MODELS_DIR / "model_results.csv",
    index=False
)

# ==========================================================
# Confusion Matrix
# ==========================================================

prediction = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d"
)

plt.title(best_name + " Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(PLOTS_DIR / "confusion_matrix.png")

plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

probability = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(
    y_test,
    probability
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label="ROC Curve"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(best_name + " ROC Curve")

plt.legend()

plt.savefig(PLOTS_DIR / "roc_curve.png")

plt.close()

# ==========================================================
# Classification Report
# ==========================================================

print("\n")
print("=" * 50)
print("Classification Report")
print("=" * 50)

print(
    classification_report(
        y_test,
        prediction
    )
)

print("\n")
print("=" * 50)
print("Training Completed Successfully")
print("=" * 50)
