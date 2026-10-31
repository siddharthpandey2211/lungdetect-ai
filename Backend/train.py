
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
DATA_PATH = os.path.join("Data", "lung_cancer.csv")
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "gaussian_nb.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "gaussian_nb_metadata.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load and clean the ORIGINAL dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Remove accidental spaces in column names
df.columns = df.columns.str.strip()

target = "LUNG_CANCER"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove duplicate rows and rows with missing values
df = df.drop_duplicates().dropna().reset_index(drop=True)

# Normalize text values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.upper()

# Convert the target to binary labels
target_mapping = {"NO": 0, "YES": 1}
df[target] = df[target].map(target_mapping)

if df[target].isna().any():
    raise ValueError("Target must contain only YES and NO.")

X = df.drop(columns=[target]).copy()
y = df[target].astype(int)

# Convert gender to a numeric feature
if "GENDER" in X.columns:
    X["GENDER"] = X["GENDER"].map({"F": 0, "M": 1})

    if X["GENDER"].isna().any():
        raise ValueError("GENDER must contain only F or M.")

# Ensure all remaining input columns are numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="raise")

# Record valid input rules from the training data
metadata = {
    "features": X.columns.tolist(),
    "target_mapping": target_mapping,
    "gender_mapping": {"F": 0, "M": 1},
    "min_values": X.min().to_dict(),
    "max_values": X.max().to_dict(),
    # Exact observed values for categorical/binary features
    "allowed_values": {
        col: sorted(X[col].unique().tolist())
        for col in X.columns
        if col != "AGE"
    },
}

# --------------------------------------------------
# 3. Split data for evaluation
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Train Gaussian Naive Bayes
# --------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Evaluate on held-out test data
# --------------------------------------------------
y_pred = model.predict(X_test)

print("\n===== GAUSSIAN NAIVE BAYES RESULTS =====")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["NO (0)", "YES (1)"],
    zero_division=0
))

# --------------------------------------------------
# 6. Refit on all cleaned data for deployment
# --------------------------------------------------
final_model = GaussianNB()
final_model.fit(X, y)

joblib.dump(final_model, MODEL_PATH)
joblib.dump(metadata, METADATA_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Validation rules saved to:", METADATA_PATH)
print("Training complete.")
