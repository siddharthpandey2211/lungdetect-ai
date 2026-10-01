# ==========================================================
# LungDetect AI
# File: predict.py
# ==========================================================

from pathlib import Path
import joblib
import pandas as pd

# ==========================================================
# Load Model
# ==========================================================

print("=" * 50)
print("Loading AI Model...")
print("=" * 50)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "Data"
PROCESSED_DATASET_PATH = DATA_DIR / "processed.csv"

model = joblib.load(MODELS_DIR / "gaussian_nb.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
encoders = joblib.load(MODELS_DIR / "encoders.pkl")
model_name = "Gaussian Naive Bayes"

TARGET_COLUMN = None
FEATURE_COLUMNS = []

if PROCESSED_DATASET_PATH.exists():
    processed_df = pd.read_csv(PROCESSED_DATASET_PATH)
    for candidate in ["LUNG_CANCER", "survived"]:
        if candidate in processed_df.columns:
            TARGET_COLUMN = candidate
            break

    if TARGET_COLUMN is not None:
        FEATURE_COLUMNS = [column for column in processed_df.columns if column != TARGET_COLUMN]
    else:
        FEATURE_COLUMNS = list(processed_df.columns)

print("Model Loaded Successfully!")

# ==========================================================
# Prediction Function
# ==========================================================

def _normalize_patient_data(patient_data):
    normalized = {}
    for key, value in patient_data.items():
        canonical_key = str(key).strip().lower().replace(" ", "_").replace("-", "_")
        normalized[canonical_key] = value

    fallback_values = {
        "gender": "M",
        "age": 50,
        "smoking": 1,
        "yellow_fingers": 1,
        "anxiety": 1,
        "peer_pressure": 1,
        "chronic_disease": 1,
        "fatigue": 1,
        "allergy": 1,
        "wheezing": 1,
        "alcohol_consuming": 1,
        "coughing": 1,
        "shortness_of_breath": 1,
        "swallowing_difficulty": 1,
        "chest_pain": 1,
    }

    for key, value in fallback_values.items():
        if key not in normalized:
            normalized[key] = value

    if FEATURE_COLUMNS:
        ordered_features = []
        for column in FEATURE_COLUMNS:
            canonical_column = column.strip().lower().replace(" ", "_").replace("-", "_")
            ordered_features.append(normalized.get(canonical_column, fallback_values.get(canonical_column, "")))
        df = pd.DataFrame([ordered_features], columns=FEATURE_COLUMNS)
    else:
        df = pd.DataFrame([normalized])

    for column in df.columns:
        if column in encoders:
            encoder = encoders[column]
            values = df[column].astype(str)
            known = list(encoder.classes_)
            values = values.apply(lambda value: value if value in known else known[0])
            df[column] = encoder.transform(values)
        else:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except Exception:
                pass

    return df


def predict_lung_cancer(patient_data):
    """Predict lung cancer risk from a dictionary of symptom features."""

    df = _normalize_patient_data(patient_data)

    X = scaler.transform(df)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    cancer_probability = probability[1] * 100

    return {
        "Model": model_name,
        "Prediction": "High Risk of Lung Cancer" if prediction == 1 else "Low Risk of Lung Cancer",
        "Cancer Probability (%)": float(round(float(cancer_probability), 2)),
        "Survival Chance (%)": float(round(float(100 - cancer_probability), 2)),
        "Confidence (%)": float(round(float(max(probability) * 100), 2)),
    }

# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    sample_patient = {

        "GENDER":"M",

        "AGE":67,

        "SMOKING":"YES",

        "YELLOW_FINGERS":"YES",

        "ANXIETY":"NO",

        "PEER_PRESSURE":"YES",

        "CHRONIC DISEASE":"NO",

        "FATIGUE ":"YES",

        "ALLERGY ":"NO",

        "WHEEZING":"YES",

        "ALCOHOL CONSUMING":"YES",

        "COUGHING":"YES",

        "SHORTNESS OF BREATH":"YES",

        "SWALLOWING DIFFICULTY":"NO",

        "CHEST PAIN":"YES"

    }

    result = predict_lung_cancer(sample_patient)

    print("\n")
    print("=" * 50)
    print("Prediction Result")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 50)
