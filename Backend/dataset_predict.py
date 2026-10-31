from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "dataset_best_model.pkl")
metadata = joblib.load(MODELS_DIR / "dataset_best_model_metadata.pkl")
FEATURE_COLUMNS = metadata["feature_columns"]
POSITIVE_CLASS = metadata["positive_class"]
MODEL_NAME = metadata["name"]

FIELD_ALIASES = {
    "five_year_survival_probability": "5_year_survival_probability",
}


def predict_dataset_record(record: dict) -> dict:
    """Run the experimental Final_Prediction model on one dataset record."""
    normalized = {
        str(key).strip().lower().replace(" ", "_").replace("-", "_"): value
        for key, value in record.items()
    }
    feature_values = {}
    for column in FEATURE_COLUMNS:
        normalized_column = column.strip().lower().replace(" ", "_").replace("-", "_")
        normalized_column = FIELD_ALIASES.get(normalized_column, normalized_column)
        feature_values[column] = normalized.get(normalized_column)
    features = pd.DataFrame(
        [[feature_values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    positive_probability = float(
        probabilities[classes.index(POSITIVE_CLASS)]
    )

    return {
        "Model": MODEL_NAME,
        "Target": metadata["target_column"],
        "Prediction": metadata["class_names"][prediction],
        "Class": prediction,
        "Cancer Probability (%)": round(positive_probability * 100, 2),
        "Confidence (%)": round(float(max(probabilities)) * 100, 2),
        "Experimental Warning": (
            "This model predicts the dataset's Final_Prediction label only. "
            "It is not a clinically validated cancer diagnostic."
        ),
    }
