
from pathlib import Path

import joblib
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gaussian_nb.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "gaussian_nb_metadata.pkl"

model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)


def predict_lung_cancer(user_input):
    """
    Returns a prediction only when all input fields pass validation.
    Invalid input raises ValueError; no cancer prediction is returned.
    """

    expected_features = metadata["features"]

    # 1. Require exactly the expected fields
    if not isinstance(user_input, dict):
        raise ValueError("Input must be a dictionary.")

    # API clients use readable snake_case names while the training data uses
    # uppercase names and spaces (for example, shortness_of_breath vs
    # SHORTNESS OF BREATH). Normalize both forms at this boundary.
    normalized_keys = {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): key
        for key in user_input
    }
    expected_by_normalized_key = {
        re.sub(r"[^a-z0-9]", "", str(feature).lower()): feature
        for feature in expected_features
    }

    unknown_keys = set(normalized_keys) - set(expected_by_normalized_key)
    missing = set(expected_by_normalized_key) - set(normalized_keys)

    if missing:
        missing_names = sorted(expected_by_normalized_key[key] for key in missing)
        raise ValueError(f"Missing fields: {missing_names}")

    if unknown_keys:
        extra_names = sorted(normalized_keys[key] for key in unknown_keys)
        raise ValueError(f"Unexpected fields: {extra_names}")

    row = {
        expected_by_normalized_key[key]: user_input[original_key]
        for key, original_key in normalized_keys.items()
    }

    # The training data covers ages 21 through 87 only. Do not extrapolate
    # the model outside that observed age range.
    age_feature = next(
        (feature for feature in expected_features if str(feature).upper() == "AGE"),
        None,
    )
    if age_feature is not None:
        try:
            age = float(row[age_feature])
        except (TypeError, ValueError):
            age = None

        if age is not None and pd.notna(age) and (age <= 20 or age > 87):
            return {
                "available": False,
                "prediction": "NO DATA AVAILABLE",
                "cancer_probability": None,
                "message": (
                    "No training data is available for this age. "
                    "The model supports ages 21 through 87 only."
                ),
            }

    # 2. Normalize gender
    if "GENDER" in row:
        gender = str(row["GENDER"]).strip().upper()

        if gender not in metadata["gender_mapping"]:
            raise ValueError("GENDER must be F or M.")

        row["GENDER"] = metadata["gender_mapping"][gender]

    # 3. Validate numeric inputs and allowed values
    validated = {}

    for feature in expected_features:
        value = row[feature]

        if value is None or isinstance(value, bool):
            raise ValueError(f"{feature} must have a valid value.")

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be numeric.")

        if not pd.notna(number):
            raise ValueError(f"{feature} cannot be missing.")

        # Reject values outside the training range.
        minimum = float(metadata["min_values"][feature])
        maximum = float(metadata["max_values"][feature])

        if number < minimum or number > maximum:
            raise ValueError(
                f"{feature} must be between {minimum:g} and {maximum:g}."
            )

        # Symptom indicators must use values observed during training.
        if feature in metadata["allowed_values"]:
            allowed = metadata["allowed_values"][feature]

            if number not in allowed:
                raise ValueError(
                    f"Invalid {feature}: {value}. "
                    f"Allowed values: {allowed}"
                )

        validated[feature] = number

    # 4. Predict only after every validation succeeds
    X = pd.DataFrame([validated], columns=expected_features)

    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0]
    cancer_probability = float(probability[list(model.classes_).index(1)])
    confidence = float(probability.max())

    return {
        "prediction": "YES" if prediction == 1 else "NO",
        "model": "Gaussian Naive Bayes",
        # Keep the model calculation as a fraction internally, but expose
        # the API value as a percentage for the frontend.
        "cancer_probability": round(cancer_probability * 100, 2),
        "confidence": round(confidence * 100, 2),
        "message": (
            "Educational model output only; not a medical diagnosis."
        )
    }


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    sample = {
        "GENDER": "M",
        "AGE": 150,
        "SMOKING": 1,
        "YELLOW_FINGERS": 2,
        "ANXIETY": 2,
        "PEER_PRESSURE": 1,
        "CHRONIC DISEASE": 1,
        "FATIGUE": 2,
        "ALLERGY": 1,
        "WHEEZING": 2,
        "ALCOHOL CONSUMING": 2,
        "COUGHING": 2,
        "SHORTNESS OF BREATH": 2,
        "SWALLOWING DIFFICULTY": 2,
        "CHEST PAIN": 2
    }

    try:
        result = predict_lung_cancer(sample)
        print(result)
    except ValueError as error:
        print("INPUT REJECTED:", error)
