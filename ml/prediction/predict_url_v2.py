
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.url_feature_extractor import (
    normalize_url,
    extract_url_features,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "url_model_v2.pkl"
METADATA_PATH = PROJECT_ROOT / "ml" / "models" / "url_model_v2_metadata.json"
SCHEMA_PATH = PROJECT_ROOT / "ml" / "url_feature_schema.json"


# ============================================================
# 2. LOAD AND VALIDATE MODEL ARTIFACTS
# ============================================================

@lru_cache(maxsize=1)
def load_artifacts():

    for path in [MODEL_PATH, METADATA_PATH, SCHEMA_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    feature_order = schema["feature_order"]
    metadata_feature_order = metadata["feature_order"]

    # Verify schema compatibility.
    if feature_order != metadata_feature_order:
        raise ValueError("Model metadata and feature schema order do not match.")

    if len(feature_order) != 33:
        raise ValueError("Expected exactly 33 features in V2 schema.")

    if metadata["schema_version"] != schema["schema_version"]:
        raise ValueError("Schema version mismatch.")

    if metadata["extractor_version"] != schema["extractor_version"]:
        raise ValueError("Extractor version mismatch.")

    if metadata["feature_count"] != len(feature_order):
        raise ValueError("Metadata feature count mismatch.")

    # Verify model input compatibility.
    if model.n_features_in_ != len(feature_order):
        raise ValueError("Model expects a different number of features.")

    # If feature names were saved by scikit-learn, verify their order too.
    if hasattr(model, "feature_names_in_"):
        model_feature_order = list(model.feature_names_in_)

        if model_feature_order != feature_order:
            raise ValueError("Model feature order does not match V2 schema.")

    if set(model.classes_.tolist()) != {0, 1}:
        raise ValueError("Model must support classes 0 and 1.")

    threshold = float(metadata["decision_threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Invalid decision threshold.")

    return model, metadata, feature_order, threshold


# ============================================================
# 3. PREDICT URL
# ============================================================

def predict_url(url: str) -> dict:
    """
    Predict whether a URL is Benign or Phishing.

    Uses the canonical V2 URL normalizer and feature extractor.
    """

    # Load model and schema.
    model, metadata, feature_order, threshold = load_artifacts()

    # Normalize and validate the input URL.
    normalized_url = normalize_url(url)

    # Extract canonical V2 features.
    features = extract_url_features(normalized_url)

    if not isinstance(features, dict):
        raise ValueError("Feature extractor must return a dictionary.")

    # Verify exact feature names.
    missing_features = set(feature_order) - set(features.keys())
    extra_features = set(features.keys()) - set(feature_order)

    if missing_features or extra_features:
        raise ValueError(
            f"Feature mismatch. "
            f"Missing: {sorted(missing_features)}; "
            f"Extra: {sorted(extra_features)}"
        )

    # Arrange features in the exact training order.
    ordered_features = {}

    for feature_name in feature_order:

        value = features[feature_name]

        if not isinstance(value, (int, float, bool)):
            raise ValueError(
                f"Feature '{feature_name}' is not numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"Feature '{feature_name}' contains an invalid value."
            )

        ordered_features[feature_name] = value

    # Convert to DataFrame with exact feature names and order.
    input_data = pd.DataFrame(
        [ordered_features],
        columns=feature_order
    )

    # Get prediction probabilities.
    probabilities = model.predict_proba(input_data)[0]

    class_indices = {
        int(class_label): index
        for index, class_label in enumerate(model.classes_)
    }

    benign_probability = float(probabilities[class_indices[0]])
    phishing_probability = float(probabilities[class_indices[1]])

    # Apply the saved decision threshold.
    predicted_label = (
        1 if phishing_probability >= threshold else 0
    )

    label_encoding = metadata["label_encoding"]

    return {
        "original_url": url,
        "normalized_url": normalized_url,
        "prediction": label_encoding[str(predicted_label)],
        "label": predicted_label,
        "confidence": max(
            benign_probability,
            phishing_probability
        ),
        "probabilities": {
            "benign": benign_probability,
            "phishing": phishing_probability
        },
        "model_version": metadata["model_version"],
        "schema_version": metadata["schema_version"]
    }


# ============================================================
# 4. BASIC MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "https://login.account-example.com/verify?password=test"
    ]

    print("\nCyberShield URL Prediction V2")
    print("=" * 60)

    for test_url in test_urls:

        try:
            result = predict_url(test_url)

            print("\nURL:", result["original_url"])
            print("Normalized:", result["normalized_url"])
            print("Prediction:", result["prediction"])
            print("Label:", result["label"])

            print(
                "Benign probability:",
                f'{result["probabilities"]["benign"]:.4f}'
            )

            print(
                "Phishing probability:",
                f'{result["probabilities"]["phishing"]:.4f}'
            )

        except (ValueError, TypeError) as error:
            print("\nURL:", test_url)
            print("Prediction error:", error)

    print("\n" + "=" * 60)
