import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_PATH = PROJECT_ROOT / "ml" / "url_feature_schema.json"

SPLIT_DIR = PROJECT_ROOT / "ml" / "datasets" / "splits"

TRAIN_PATH = SPLIT_DIR / "v2_train.csv"
VALIDATION_PATH = SPLIT_DIR / "v2_validation.csv"
TEST_PATH = SPLIT_DIR / "v2_test.csv"

MODEL_DIR = PROJECT_ROOT / "ml" / "models"
REPORT_DIR = PROJECT_ROOT / "ml" / "reports"

MODEL_PATH = MODEL_DIR / "url_model_v2.pkl"
METADATA_PATH = MODEL_DIR / "url_model_v2_metadata.json"

VALIDATION_REPORT_PATH = REPORT_DIR / "v2_validation_results.csv"
TEST_REPORT_PATH = REPORT_DIR / "v2_final_test_results.json"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "v2_feature_importance.csv"

RANDOM_STATE = 42
CLASS_NAMES = ["Benign", "Phishing"]


# ============================================================
# 2. LOAD THE CANONICAL FEATURE SCHEMA
# ============================================================

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)

FEATURE_COLUMNS = schema["feature_order"]

EXPECTED_FEATURE_COUNT = schema["feature_count"]

LABEL_COLUMN = schema["label"]["name"]

IDENTIFIER_COLUMNS = ["url", "normalized_url"]

METADATA_COLUMNS = ["source_dataset"]

EXPECTED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + METADATA_COLUMNS
    + FEATURE_COLUMNS
    + [LABEL_COLUMN]
)

if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Schema mismatch: expected {EXPECTED_FEATURE_COUNT} "
        f"features, found {len(FEATURE_COLUMNS)}."
    )

if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise ValueError("Duplicate feature names found in schema.")

print("=" * 65)
print("CYBERSHIELD V2 URL PHISHING MODEL TRAINING")
print("=" * 65)

print(f"Schema version: {schema['schema_version']}")
print(f"Extractor version: {schema['extractor_version']}")
print(f"Feature count: {len(FEATURE_COLUMNS)}")


# ============================================================
# 3. LOAD AND VALIDATE THE DATASET SPLITS
# ============================================================

def load_and_validate_split(file_path, split_name):

    print(f"\nLoading {split_name} dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"{split_name} file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"{split_name} dataset is empty.")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"{split_name} columns do not match the canonical schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Received: {list(df.columns)}"
        )

    if df.isnull().any().any():
        raise ValueError(
            f"Missing values found in {split_name} dataset."
        )

    if df["normalized_url"].duplicated().any():
        raise ValueError(
            f"Duplicate normalized URLs found in {split_name}."
        )

    labels = set(df[LABEL_COLUMN].unique())

    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(
            f"{split_name} must contain both labels 0 and 1 only."
        )

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    feature_values = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            f"Non-finite feature values found in {split_name}."
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="raise"
    ).astype(int)

    print(f"Records: {len(df):,}")
    print(f"Benign: {(df[LABEL_COLUMN] == 0).sum():,}")
    print(f"Phishing: {(df[LABEL_COLUMN] == 1).sum():,}")
    print("Validation: PASSED")

    return df


train_df = load_and_validate_split(TRAIN_PATH, "Training")
validation_df = load_and_validate_split(
    VALIDATION_PATH, "Validation"
)
test_df = load_and_validate_split(TEST_PATH, "Test")


# ============================================================
# 4. VERIFY SPLIT INTEGRITY
# ============================================================

train_urls = set(train_df["normalized_url"])
validation_urls = set(validation_df["normalized_url"])
test_urls = set(test_df["normalized_url"])

if train_urls & validation_urls:
    raise ValueError("URL leakage between training and validation.")

if train_urls & test_urls:
    raise ValueError("URL leakage between training and test.")

if validation_urls & test_urls:
    raise ValueError("URL leakage between validation and test.")

all_urls = train_urls | validation_urls | test_urls

if len(all_urls) != (
    len(train_df) + len(validation_df) + len(test_df)
):
    raise ValueError("Unexpected duplicate URLs across splits.")

print("\nSplit integrity: PASSED")
print("Cross-split normalized URL overlaps: 0")


# ============================================================
# 5. PREPARE FEATURE MATRICES AND LABELS
# ============================================================

# URL columns are identifiers only.
# They must never be used as model input features.

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[LABEL_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[LABEL_COLUMN]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[LABEL_COLUMN]

print("\nFeature matrix shapes:")
print(f"Training:   {X_train.shape}")
print(f"Validation: {X_validation.shape}")
print(f"Test:       {X_test.shape}")


# ============================================================
# 6. DEFINE CANDIDATE MODELS
# ============================================================

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 7. METRIC CALCULATION
# ============================================================

def calculate_metrics(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "classification_report": classification_report(
            y,
            predictions,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


# ============================================================
# 8. TRAIN AND COMPARE MODELS ON VALIDATION SET
# ============================================================

print("\n" + "=" * 65)
print("MODEL TRAINING AND VALIDATION")
print("=" * 65)

validation_results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()

    metrics = calculate_metrics(
        model,
        X_validation,
        y_validation
    )

    prediction_seconds = time.perf_counter() - start_time

    result = {
        "model": model_name,
        "training_seconds": round(training_seconds, 3),
        "validation_prediction_seconds": round(
            prediction_seconds, 3
        ),
        **{
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
            and key != "confusion_matrix"
        },
    }

    validation_results.append(result)
    trained_models[model_name] = model

    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Validation accuracy: {metrics['accuracy']:.6f}")
    print(f"Validation precision: {metrics['precision']:.6f}")
    print(f"Validation recall: {metrics['recall']:.6f}")
    print(f"Validation F1: {metrics['f1_score']:.6f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.6f}")
    print(
        "Validation false positive rate: "
        f"{metrics['false_positive_rate']:.6f}"
    )


# ============================================================
# 9. SELECT THE BEST MODEL
# ============================================================

# Selection priority:
# 1. Highest validation F1 score
# 2. Highest validation phishing recall
# 3. Highest validation precision

validation_results.sort(
    key=lambda result: (
        result["f1_score"],
        result["recall"],
        result["precision"],
    ),
    reverse=True,
)

best_model_name = validation_results[0]["model"]

print("\n" + "=" * 65)
print("VALIDATION COMPARISON")
print("=" * 65)

validation_results_df = pd.DataFrame(validation_results)

print(
    validation_results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "false_positive_rate",
        ]
    ].to_string(index=False)
)

print(f"\nSelected model: {best_model_name}")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

validation_results_df.to_csv(
    VALIDATION_REPORT_PATH,
    index=False
)


# ============================================================
# 10. RETRAIN SELECTED MODEL ON TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL MODEL TRAINING")
print("=" * 65)

X_train_validation = pd.concat(
    [X_train, X_validation],
    ignore_index=True
)

y_train_validation = pd.concat(
    [y_train, y_validation],
    ignore_index=True
)

final_model = models[best_model_name]

start_time = time.perf_counter()

final_model.fit(
    X_train_validation,
    y_train_validation
)

final_training_seconds = time.perf_counter() - start_time

print(f"Final training records: {len(X_train_validation):,}")
print(
    "Final training time: "
    f"{final_training_seconds:.2f} seconds"
)


# ============================================================
# 11. FINAL TEST EVALUATION — RUN ONCE
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

test_metrics = calculate_metrics(
    final_model,
    X_test,
    y_test
)

print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
print(f"Test precision: {test_metrics['precision']:.6f}")
print(f"Test recall: {test_metrics['recall']:.6f}")
print(f"Test F1: {test_metrics['f1_score']:.6f}")
print(f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
print(f"Test specificity: {test_metrics['specificity']:.6f}")
print(
    "Test false positive rate: "
    f"{test_metrics['false_positive_rate']:.6f}"
)

print("\nConfusion matrix:")
print(np.array(test_metrics["confusion_matrix"]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        final_model.predict(X_test),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


# ============================================================
# 12. SAVE THE FINAL MODEL
# ============================================================

joblib.dump(final_model, MODEL_PATH)

print(f"\nModel saved: {MODEL_PATH}")


# ============================================================
# 13. SAVE FEATURE IMPORTANCE
# ============================================================

if hasattr(final_model, "feature_importances_"):

    feature_importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": final_model.feature_importances_,
    })

    feature_importance_df = feature_importance_df.sort_values(
        "importance",
        ascending=False
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    print(f"Feature importance saved: {FEATURE_IMPORTANCE_PATH}")


# ============================================================
# 14. SAVE FINAL TEST REPORT AND MODEL METADATA
# ============================================================

test_report = {
    "project": schema["project"],
    "schema_version": schema["schema_version"],
    "extractor_version": schema["extractor_version"],
    "selected_model": best_model_name,
    "selection_metric": "validation_f1",
    "decision_threshold": 0.5,
    "feature_count": len(FEATURE_COLUMNS),
    "feature_order": FEATURE_COLUMNS,
    "training_records": int(len(X_train)),
    "validation_records": int(len(X_validation)),
    "final_training_records": int(len(X_train_validation)),
    "test_records": int(len(X_test)),
    "final_training_seconds": round(
        final_training_seconds, 3
    ),
    "test_metrics": test_metrics,
    "validation_comparison": validation_results,
}

with open(TEST_REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        test_report,
        file,
        indent=4
    )

metadata = {
    "project": schema["project"],
    "model_version": "2.0",
    "schema_version": schema["schema_version"],
    "extractor_version": schema["extractor_version"],
    "selected_model": best_model_name,
    "feature_count": len(FEATURE_COLUMNS),
    "feature_order": FEATURE_COLUMNS,
    "label_encoding": schema["label_encoding"],
    "decision_threshold": 0.5,
    "prediction_input_requirements": {
        "feature_count": 33,
        "feature_order_must_match": True,
        "url_and_normalized_url_are_not_model_features": True,
        "same_canonical_extractor_required": True,
        "same_public_suffix_list_required": True,
    },
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 15. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CYBERSHIELD V2 TRAINING COMPLETED")
print("=" * 65)

print(f"Selected model: {best_model_name}")
print(f"Test F1 score: {test_metrics['f1_score']:.6f}")
print(f"Test ROC-AUC: {test_metrics['roc_auc']:.6f}")

print("\nGenerated files:")
print(f"1. {MODEL_PATH}")
print(f"2. {METADATA_PATH}")
print(f"3. {VALIDATION_REPORT_PATH}")
print(f"4. {TEST_REPORT_PATH}")
print(f"5. {FEATURE_IMPORTANCE_PATH}")

print("\nTraining pipeline completed successfully.")
