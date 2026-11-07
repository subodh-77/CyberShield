

import sys
import json
import math
import numbers
import pandas as pd

from pathlib import Path
from collections import Counter

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.url_feature_extractor import (
    normalize_url,
    extract_url_features
)

DATASET_DIR = PROJECT_ROOT / "ml" / "datasets" / "processed"

BENIGN_FILE = DATASET_DIR / "benign_urls.csv"
PHISHING_FILE = DATASET_DIR / "phishing_urls.csv"

OUTPUT_FILE = DATASET_DIR / "final_url_dataset_v2.csv"
REJECTED_FILE = DATASET_DIR / "v2_rejected_urls_during_build.csv"
CONFLICT_FILE = DATASET_DIR / "v2_conflicting_urls.csv"
DUPLICATE_FILE = DATASET_DIR / "v2_canonical_duplicates.csv"
FEATURE_ERROR_FILE = DATASET_DIR / "v2_feature_errors.csv"
SUMMARY_FILE = DATASET_DIR / "v2_build_summary.json"


# --------------------------------------------------
# CANONICAL 33-FEATURE ORDER
# --------------------------------------------------

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "domain_length",
    "path_length",
    "num_subdomains",
    "subdomain_ratio",
    "max_subdomain_length",
    "domain_entropy",
    "domain_digit_count",
    "domain_hyphen_count",
    "contains_unicode",
    "non_ascii_count",
    "mixed_script",
    "contains_punycode",
    "has_https",
    "has_ip_address",
    "has_at_symbol_in_authority",
    "has_double_slash_in_path",
    "has_encoded_characters",
    "percent_encoded_count",
    "has_nonstandard_port",
    "has_fragment",
    "query_param_count",
    "query_length",
    "path_segment_count",
    "max_path_segment_length",
    "suspicious_keyword_subdomain_count",
    "suspicious_keyword_domain_count",
    "suspicious_keyword_path_count",
    "suspicious_keyword_query_count",
    "tld_length",
    "has_suspicious_tld",
    "has_url_shortener"
]


# --------------------------------------------------
# LOAD INPUT DATA
# --------------------------------------------------

print("========== LOADING INPUT DATA ==========")

benign = pd.read_csv(BENIGN_FILE)
phishing = pd.read_csv(PHISHING_FILE)

required_columns = {"url", "label"}

for filename, df, expected_label in [
    ("benign_urls.csv", benign, 0),
    ("phishing_urls.csv", phishing, 1)
]:
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain columns: {required_columns}"
        )

    labels = set(df["label"].dropna().unique())

    if labels != {expected_label}:
        raise ValueError(
            f"Unexpected labels in {filename}: {labels}"
        )

print("Benign records:", len(benign))
print("Phishing records:", len(phishing))
print("Total input records:", len(benign) + len(phishing))


# --------------------------------------------------
# NORMALIZE URLS
# --------------------------------------------------

print("\n========== NORMALIZING URLS ==========")

records = []
rejected = []

for source_name, df, expected_label in [
    ("benign", benign, 0),
    ("phishing", phishing, 1)
]:
    for row_number, row in df.iterrows():

        original_url = row["url"]

        try:
            if not isinstance(original_url, str) or not original_url.strip():
                raise ValueError("Empty or invalid original URL")

            canonical_url = normalize_url(original_url)

            if not isinstance(canonical_url, str) or not canonical_url:
                raise ValueError("Normalizer returned an invalid result")

            records.append({
                "url": original_url,
                "normalized_url": canonical_url,
                "label": expected_label,
                "source_dataset": source_name
            })

        except Exception as exc:
            rejected.append({
                "source_dataset": source_name,
                "source_row": row_number,
                "url": original_url,
                "label": expected_label,
                "length": len(str(original_url)),
                "error": str(exc)
            })

rejected_df = pd.DataFrame(
    rejected,
    columns=[
        "source_dataset", "source_row", "url",
        "label", "length", "error"
    ]
)

rejected_df.to_csv(REJECTED_FILE, index=False)

normalized_df = pd.DataFrame(records)

print("Successfully normalized:", len(normalized_df))
print("Rejected URLs:", len(rejected_df))


# --------------------------------------------------
# FIND CROSS-CLASS EXACT URL CONFLICTS
# --------------------------------------------------

print("\n========== EXACT URL CONFLICT AUDIT ==========")

label_counts = normalized_df.groupby(
    "normalized_url"
)["label"].nunique()

conflicting_urls = set(label_counts[label_counts > 1].index)

conflicts_df = normalized_df[
    normalized_df["normalized_url"].isin(conflicting_urls)
].copy()

conflicts_df.to_csv(CONFLICT_FILE, index=False)

print("Conflicting canonical URLs:", len(conflicting_urls))
print("Records quarantined:", len(conflicts_df))

# Remove BOTH labels for every conflicting canonical URL.
clean_df = normalized_df[
    ~normalized_df["normalized_url"].isin(conflicting_urls)
].copy()


# --------------------------------------------------
# AUDIT SAME-CLASS CANONICAL DUPLICATES
# --------------------------------------------------

print("\n========== CANONICAL DUPLICATE AUDIT ==========")

duplicate_mask = clean_df.duplicated(
    subset=["normalized_url", "label"],
    keep="first"
)

duplicates_df = clean_df[duplicate_mask].copy()
duplicates_df.to_csv(DUPLICATE_FILE, index=False)

print("Additional same-class duplicate records:", len(duplicates_df))

# Do not silently remove these records in this version.
# They are retained for separate review.


# --------------------------------------------------
# EXTRACT AND VALIDATE FEATURES
# --------------------------------------------------

print("\n========== EXTRACTING 33 FEATURES ==========")

feature_records = []
feature_errors = []

for index, row in clean_df.iterrows():

    url = row["normalized_url"]

    try:
        features = extract_url_features(url)

        # Enforce exact feature names and order.
        if list(features.keys()) != FEATURE_NAMES:
            raise ValueError(
                "Feature names or order do not match canonical schema"
            )

        # Ensure all values are numeric and finite.
        for name, value in features.items():

            if not isinstance(value, numbers.Real):
                raise ValueError(
                    f"Non-numeric feature: {name} = {value}"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite feature: {name} = {value}"
                )

        feature_records.append({
            **row.to_dict(),
            **features
        })

    except Exception as exc:
        feature_errors.append({
            "source_index": index,
            "url": row["url"],
            "normalized_url": url,
            "label": row["label"],
            "error": str(exc)
        })

feature_errors_df = pd.DataFrame(
    feature_errors,
    columns=[
        "source_index", "url", "normalized_url",
        "label", "error"
    ]
)

feature_errors_df.to_csv(FEATURE_ERROR_FILE, index=False)

print("Successfully extracted:", len(feature_records))
print("Feature extraction failures:", len(feature_errors_df))


# --------------------------------------------------
# FAIL SAFELY IF EXTRACTION FAILED
# --------------------------------------------------

if feature_errors:
    print("\nBUILD STOPPED: Feature extraction errors found.")
    print("Review:", FEATURE_ERROR_FILE)
    print("No new V2 dataset was written.")
    raise SystemExit(1)


# --------------------------------------------------
# BUILD FINAL DATASET
# --------------------------------------------------

print("\n========== BUILDING V2 DATASET ==========")

final_df = pd.DataFrame(feature_records)

metadata_columns = [
    "url",
    "normalized_url",
    "source_dataset"
]

final_columns = metadata_columns + FEATURE_NAMES + ["label"]

final_df = final_df[final_columns]

# Final schema validation.
assert len(FEATURE_NAMES) == 33
assert final_df.columns.tolist() == final_columns
assert final_df[FEATURE_NAMES].notna().all().all()

final_df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# BUILD SUMMARY REPORT
# --------------------------------------------------

summary = {
    "input_records": len(benign) + len(phishing),
    "benign_input_records": len(benign),
    "phishing_input_records": len(phishing),
    "normalization_success": len(normalized_df),
    "normalization_rejected": len(rejected_df),
    "conflicting_canonical_urls": len(conflicting_urls),
    "conflicting_records_quarantined": len(conflicts_df),
    "same_class_canonical_duplicate_records": len(duplicates_df),
    "feature_extraction_failures": len(feature_errors_df),
    "final_records": len(final_df),
    "feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "label_distribution": {
        str(k): int(v)
        for k, v in final_df["label"].value_counts().sort_index().items()
    },
    "output_file": str(OUTPUT_FILE)
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n========== V2 BUILD SUMMARY ==========")
print(json.dumps(summary, indent=4))

print("\n========== OUTPUT FILES ==========")
print("V2 dataset:", OUTPUT_FILE)
print("Rejected URLs:", REJECTED_FILE)
print("Conflicting URLs:", CONFLICT_FILE)
print("Canonical duplicates:", DUPLICATE_FILE)
print("Feature errors:", FEATURE_ERROR_FILE)
print("Build summary:", SUMMARY_FILE)

print("\nV2 DATASET BUILD COMPLETED SUCCESSFULLY.")
