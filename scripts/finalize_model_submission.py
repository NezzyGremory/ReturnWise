"""Refit the recorded raw LightGBM candidate and create a Kaggle submission.

This data-only finalization script never touches the previous model artifact.
It refuses to start or advance to the next major phase when available RAM is
below the configured safety buffer.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MIN_AVAILABLE_GIB = 0.55  # runner floor 0.45 GiB plus a 0.10 GiB reserve
FEATURE_COLUMNS = [
    "customer_age",
    "product_price",
    "discount_percent",
    "product_rating",
    "session_length_minutes",
    "num_product_views",
    "device_type",
    "product_category",
    "shipping_method",
    "payment_method",
    "used_coupon",
]
TARGET = "returned"
CV_EVIDENCE = {
    "metric": "roc_auc",
    "mean": 0.5865764956658149,
    "fold_standard_deviation": 0.0032309501549188766,
    "folds": 3,
    "seed": 42,
    "development_rows": 160000,
    "feature_set": "raw",
    "tuning": "untuned Quick Mode candidate; not the two-trial Optuna pilot",
}
OUTPUT_NAMES = [
    "final_model.joblib",
    "final_metadata.json",
    "final_feature_list.json",
    "final_preprocessing_config.json",
    "final_test_predictions.csv",
    "final_submission.csv",
]


def available_gib() -> float:
    return psutil.virtual_memory().available / (1024**3)


def require_memory(stage: str) -> float:
    available = available_gib()
    print(f"RAM before {stage}: {available:.3f} GiB available")
    if available < MIN_AVAILABLE_GIB:
        raise MemoryError(
            f"Stopping before {stage}: {available:.3f} GiB available is below "
            f"the {MIN_AVAILABLE_GIB:.2f} GiB safety threshold."
        )
    return available


def atomic_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def library_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> None:
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    destinations = [artifacts / name for name in OUTPUT_NAMES]
    existing = [path.name for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing final outputs: " + ", ".join(existing)
        )

    require_memory("scientific-library imports")
    import joblib
    import numpy as np
    import pandas as pd
    from returnwise.models import resolve_candidate

    require_memory("training data load")
    data_dir = PROJECT_ROOT / "data" / "kaggle"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    if TARGET not in train.columns:
        raise ValueError(f"Training CSV is missing target `{TARGET}`.")
    missing_features = [column for column in FEATURE_COLUMNS if column not in train.columns]
    if missing_features:
        raise ValueError(f"Training CSV is missing expected features: {missing_features}")
    if train[TARGET].isna().any() or not set(train[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Training target must be complete and binary 0/1.")
    X_train = train.loc[:, FEATURE_COLUMNS]
    y_train = train[TARGET].astype("int8")
    if len(X_train) != 200000:
        raise ValueError(f"Expected 200,000 labeled rows; found {len(X_train)}.")
    if TARGET in X_train.columns or "order_id" in X_train.columns:
        raise ValueError("Target/identifier leakage detected in the training feature frame.")
    pre_fit_memory = require_memory("LightGBM fit")

    # resolve_candidate uses the exact Quick Mode registry defaults measured by
    # the raw-feature CV comparison (not the separate two-trial Optuna study).
    model = resolve_candidate(X_train, "LightGBM", mode="quick", seed=42)
    estimator = model.named_steps["model"]
    expected_parameters = {
        "n_estimators": 48,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "n_jobs": 1,
        "random_state": 42,
    }
    actual_parameters = estimator.get_params(deep=False)
    for name, expected in expected_parameters.items():
        if actual_parameters.get(name) != expected:
            raise ValueError(
                f"Registry parameter mismatch for {name}: "
                f"expected {expected!r}, got {actual_parameters.get(name)!r}."
            )
    if "features" in model.named_steps or "feature_selection" in model.named_steps:
        raise ValueError("The selected CV configuration must use raw inputs without selection.")

    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    post_fit_memory = require_memory("test prediction")
    train_rows = len(X_train)
    target_prevalence = float(y_train.mean())
    del X_train, y_train, train
    gc.collect()

    sample = pd.read_csv(sample_path)
    if TARGET not in sample.columns:
        raise ValueError("Sample submission is missing the `returned` column.")
    id_columns = [column for column in sample.columns if column != TARGET]
    if len(id_columns) != 1:
        raise ValueError(f"Expected one submission identifier column; found {id_columns}.")
    id_column = id_columns[0]
    if len(sample) != 50000:
        raise ValueError(f"Expected 50,000 sample submission rows; found {len(sample)}.")
    expected_ids = sample[id_column].to_numpy(copy=True)
    if pd.Series(expected_ids).duplicated().any():
        raise ValueError("sample_submission identifier contains duplicates.")

    test_probabilities: list[np.ndarray] = []
    row_count = 0
    with pd.read_csv(test_path, chunksize=10000) as chunks:
        for chunk in chunks:
            if TARGET in chunk.columns:
                raise ValueError("Test CSV unexpectedly contains the target column.")
            missing = [column for column in FEATURE_COLUMNS if column not in chunk.columns]
            if missing:
                raise ValueError(f"Test CSV is missing expected features: {missing}")
            if "order_id" not in chunk.columns:
                raise ValueError("Test CSV is missing the expected order_id identifier.")
            end = row_count + len(chunk)
            if not np.array_equal(chunk["order_id"].to_numpy(), expected_ids[row_count:end]):
                raise ValueError("Test order_id values differ from sample_submission ID order.")
            batch = chunk.loc[:, FEATURE_COLUMNS]
            probabilities = model.predict_proba(batch)[:, 1]
            if len(probabilities) != len(chunk):
                raise ValueError("Prediction row count differs from input batch.")
            if not np.isfinite(probabilities).all():
                raise ValueError("Predictions contain NaN or infinite values.")
            if ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Predictions fall outside [0, 1].")
            test_probabilities.append(probabilities.astype(np.float64, copy=False))
            row_count = end
            del batch, chunk, probabilities
            if row_count % 20000 == 0:
                require_memory(f"prediction batch {row_count}")

    predictions = np.concatenate(test_probabilities)
    if row_count != 50000 or len(predictions) != 50000:
        raise ValueError(f"Expected 50,000 test predictions; found {len(predictions)}.")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Final probability vector failed finite/range validation.")

    prediction_frame = pd.DataFrame(
        {id_column: expected_ids, "probability_returned": predictions}
    )
    submission = sample.copy()
    submission[TARGET] = predictions
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not exactly match sample_submission.csv.")

    final_params = estimator.get_params(deep=False)
    preprocessing_config = {
        "input_feature_columns_in_order": FEATURE_COLUMNS,
        "feature_set": "raw expanded order-time candidate",
        "point_in_time_status": "not verified from supplied data",
        "feature_engineering": False,
        "feature_selection": None,
        "numeric_features": [
            "customer_age",
            "product_price",
            "discount_percent",
            "product_rating",
            "session_length_minutes",
            "num_product_views",
            "used_coupon",
        ],
        "categorical_features": [
            "device_type",
            "product_category",
            "shipping_method",
            "payment_method",
        ],
        "numeric_imputation": "median, fitted on training data",
        "categorical_imputation": "most_frequent, fitted on training data",
        "categorical_encoding": {
            "method": "OneHotEncoder",
            "handle_unknown": "ignore",
            "min_frequency": 2,
        },
        "numeric_scaling": False,
        "test_data_used_for_fit": False,
        "raw_values_modified": False,
        "probability_calibration": "none; raw LightGBM probabilities",
    }
    metadata = {
        "model_name": "LightGBM",
        "estimator": "LGBMClassifier",
        "parameters": final_params,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "training_rows": train_rows,
        "training_target_prevalence": target_prevalence,
        "training_data_fingerprint": "2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56",
        "cv_selection_evidence": CV_EVIDENCE,
        "cv_is_not_test_performance": True,
        "independent_holdout_evaluation_for_this_refit": False,
        "historical_holdout_reused_for_selection": False,
        "probability_calibration": "none; raw probabilities",
        "fit_seconds": fit_seconds,
        "ram_available_before_fit_gib": round(pre_fit_memory, 4),
        "ram_available_after_fit_gib": round(post_fit_memory, 4),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": library_version("numpy"),
            "pandas": library_version("pandas"),
            "scikit_learn": library_version("scikit-learn"),
            "lightgbm": library_version("lightgbm"),
            "joblib": library_version("joblib"),
        },
        "preprocessing_config_file": "final_preprocessing_config.json",
    }

    with tempfile.TemporaryDirectory(prefix="returnwise-final-", dir=artifacts) as temp_name:
        staging = Path(temp_name)
        staged = {name: staging / name for name in OUTPUT_NAMES}
        joblib.dump(model, staged["final_model.joblib"], compress=3)
        atomic_json(staged["final_metadata.json"], metadata)
        atomic_json(
            staged["final_feature_list.json"],
            {"target": TARGET, "feature_columns_in_order": FEATURE_COLUMNS},
        )
        atomic_json(staged["final_preprocessing_config.json"], preprocessing_config)
        prediction_frame.to_csv(staged["final_test_predictions.csv"], index=False)
        submission.to_csv(staged["final_submission.csv"], index=False)

        # Reload the actual serialized artifact and compare a deterministic sample.
        loaded_model = joblib.load(staged["final_model.joblib"])
        sample_batch = pd.read_csv(test_path, nrows=128).loc[:, FEATURE_COLUMNS]
        reloaded_probabilities = loaded_model.predict_proba(sample_batch)[:, 1]
        if not np.allclose(reloaded_probabilities, predictions[:128], rtol=0, atol=1e-12):
            raise ValueError("Reloaded model predictions differ from the original model.")

        reread_predictions = pd.read_csv(staged["final_test_predictions.csv"])
        reread_submission = pd.read_csv(staged["final_submission.csv"])
        if len(reread_predictions) != 50000 or len(reread_submission) != 50000:
            raise ValueError("Staged prediction/submission files do not have 50,000 rows.")
        if list(reread_submission.columns) != list(sample.columns):
            raise ValueError("Staged submission schema differs from sample_submission.csv.")
        if reread_submission[id_column].duplicated().any():
            raise ValueError("Staged submission contains duplicate IDs.")
        final_values = reread_submission[TARGET].to_numpy()
        if not np.isfinite(final_values).all() or ((final_values < 0) | (final_values > 1)).any():
            raise ValueError("Staged submission probabilities failed validation.")
        if not np.array_equal(reread_submission[id_column].to_numpy(), expected_ids):
            raise ValueError("Staged submission ID order differs from sample_submission.csv.")

        for name, destination in zip(OUTPUT_NAMES, destinations, strict=True):
            os.replace(staged[name], destination)

    print(json.dumps({
        "model": metadata["model_name"],
        "training_rows": train_rows,
        "cv_roc_auc_mean": CV_EVIDENCE["mean"],
        "cv_roc_auc_sd": CV_EVIDENCE["fold_standard_deviation"],
        "fit_seconds": round(fit_seconds, 3),
        "test_predictions": row_count,
        "submission_rows": len(submission),
        "submission_columns": list(submission.columns),
        "probability_min": float(predictions.min()),
        "probability_max": float(predictions.max()),
        "reload_sample_max_abs_difference": float(np.max(np.abs(reloaded_probabilities - predictions[:128]))),
        "artifact_paths": [str(path.relative_to(PROJECT_ROOT)) for path in destinations],
    }, indent=2))


if __name__ == "__main__":
    main()
