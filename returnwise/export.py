"""Versioned local model export with a strict inference schema."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import joblib
import pandas as pd

from returnwise import __version__
from returnwise.paths import ARTIFACT_DIR
from returnwise.storage import MODEL_FILE


def export_model(
    model,
    feature_columns: list[str],
    selected_model: str,
    dataset_fingerprint: str | None = None,
    calibrators: dict | None = None,
    selected_calibrator: str | None = None,
    permutation_importance: list[dict] | None = None,
    prediction_scenario: str = "pre_shipment",
    validation_metrics: dict | None = None,
    decision_configuration: dict | None = None,
) -> dict:
    required_metrics = ("roc_auc", "pr_auc", "brier_score", "log_loss")
    if not feature_columns or not selected_model:
        raise ValueError("Model export requires a selected model and a non-empty feature schema.")
    if not validation_metrics or any(
        key not in validation_metrics or not math.isfinite(float(validation_metrics[key]))
        for key in required_metrics
    ):
        raise ValueError("Model export requires finite one-time holdout ROC-AUC, PR-AUC, Brier score, and log loss.")
    if int(validation_metrics.get("holdout_rows", 0)) < 1:
        raise ValueError("Model export requires a non-empty labeled holdout evaluation.")
    prevalence = float(validation_metrics.get("holdout_prevalence", float("nan")))
    baseline_brier = float(validation_metrics.get("constant_prevalence_brier", float("nan")))
    baseline_log_loss = float(validation_metrics.get("constant_prevalence_log_loss", float("nan")))
    if not all(math.isfinite(value) for value in (prevalence, baseline_brier, baseline_log_loss)):
        raise ValueError("Model export requires constant-prevalence holdout baselines for model validation.")
    if float(validation_metrics["roc_auc"]) <= 0.5 or float(validation_metrics["pr_auc"]) <= prevalence:
        raise ValueError("Model export stopped: holdout discrimination does not exceed the random/prevalence baseline.")
    if float(validation_metrics["brier_score"]) >= baseline_brier or float(validation_metrics["log_loss"]) >= baseline_log_loss:
        raise ValueError("Model export stopped: holdout probability quality does not improve on the constant-prevalence baseline.")
    if selected_calibrator and selected_calibrator != "uncalibrated" and selected_calibrator not in (calibrators or {}):
        raise ValueError("The selected calibrator is missing from the export payload.")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "application_version": __version__,
        "selected_model": selected_model,
        "feature_columns": list(feature_columns),
        "prediction_scenario": prediction_scenario,
        "selected_calibrator": selected_calibrator,
        "dataset_fingerprint": dataset_fingerprint,
        "validation_metrics": validation_metrics or {},
        "decision_configuration": decision_configuration or {},
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probability_is_calibrated": bool(selected_calibrator and selected_calibrator != "uncalibrated" and calibrators and selected_calibrator in calibrators),
    }
    payload = {
        "model": model,
        "calibrators": calibrators or {},
        "feature_columns": list(feature_columns),
        "permutation_importance": permutation_importance or [],
        "metadata": metadata,
    }
    joblib.dump(payload, MODEL_FILE, compress=3)
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def predict_with_artifact(payload: dict, frame):
    features = frame.reindex(columns=payload["feature_columns"])
    model = payload["model"]
    probabilities = model.predict_proba(features)[:, 1]
    calibrators = payload.get("calibrators", {})
    selected = payload.get("metadata", {}).get("selected_calibrator")
    if selected and selected in calibrators and selected != "uncalibrated":
        probabilities_calibrated = calibrators[selected].predict(probabilities)
    else:
        probabilities_calibrated = probabilities
    return probabilities, probabilities_calibrated


def make_competition_submission(model_payload: dict, competition_test: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Create the expected submission shape using probabilities and no test labels."""
    target = "returned"
    if target not in sample_submission.columns:
        raise ValueError("sample_submission.csv does not contain the documented `returned` prediction column.")
    if len(competition_test) != len(sample_submission):
        raise ValueError("Competition test and sample submission row counts differ.")
    shared_ids = [c for c in sample_submission.columns if c != target and c in competition_test.columns]
    if shared_ids and not sample_submission[shared_ids].reset_index(drop=True).equals(
        competition_test[shared_ids].reset_index(drop=True)
    ):
        raise ValueError("Competition test identifiers do not match sample_submission.csv order.")
    raw, _ = predict_with_artifact(model_payload, competition_test)
    submission = sample_submission.copy()
    submission[target] = raw
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Generated submission schema differs from sample_submission.csv.")
    return submission
