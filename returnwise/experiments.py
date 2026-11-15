"""Experiment orchestration helpers; all scores originate from supplied data."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from returnwise.calibration import ProbabilityCalibrator, expected_calibration_error
from returnwise.models import _probabilities, clone_pipeline, model_registry, resolve_candidate
from returnwise.paths import EXPERIMENT_DIR


def run_calibration_cv(
    X: pd.DataFrame, y, model_name: str, mode: str = "quick", seed: int = 42, parameters: dict | None = None
):  # noqa: N803
    """Nested split per fold: base fit, calibration fit, outer validation evaluation."""
    y = np.asarray(y, dtype=int)
    candidate = resolve_candidate(X, model_name, mode, seed, parameters)
    folds = 3 if mode.lower() == "quick" else 5
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (development, validation) in enumerate(outer.split(X, y), 1):
        base_idx, calibration_idx = train_test_split(
            development,
            test_size=0.25,
            random_state=seed + fold,
            stratify=y[development],
        )
        model = clone_pipeline(candidate)
        model.fit(X.iloc[base_idx], y[base_idx])
        raw_cal = _probabilities(model, X.iloc[calibration_idx])
        raw_valid = _probabilities(model, X.iloc[validation])
        methods = {"uncalibrated": None}
        for method in ("sigmoid", "isotonic", "beta"):
            try:
                calibrator = ProbabilityCalibrator(method).fit(raw_cal, y[calibration_idx])
                methods[method] = calibrator
            except (ValueError, ArithmeticError) as exc:
                methods[method] = exc
        for method, calibrator in methods.items():
            if isinstance(calibrator, Exception):
                rows.append({"fold": fold, "method": method, "status": f"failed: {calibrator}"})
                continue
            p = raw_valid if calibrator is None else calibrator.predict(raw_valid)
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "status": "completed",
                    "roc_auc": roc_auc_score(y[validation], p),
                    "pr_auc": average_precision_score(y[validation], p),
                    "brier_score": brier_score_loss(y[validation], p),
                    "log_loss": log_loss(y[validation], p, labels=[0, 1]),
                    "expected_calibration_error": expected_calibration_error(y[validation], p),
                }
            )
    return pd.DataFrame(rows)


def make_holdout_predictions(
    X: pd.DataFrame,
    y,
    model_name: str,
    mode: str = "quick",
    seed: int = 42,
    parameters: dict | None = None,
):  # noqa: N803
    """Create development, calibration, and one untouched final holdout split."""
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    development, holdout = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=seed, stratify=y
    )
    base_idx, calibration_idx = train_test_split(
        development, test_size=0.25, random_state=seed + 1, stratify=y.iloc[development]
    )
    model = resolve_candidate(X, model_name, mode, seed, parameters)
    model.fit(X.iloc[base_idx], y.iloc[base_idx])
    calibration_raw = _probabilities(model, X.iloc[calibration_idx])
    holdout_raw = _probabilities(model, X.iloc[holdout])
    calibrators = {"uncalibrated": None}
    for method in ("sigmoid", "isotonic", "beta"):
        try:
            calibrators[method] = ProbabilityCalibrator(method).fit(calibration_raw, y.iloc[calibration_idx])
        except ValueError:
            continue
    prediction_frame = pd.DataFrame(
        {
            "row_index": holdout,
            "y_true": y.iloc[holdout].to_numpy(),
            "probability_raw": holdout_raw,
        }
    )
    for method, calibrator in calibrators.items():
        prediction_frame[f"probability_{method}"] = holdout_raw if calibrator is None else calibrator.predict(holdout_raw)
    metrics = []
    for method in calibrators:
        p = prediction_frame[f"probability_{method}"].to_numpy()
        metrics.append(
            {
                "method": method,
                "roc_auc": roc_auc_score(prediction_frame["y_true"], p),
                "pr_auc": average_precision_score(prediction_frame["y_true"], p),
                "brier_score": brier_score_loss(prediction_frame["y_true"], p),
                "log_loss": log_loss(prediction_frame["y_true"], p, labels=[0, 1]),
                "expected_calibration_error": expected_calibration_error(prediction_frame["y_true"], p),
            }
        )
    return model, calibrators, prediction_frame, pd.DataFrame(metrics)


def ablation_table(predictions: pd.DataFrame, costs: dict[str, float] | None = None):
    """Evaluation variants A-E on a precomputed untouched labeled holdout."""
    from returnwise.decision import evaluate_review_strategies

    y = predictions["y_true"].to_numpy()
    p = predictions["probability_uncalibrated"].to_numpy()
    selected_calibrator = predictions.attrs.get("selected_calibrator")
    preferred = f"probability_{selected_calibrator}" if selected_calibrator else None
    calibrated_col = preferred if preferred in predictions.columns else next(
        (c for c in predictions if c.startswith("probability_") and c not in {"probability_uncalibrated", "probability_raw"}), None
    )
    p_cal = predictions[calibrated_col].to_numpy() if calibrated_col else p
    outputs = []
    specs = [
        ("A: probability-ranking baseline", p, p, None, None),
        ("B: calibration", p_cal, p_cal, None, None),
        ("C: cost-sensitive", p, p, "cost_sensitive", None),
        ("D: calibration + cost-sensitive", p_cal, p_cal, "calibration_aware_cost_sensitive", None),
        ("E: calibration + cost + uncertainty review", p_cal, p_cal, "uncertainty_aware", 0.60),
    ]
    for label, raw, calibrated, strategy, threshold in specs:
        started = time.perf_counter()
        selected_strategy = strategy or "probability_only"
        evals = evaluate_review_strategies(
            raw,
            y_true=y,
            calibrated_probabilities=calibrated,
            costs=costs,
            capacities=(0.05, 0.10, 0.20, 0.30),
            uncertainty_threshold=threshold or 0.60,
        )
        evals = evals[evals["strategy"] == selected_strategy].copy()
        evals.insert(0, "experiment", label)
        evals["runtime_seconds"] = time.perf_counter() - started
        outputs.extend(evals.to_dict(orient="records"))
    return pd.DataFrame(outputs)


def log_experiment(name: str, dataset_fingerprint: str | None, metrics: dict, details: dict | None = None) -> dict:
    if dataset_fingerprint is None:
        from returnwise.storage import active_dataset_fingerprint

        dataset_fingerprint = active_dataset_fingerprint()
    record = {
        "run_id": str(uuid.uuid4()),
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_fingerprint": dataset_fingerprint,
        "metrics": metrics,
        "details": details or {},
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT_DIR / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return record


def read_experiment_history() -> pd.DataFrame:
    path = EXPERIMENT_DIR / "history.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.json_normalize(records) if records else pd.DataFrame()
