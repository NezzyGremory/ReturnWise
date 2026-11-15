"""Capacity-constrained review queue and transparent cost scenario simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


CAPACITIES = (0.05, 0.10, 0.20, 0.30)


def priority_queue(
    frame: pd.DataFrame,
    probabilities,
    calibrated_probabilities=None,
    uncertainty_threshold: float = 0.60,
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["return_probability"] = np.asarray(probabilities, dtype=float)
    result["calibrated_probability"] = (
        np.asarray(calibrated_probabilities, dtype=float) if calibrated_probabilities is not None else result["return_probability"]
    )
    result["probability_margin"] = np.abs(result["calibrated_probability"] - 0.5) * 2
    result["uncertain_review"] = np.maximum(result["calibrated_probability"], 1 - result["calibrated_probability"]) < uncertainty_threshold
    result["risk_band"] = pd.cut(
        result["calibrated_probability"],
        bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["low", "moderate", "high", "very high"],
    ).astype(str)
    result["priority_rank"] = result["calibrated_probability"].rank(method="first", ascending=False).astype(int)
    return result.sort_values(["priority_rank"], kind="stable").reset_index(drop=True)


def _select_indices(p, k, strategy, costs, seed, threshold, uncertain_mask):
    n = len(p)
    k = min(max(int(k), 0), n)
    if strategy == "fixed_threshold":
        eligible = np.flatnonzero(p >= threshold)
        if len(eligible) > k:
            eligible = eligible[np.argsort(-p[eligible], kind="stable")[:k]]
        return eligible
    if strategy == "uncertainty_aware":
        scores = np.asarray(uncertain_mask, dtype=int) * 2 + (1 - np.abs(p - 0.5) * 2)
    elif strategy in {"cost_sensitive", "calibration_aware_cost_sensitive"}:
        fp, fn, review = costs
        # Expected cost reduction from review versus leaving the case untouched.
        scores = p * fn - (review + (1 - p) * fp)
    elif strategy == "random":
        return np.random.default_rng(seed).permutation(n)[:k]
    else:
        scores = p
    return np.argsort(-scores, kind="stable")[:k]


def _cost_metrics(y_true, p, selected, costs):
    fp_cost, fn_cost, review_cost = costs
    n = len(p)
    chosen = np.zeros(n, dtype=bool)
    chosen[selected] = True
    if y_true is None:
        expected_reviewed_returns = float(p[chosen].sum())
        expected_false_positives = float((1 - p[chosen]).sum())
        expected_false_negatives = float(p[~chosen].sum())
        expected_cost = review_cost * int(chosen.sum()) + fp_cost * expected_false_positives + fn_cost * expected_false_negatives
        precision = expected_reviewed_returns / max(int(chosen.sum()), 1)
        recall = expected_reviewed_returns / max(float(p.sum()), 1e-12)
        return {
            "expected_decision_cost": expected_cost,
            "false_positives": expected_false_positives,
            "false_negatives": expected_false_negatives,
            "return_cases_identified": expected_reviewed_returns,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "count_type": "expected under predicted probabilities",
        }
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if len(y) != n:
        raise ValueError("y_true and probabilities must have the same length.")
    fp = int(np.sum(chosen & (y == 0)))
    fn = int(np.sum(~chosen & (y == 1)))
    tp = int(np.sum(chosen & (y == 1)))
    return {
        "expected_decision_cost": review_cost * int(chosen.sum()) + fp_cost * fp + fn_cost * fn,
        "false_positives": fp,
        "false_negatives": fn,
        "return_cases_identified": tp,
        "precision_at_k": tp / max(int(chosen.sum()), 1),
        "recall_at_k": tp / max(int((y == 1).sum()), 1),
        "count_type": "observed on labeled validation data",
    }


def evaluate_review_strategies(
    probabilities,
    y_true=None,
    calibrated_probabilities=None,
    costs: dict[str, float] | None = None,
    capacities: tuple[float, ...] = CAPACITIES,
    threshold: float = 0.5,
    uncertainty_threshold: float = 0.60,
    seed: int = 42,
) -> pd.DataFrame:
    p_raw = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0, 1)
    p_cal = np.clip(np.asarray(calibrated_probabilities if calibrated_probabilities is not None else p_raw, dtype=float).reshape(-1), 0, 1)
    costs = costs or {"false_positive": 1.0, "false_negative": 5.0, "review": 0.25}
    cost_tuple = (float(costs["false_positive"]), float(costs["false_negative"]), float(costs["review"]))
    if y_true is not None and len(y_true) != len(p_raw):
        raise ValueError("y_true must align with probabilities.")
    uncertain = np.maximum(p_cal, 1 - p_cal) < uncertainty_threshold
    strategies = (
        ("random", p_raw),
        ("probability_only", p_raw),
        ("fixed_threshold", p_raw),
        ("cost_sensitive", p_raw),
        ("calibration_aware_cost_sensitive", p_cal),
        ("uncertainty_aware", p_cal),
    )
    rows = []
    for capacity in capacities:
        k = max(1, int(np.ceil(len(p_raw) * capacity))) if len(p_raw) else 0
        for strategy, probs in strategies:
            selected = _select_indices(probs, k, strategy, cost_tuple, seed, threshold, uncertain)
            metrics = _cost_metrics(y_true, probs, selected, cost_tuple)
            row = {
                "strategy": strategy,
                "capacity_percent": capacity * 100,
                "capacity_rows": k,
                "review_count": len(selected),
                "false_positive_cost_assumption": cost_tuple[0],
                "false_negative_cost_assumption": cost_tuple[1],
                "review_cost_assumption": cost_tuple[2],
                **metrics,
            }
            if y_true is not None:
                y = np.asarray(y_true, dtype=int)
                row.update(
                    {
                        "roc_auc": roc_auc_score(y, probs) if len(np.unique(y)) == 2 else np.nan,
                        "pr_auc": average_precision_score(y, probs) if len(np.unique(y)) == 2 else np.nan,
                        "brier_score": brier_score_loss(y, probs),
                        "log_loss": log_loss(y, probs, labels=[0, 1]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)
