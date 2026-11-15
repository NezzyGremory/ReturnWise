"""Model-agnostic explanation helpers with safe optional SHAP support."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def _positive_class_roc_auc(estimator, X, y) -> float:
    """Score a binary estimator using only its positive-class probability."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = getattr(estimator, "classes_", None)
        if isinstance(probabilities, list):
            probabilities = probabilities[-1]
        if getattr(probabilities, "ndim", 1) == 2:
            positive_index = list(classes).index(1) if classes is not None and 1 in classes else 1
            probabilities = probabilities[:, positive_index]
    elif hasattr(estimator, "decision_function"):
        probabilities = estimator.decision_function(X)
    else:
        raise TypeError("Permutation ROC-AUC requires predict_proba or decision_function.")
    return float(roc_auc_score(y, probabilities))


def permutation_importance_table(model, X: pd.DataFrame, y, repeats: int = 5, seed: int = 42) -> pd.DataFrame:  # noqa: N803
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=repeats,
        random_state=seed,
        # An explicit 1-D positive-class score avoids version-dependent scorer
        # behavior for soft-voting estimators that expose an (n, 2) matrix.
        scoring=_positive_class_roc_auc,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean_roc_auc_decrease": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_roc_auc_decrease", ascending=False, ignore_index=True)


def shap_explanation(model, X: pd.DataFrame, max_rows: int = 500):  # noqa: N803
    """Return a SHAP Explanation when supported; otherwise signal the caller to use permutation importance."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed; use permutation importance for a valid model-agnostic explanation.") from exc
    sample = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
    try:
        explainer = shap.Explainer(model.predict_proba, sample)
        return explainer(sample)
    except Exception as exc:
        raise RuntimeError(f"SHAP could not explain this model; use permutation importance instead ({exc}).") from exc


def local_shap_explanation(model, background: pd.DataFrame, row: pd.DataFrame) -> pd.DataFrame:
    """Explain one binary-class prediction against a small background sample."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed.") from exc
    if row.empty:
        raise ValueError("Select a transaction row to explain.")
    if background.empty:
        raise ValueError("At least one background transaction is required.")
    try:
        explainer = shap.Explainer(model.predict_proba, background)
        explanation = explainer(row)
        values = explanation.values
        if isinstance(values, list):
            values = values[1]
        values = values[0] if len(values.shape) >= 2 else values
        if len(values.shape) == 2:
            values = values[:, 1]
        contributions = pd.DataFrame({"feature": row.columns, "contribution": values})
        contributions["direction"] = contributions["contribution"].map(lambda value: "raises return score" if value > 0 else "lowers return score")
        contributions["absolute_contribution"] = contributions["contribution"].abs()
        return contributions.sort_values("absolute_contribution", ascending=False, ignore_index=True)
    except Exception as exc:
        raise RuntimeError(f"SHAP could not explain this row for the selected model ({exc}).") from exc
