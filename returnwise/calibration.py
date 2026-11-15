"""Post-hoc probability calibration and reliability metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Fit sigmoid, isotonic, or experimental beta calibration on held-out scores."""

    def __init__(self, method: str = "sigmoid"):
        allowed = {"sigmoid", "isotonic", "beta"}
        if method not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        self.method = method
        self.model = None

    @staticmethod
    def _clip(probabilities):
        return np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)

    def _design(self, probabilities):
        p = self._clip(probabilities)
        if self.method == "sigmoid":
            return np.log(p / (1 - p)).reshape(-1, 1)
        return p

    def fit(self, probabilities, y):
        p = self._clip(probabilities)
        y = np.asarray(y, dtype=int).reshape(-1)
        if len(p) != len(y) or len(np.unique(y)) != 2:
            raise ValueError("Calibration requires aligned probabilities and both target classes.")
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        elif self.method == "beta":
            log_p = np.log(p)
            neg_log_one_minus_p = -np.log1p(-p)

            def objective(parameters):
                a, b, intercept = parameters
                logits = intercept + a * log_p + b * neg_log_one_minus_p
                loss = np.logaddexp(0, logits) - y * logits
                residual = expit(logits) - y
                gradient = np.array(
                    [np.dot(residual, log_p), np.dot(residual, neg_log_one_minus_p), residual.sum()]
                )
                return float(loss.mean()), gradient / len(y)

            initial = np.array([1.0, 1.0, 0.0])
            fitted = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                jac=True,
                bounds=[(0, None), (0, None), (None, None)],
            )
            if not fitted.success and not np.isfinite(fitted.fun):
                raise ValueError(f"Beta calibration did not converge: {fitted.message}")
            self.model = fitted.x
        else:
            self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(self._design(p), y)
        return self

    def predict(self, probabilities):
        if self.model is None:
            raise RuntimeError("Call fit using a calibration split before requesting calibrated values.")
        p = self._clip(probabilities)
        if self.method == "isotonic":
            result = self.model.predict(p)
        elif self.method == "beta":
            a, b, intercept = self.model
            result = expit(intercept + a * np.log(p) + b * -np.log1p(-p))
        else:
            result = self.model.predict_proba(self._design(p))[:, 1]
        return np.clip(result, 0, 1)


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1 else p <= high)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece) if total else float("nan")
