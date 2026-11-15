"""Feature timing audit and reproducible tabular feature transformations."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def temporal_feature_audit(columns: list[str], target: str = "returned") -> pd.DataFrame:
    rows = []
    for name in columns:
        low = name.strip().lower()
        if name == target:
            availability, decision = "label; never a feature", "exclude"
        elif low == "delivery_delay_days" or "actual_delivery" in low:
            availability, decision = "post-shipment outcome", "exclude from pre-shipment prediction"
        elif low.endswith("_id") or low == "id":
            availability, decision = "identifier; not a transaction behavior", "exclude unless a documented use case justifies it"
        elif low in {"past_return_rate", "past_purchase_count"}:
            availability, decision = "historical aggregate; event order and cutoff unavailable", "exclude until leakage-safe construction is verified"
        elif re.search(r"return|refund|delivered|delivery_date|outcome|label", low):
            availability, decision = "potentially post-event or target-derived", "exclude pending documentation"
        else:
            availability, decision = "likely available by order placement; verify with data documentation", "eligible for pre-shipment candidate set"
        rows.append({"feature": name, "availability_assessment": availability, "decision": decision})
    return pd.DataFrame(rows)


def select_features(df: pd.DataFrame, target: str = "returned", scenario: str = "pre_shipment") -> list[str]:
    audit = temporal_feature_audit(list(df.columns), target)
    excluded = {target}
    if scenario == "pre_shipment":
        excluded.update(audit.loc[audit["decision"].str.startswith("exclude"), "feature"].tolist())
        excluded.update(audit.loc[audit["decision"].str.startswith("exclude until"), "feature"].tolist())
    else:
        excluded.update(audit.loc[audit["decision"].str.startswith("exclude unless"), "feature"].tolist())
        excluded.update(audit.loc[audit["decision"].str.startswith("exclude pending"), "feature"].tolist())
    return [c for c in df.columns if c not in excluded]


class ReturnWiseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic interactions based only on supplied row-level inputs."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        result = pd.DataFrame(X).copy()
        if {"product_price", "discount_percent"}.issubset(result.columns):
            result["price_discount_interaction"] = pd.to_numeric(result["product_price"], errors="coerce") * pd.to_numeric(
                result["discount_percent"], errors="coerce"
            )
        if {"num_product_views", "session_length_minutes"}.issubset(result.columns):
            duration = pd.to_numeric(result["session_length_minutes"], errors="coerce").replace(0, np.nan)
            result["views_per_session_minute"] = pd.to_numeric(result["num_product_views"], errors="coerce") / duration
        if {"product_price", "product_rating"}.issubset(result.columns):
            result["price_rating_interaction"] = pd.to_numeric(result["product_price"], errors="coerce") * pd.to_numeric(
                result["product_rating"], errors="coerce"
            )
        for column in result.select_dtypes(include=["number"]).columns:
            result[column] = result[column].replace([float("inf"), float("-inf")], np.nan)
        return result
