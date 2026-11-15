"""Schema discovery, conservative quality checks, and descriptive summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TARGET = "returned"
IDENTIFIER_NAMES = {"id", "transaction_id", "order_id", "customer_id", "product_id", "record_id"}
KNOWN_DESCRIPTIONS = {
    "customer_age": "Customer age (competition description).",
    "past_purchase_count": "Number of the customer's past purchases (competition description; timing needs verification).",
    "past_return_rate": "Historical customer return rate (competition description; leakage safety needs verification).",
    "product_price": "Product price at purchase (competition description).",
    "discount_percent": "Discount applied to the product (competition description).",
    "product_rating": "Average product rating (competition description).",
    "session_length_minutes": "Duration of the browsing session in minutes (competition description).",
    "num_product_views": "Products viewed during the session (competition description).",
    "used_coupon": "Whether a coupon was used (competition description).",
    "shipping_method": "Selected shipping option (competition description).",
    "delivery_delay_days": "Delay against estimated delivery (competition description; likely post-shipment).",
    "device_type": "Device category used for the session (competition description).",
    "product_category": "Product category (competition description).",
    "payment_method": "Payment type (competition description).",
    "returned": "Binary competition target: 1 returned, 0 kept.",
}


def _role(name: str, series: pd.Series, target: str | None) -> str:
    low = name.strip().lower()
    if target and name == target:
        return "target"
    if low in IDENTIFIER_NAMES or low.endswith("_id") or low == "id":
        return "identifier candidate"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric feature"
    return "categorical feature"


def data_dictionary(df: pd.DataFrame, target: str | None = TARGET) -> pd.DataFrame:
    rows = []
    for name in df.columns:
        series = df[name]
        examples = series.dropna().astype(str).drop_duplicates().head(4).tolist()
        rows.append(
            {
                "column": name,
                "role": _role(name, series, target),
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "unique_values": int(series.nunique(dropna=True)),
                "examples_from_file": ", ".join(examples),
                "description": KNOWN_DESCRIPTIONS.get(name, "Meaning not documented; inferred from observed values only."),
            }
        )
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame, target: str | None = TARGET) -> dict[str, Any]:
    nrows, ncols = df.shape
    missing = df.isna().sum().rename("missing_rows").to_frame()
    missing["missing_percent"] = (missing["missing_rows"] / max(nrows, 1) * 100).round(3)
    missing = missing[missing["missing_rows"] > 0].sort_values("missing_rows", ascending=False)
    id_cols = [c for c in df.columns if c.strip().lower() in IDENTIFIER_NAMES or c.strip().lower().endswith("_id")]
    duplicate_ids = {c: int(df[c].duplicated().sum()) for c in id_cols}
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    outliers: dict[str, int] = {}
    invalid_values: dict[str, dict[str, Any]] = {}
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        finite = values[np.isfinite(values)]
        non_finite = int((~np.isfinite(values)).sum())
        if non_finite:
            invalid_values[col] = {"issue": "Positive or negative infinity", "rows": non_finite}
        values = finite
        if len(values) > 0:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers[col] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    target_counts = None
    if target and target in df.columns:
        target_counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="rows")
        target_counts["percent"] = target_counts["rows"] / max(nrows, 1) * 100
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        invalid_target = int((df[target].notna() & ~target_numeric.isin([0, 1])).sum())
        if invalid_target:
            invalid_values[target] = {"issue": "Target values outside binary 0/1", "rows": invalid_target}
    nonnegative = ("product_price", "past_purchase_count", "session_length_minutes", "num_product_views")
    for col in nonnegative:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            count = int((values < 0).sum())
            if count:
                invalid_values[col] = {"issue": "Negative value where the documented quantity is non-negative", "rows": count}
    if "customer_age" in df.columns:
        values = pd.to_numeric(df["customer_age"], errors="coerce")
        count = int(((values < 0) | (values > 120)).sum())
        if count:
            invalid_values["customer_age"] = {"issue": "Age outside the screening range 0–120 years", "rows": count}
    if "discount_percent" in df.columns:
        values = pd.to_numeric(df["discount_percent"], errors="coerce")
        count = int(((values < 0) | (values > 100)).sum())
        if count:
            invalid_values["discount_percent"] = {"issue": "Discount percent outside 0–100", "rows": count}
    if "product_rating" in df.columns:
        values = pd.to_numeric(df["product_rating"], errors="coerce")
        count = int(((values < 0) | (values > 5)).sum())
        if count:
            invalid_values["product_rating"] = {"issue": "Rating outside the common 0–5 screening range; confirm source scale", "rows": count}
    numeric_text_candidates = {}
    category_inconsistencies = {}
    for col in df.select_dtypes(exclude=["number", "bool"]).columns:
        non_null = df[col].dropna().astype("string")
        if len(non_null):
            coerced = pd.to_numeric(non_null, errors="coerce")
            conversion_rate = float(coerced.notna().mean())
            if conversion_rate >= 0.9:
                numeric_text_candidates[col] = {"numeric_parseable_percent": conversion_rate * 100, "non_numeric_examples": non_null[coerced.isna()].head(5).tolist()}
            normalized_levels = non_null.str.strip().str.casefold()
            if normalized_levels.nunique() < non_null.nunique():
                category_inconsistencies[col] = {
                    "raw_unique": int(non_null.nunique()),
                    "casefold_trim_unique": int(normalized_levels.nunique()),
                }
    from returnwise.features import temporal_feature_audit

    leakage_audit = temporal_feature_audit(list(df.columns), target=target or TARGET)
    leakage_candidates = leakage_audit[leakage_audit["decision"].str.contains("exclude", case=False)]
    duplicate_rows = int(df.duplicated().sum())
    return {
        "rows": int(nrows),
        "columns": int(ncols),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_percent": float(duplicate_rows / max(nrows, 1) * 100),
        "missing": missing.reset_index(names="column"),
        "identifier_columns": id_cols,
        "duplicate_identifiers": duplicate_ids,
        "iqr_outlier_counts": outliers,
        "target_counts": target_counts,
        "invalid_values": invalid_values,
        "numeric_text_candidates": numeric_text_candidates,
        "categorical_inconsistencies": category_inconsistencies,
        "leakage_candidates": leakage_candidates,
        "potentially_unreliable_features": sorted(set(id_cols) | set(leakage_candidates["feature"].tolist())),
        "schema": data_dictionary(df, target),
    }


def eda_summaries(df: pd.DataFrame, target: str = TARGET) -> dict[str, Any]:
    # pandas 3 introduces a distinct ``str`` dtype; selecting ``object`` also
    # includes it for compatibility but emits a deprecation warning.
    string_selector = "str" if int(pd.__version__.split(".", 1)[0]) >= 3 else "object"
    result: dict[str, Any] = {
        "shape": df.shape,
        "numeric_summary": df.describe(include=["number"]).T.reset_index(names="feature"),
        "categorical_summary": df.describe(include=[string_selector, "category", "bool"]).T.reset_index(names="feature"),
    }
    if target in df.columns:
        counts = df[target].value_counts(dropna=False).rename_axis("returned").reset_index(name="rows")
        counts["share"] = counts["rows"] / max(len(df), 1)
        result["target_distribution"] = counts
    else:
        result["target_distribution"] = None
    return result
