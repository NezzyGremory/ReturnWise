"""Association tests with effect sizes and Benjamini-Hochberg adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def _bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    adjusted = p * len(p) / ranks
    sorted_adj = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    result = np.empty(len(p), dtype=float)
    result[order] = np.clip(sorted_adj, 0, 1)
    return result.tolist()


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def _bootstrap_median_diff_ci(x: np.ndarray, y: np.ndarray, seed: int = 42, iterations: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    # Bound bootstrap memory and runtime on large competition CSVs.
    sample_x, sample_y = min(len(x), 5000), min(len(y), 5000)
    for i in range(iterations):
        differences[i] = np.median(rng.choice(x, size=sample_x, replace=True)) - np.median(
            rng.choice(y, size=sample_y, replace=True)
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def association_tests(
    df: pd.DataFrame,
    target: str = "returned",
    exclude: list[str] | None = None,
    max_groups: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare numeric distributions or categorical proportions by binary target.

    Numeric comparisons use the two-sided Mann-Whitney U test, rank-biserial
    effect size, and a bootstrap CI for the difference in medians. Categorical
    comparisons use Pearson's chi-square and Cramer's V. P-values are corrected
    together using the Benjamini-Hochberg procedure.
    """
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` is not available.")
    numeric_target = pd.to_numeric(df[target], errors="coerce")
    work = df.loc[numeric_target.isin([0, 1])].copy()
    work[target] = numeric_target.loc[work.index].astype(int)
    excluded = set(exclude or []) | {target}
    rows: list[dict] = []
    for feature in work.columns:
        if feature in excluded or work[feature].nunique(dropna=True) < 2:
            continue
        series = work[feature]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > max_groups:
            x = series[work[target].astype(int).eq(1)].dropna().to_numpy(dtype=float)
            y = series[work[target].astype(int).eq(0)].dropna().to_numpy(dtype=float)
            if min(len(x), len(y)) < 2:
                continue
            test = mannwhitneyu(x, y, alternative="two-sided")
            low, high = _bootstrap_median_diff_ci(x, y, seed=seed)
            rows.append(
                {
                    "feature": feature,
                    "test": "Mann-Whitney U (two-sided)",
                    "n_returned": len(x),
                    "n_kept": len(y),
                    "summary_returned": float(np.median(x)),
                    "summary_kept": float(np.median(y)),
                    "effect_size": _rank_biserial(x, y),
                    "effect_label": "rank-biserial correlation",
                    "median_difference_ci_low": low,
                    "median_difference_ci_high": high,
                    "p_value": float(test.pvalue),
                }
            )
        else:
            levels = series.astype("string").fillna("<missing>")
            if levels.nunique() > max_groups:
                continue
            table = pd.crosstab(levels, work[target].astype(int))
            if table.shape[1] < 2 or table.shape[0] < 2:
                continue
            chi2, pvalue, _, _ = chi2_contingency(table, correction=False)
            n = table.to_numpy().sum()
            denom = max(1, min(table.shape[0] - 1, table.shape[1] - 1))
            rows.append(
                {
                    "feature": feature,
                    "test": "Pearson chi-square",
                    "n_returned": int(table[1].sum()) if 1 in table.columns else 0,
                    "n_kept": int(table[0].sum()) if 0 in table.columns else 0,
                    "summary_returned": None,
                    "summary_kept": None,
                    "effect_size": float(np.sqrt((chi2 / n) / denom)) if n else np.nan,
                    "effect_label": "Cramer's V",
                    "median_difference_ci_low": None,
                    "median_difference_ci_high": None,
                    "p_value": float(pvalue),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_adjusted_bh"] = _bh_adjust(result["p_value"].tolist())
        result["significant_fdr_0_05"] = result["p_adjusted_bh"] < 0.05
        result = result.sort_values("p_adjusted_bh").reset_index(drop=True)
    return result
