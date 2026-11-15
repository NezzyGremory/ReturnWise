import pandas as pd
import streamlit as st

from returnwise.analysis.statistics import association_tests
from returnwise.data.profile import eda_summaries
from returnwise.features import temporal_feature_audit
from returnwise.ui import require_dataset

st.caption("WORKSPACE / EXPLORATORY ANALYSIS")
st.title("Explore dataset patterns")
frame, manifest = require_dataset()
if frame is not None:
    summary = eda_summaries(frame)
    st.caption(f"Active source: {manifest.get('source', 'unknown')} · {len(frame):,} rows")
    st.subheader("Feature availability at prediction time")
    st.caption("The research default is pre-shipment. History aggregates are excluded unless their cutoff can be verified; delivery delay is treated as post-shipment.")
    st.dataframe(temporal_feature_audit(list(frame.columns)), hide_index=True, alt="Feature-level availability audit for a pre-shipment prediction scenario")
    if summary["target_distribution"] is None:
        st.info("This dataset has no `returned` labels. Target comparisons and supervised analysis are unavailable.")
    else:
        distribution = summary["target_distribution"]
        st.subheader("Return outcome")
        returned = pd.to_numeric(frame["returned"], errors="coerce")
        rate = returned.mean()
        left, right = st.columns([1, 2])
        left.metric("Observed return proportion", f"{rate:.1%}")
        right.bar_chart(distribution, x="returned", y="rows", alt="Observed returned and kept transaction counts")
        st.dataframe(distribution, hide_index=True, alt="Observed returned class counts and proportions")
    numeric = summary["numeric_summary"]
    if not numeric.empty:
        st.subheader("Numerical feature distributions")
        st.dataframe(numeric, hide_index=True, alt="Descriptive statistics for numerical columns")
    categorical = summary["categorical_summary"]
    if not categorical.empty:
        st.subheader("Categorical feature distributions")
        st.dataframe(categorical, hide_index=True, alt="Observed level counts and frequencies for categorical columns")
    if "returned" in frame.columns:
        with st.spinner("Running non-parametric and contingency-table association tests…"):
            results = association_tests(frame)
        st.subheader("Association tests")
        st.caption("Numeric features: two-sided Mann-Whitney U, rank-biserial effect size, and bootstrap median-difference intervals. Categorical features: chi-square and Cramér’s V. Benjamini-Hochberg correction controls the false-discovery rate across tested features.")
        if results.empty:
            st.info("No eligible feature-target comparisons were found.")
        else:
            st.dataframe(results, hide_index=True, alt="Feature association tests with adjusted p values, effect sizes, and confidence intervals")
            st.warning("Associations are descriptive and do not establish causal effects.")
