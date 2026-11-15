import json

import pandas as pd
import streamlit as st

from returnwise.data.kaggle import read_metadata
from returnwise.experiments import read_experiment_history
from returnwise.report import generate_research_report
from returnwise.research_state import STATE_FILE
from returnwise.storage import active_dataset_fingerprint, load_active_dataset, load_model_artifact


@st.cache_data(show_spinner=False)
def _cached_research_report(dataset_fingerprint: str, state_modified_ns: int, metadata_json: str) -> str:
    metadata = json.loads(metadata_json) if metadata_json else None
    return generate_research_report(None, metadata)


st.caption("METHODS / DATA / MODEL STATUS")
st.title("Research evidence and limits")
st.write(
    "ReturnWise estimates return probabilities and ranks transactions for review under a capacity limit. "
    "The current evidence is exploratory and does not show that a review prevents a return or improves business outcomes."
)

frame, manifest = load_active_dataset()
artifact = load_model_artifact()
history = read_experiment_history()
meta = read_metadata()

st.warning(
    "The corrected raw-feature candidates have no independent holdout evaluation or deployment refit. The saved model is a historical artifact from the earlier pipeline.",
    title="No corrected deployment model",
)

st.subheader("Dataset and prediction target")
integrity = (meta or {}).get("integrity", {})
training_rows = integrity.get("train_rows_inspected")
test_rows = integrity.get("test_rows_inspected")
metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Labeled training rows", f"{training_rows:,}" if training_rows is not None else "Not available")
metric_b.metric("Unlabeled test rows", f"{test_rows:,}" if test_rows is not None else "Not available")
metric_c.metric("Target", "returned (binary)")
st.caption(
    "The competition uses ROC-AUC. No transaction timestamp or feature snapshot cutoff is available, so the project does not claim temporal validation or prove every candidate field was known before shipment."
)
if meta:
    st.caption(f"Dataset version: {(meta or {}).get('dataset_version', 'unknown')} · Fingerprint: {(meta or {}).get('fingerprint_sha256', 'not recorded')}")
elif manifest:
    row_label = f"{len(frame):,}" if frame is not None else "unavailable"
    st.caption(f"Active source: {manifest.get('source', 'unknown')} · {row_label} rows")
else:
    st.info("No verified dataset metadata is available in this workspace.")

st.subheader("Corrected pipeline: exploratory results")
st.metric("Three-fold OOF ROC-AUC", "0.587648")
st.caption(
    "This is the out-of-fold score of the raw Logistic Regression and HistGradientBoosting blend on one three-fold setup. It is exploratory, has no confidence interval, and is not an independent holdout result."
)

st.subheader("Retained artifact: historical internal holdout")
old_a, old_b, old_c, old_d = st.columns(4)
old_a.metric("ROC-AUC", "0.589481")
old_b.metric("Average precision", "0.556060")
old_c.metric("Brier score", "0.243089")
old_d.metric("Log loss", "0.679141")
st.caption(
    "These values belong to the original soft-vote model on a random 40,000-row internal holdout. That pipeline predates the raw-feature correction, so these scores do not evaluate the corrected candidates or establish temporal generalization."
)
st.caption("Sources: reports/final_research_findings.md, reports/FINAL_RETURNWISE_RESULTS.md, and the artifact metadata.")

with st.expander("Feature timing and preprocessing"):
    st.write(
        "The corrected candidate uses 11 raw fields: customer age, product price, discount, product rating, session length, product views, device type, product category, shipping method, payment method, and coupon use."
    )
    st.write(
        "The target returned is label-only. order_id is treated as an identifier, delivery_delay_days as post-shipment, and customer history fields remain excluded until their event cutoffs can be verified."
    )
    st.write(
        "The source data contain undocumented negative prices and product-view counts, out-of-range discounts, and ratings outside the common 0 to 5 range. Raw values were not changed because their meaning is unknown."
    )

st.subheader("Capacity-based prioritization")
st.write(
    "The queue sorts records by estimated calibrated probability. A selected capacity marks the first k records for review. The simulator sums probabilities to report expected counts for unlabeled transactions; cost inputs are assumptions."
)
st.caption("A ranking describes model output. It does not establish causality, guarantee return identification, or show that review will prevent a return.")

st.subheader("Dataset provenance")
if meta:
    st.json(
        {
            key: meta.get(key)
            for key in [
                "competition",
                "dataset_version",
                "fingerprint_sha256",
                "last_synced_at",
                "cache_validated_at",
                "last_checked_at",
                "acquisition_mode",
                "integrity",
                "archive_validation",
            ]
        }
    )
elif manifest:
    st.json(manifest)
else:
    st.info("No dataset provenance is available yet.")

st.subheader("Experiment history")
if history.empty:
    st.info("No experiment runs have been recorded. Run the sequential notebooks in notebooks/ to create research results.")
else:
    st.dataframe(history, hide_index=True, alt="Recorded experiment runs, timestamps, dataset fingerprints, and metrics")

st.subheader("Exported artifact")
if artifact:
    st.info("This artifact is retained for research scoring. It is not a corrected-pipeline deployment model.", title="Historical artifact")
    with st.expander("Artifact metadata"):
        st.json(artifact.get("metadata", {}))
else:
    st.info("No model artifact is available. ReturnWise does not display placeholder model scores.")

st.subheader("Research report")
dataset_key = active_dataset_fingerprint() or (meta or {}).get("fingerprint_sha256", "")
state_modified = STATE_FILE.stat().st_mtime_ns if STATE_FILE.is_file() else 0
metadata_json = json.dumps(artifact.get("metadata") if artifact else None, sort_keys=True, default=str)
report = _cached_research_report(dataset_key, state_modified, metadata_json)
st.download_button("Download current report", data=report.encode("utf-8"), file_name="returnwise_research_report.md", mime="text/markdown", width="content")
with st.expander("Preview report"):
    st.markdown(report)
