import json

import pandas as pd
import streamlit as st

from returnwise.data.kaggle import read_metadata
from returnwise.experiments import read_experiment_history
from returnwise.report import generate_research_report
from returnwise.storage import load_active_dataset, load_model_artifact

st.title("Models and research")
frame, manifest = load_active_dataset()
artifact = load_model_artifact()
history = read_experiment_history()

st.subheader("Dataset provenance")
meta = read_metadata()
if meta:
    st.write("Kaggle cache metadata")
    st.json({
        k: meta.get(k)
        for k in [
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
    })
elif manifest:
    st.write("Active source", manifest)
else:
    st.info("No dataset provenance is available yet.")

st.subheader("Experiment history")
if history.empty:
    st.info("No experiment runs have been recorded. Run the sequential notebooks in `notebooks/` to create research results.")
else:
    st.dataframe(history, hide_index=True, alt="Recorded experiment runs, timestamps, dataset fingerprints, and metrics")

st.subheader("Exported model")
if artifact:
    st.json(artifact.get("metadata", {}))
else:
    st.info("No model artifact is available. ReturnWise does not display placeholder model scores.")

st.subheader("Research report")
report = generate_research_report(frame, artifact.get("metadata") if artifact else None)
st.download_button("Download current report", data=report.encode("utf-8"), file_name="returnwise_research_report.md", mime="text/markdown", width="content")
with st.expander("Preview report"):
    st.markdown(report)
