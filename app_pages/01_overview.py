import streamlit as st

from returnwise import __version__
from returnwise.ui import active_data, app_status
from returnwise.storage import load_model_artifact

st.title("ReturnWise")
st.write("**Intelligent E-Commerce Return Risk Management**")
st.caption("Research workspace for return-risk analysis, calibrated probabilities, and capacity-aware transaction review.")

app_status()
frame, manifest = active_data()
artifact = load_model_artifact()

left, right = st.columns([1.4, 1], gap="large")
with left:
    st.subheader("Research workflow")
    st.write("Connect or import a dataset, inspect data quality, run the research notebooks, then export a model artifact for transaction scoring.")
    st.page_link("app_pages/02_data.py", label="Set up dataset", icon=":material/database:")
    st.page_link("app_pages/03_analysis.py", label="Explore data", icon=":material/query_stats:")
    st.page_link("app_pages/06_research.py", label="Review experiments", icon=":material/experiment:")
with right:
    st.subheader("Current workspace")
    if frame is None:
        st.info("No dataset has been selected. No dataset metrics or predictions are shown.", title="Waiting for data")
    else:
        st.metric("Rows", f"{len(frame):,}")
        st.metric("Columns", f"{len(frame.columns):,}")
        st.caption(f"Source: {manifest.get('source', 'unknown')} · Target present: {'returned' in frame.columns}")
    if artifact:
        metrics = artifact.get("metadata", {}).get("validation_metrics", {})
        st.metric("Exported model", artifact.get("metadata", {}).get("selected_model", "unknown"))
        if metrics:
            metric = metrics.get("roc_auc")
            if metric is not None:
                st.metric("Recorded validation ROC-AUC", f"{metric:.3f}")
    else:
        st.info("No trained model artifact is available. Prediction remains unavailable until a real model is trained and exported.", title="Model not trained")

with st.expander("Research scope and limitations"):
    st.write("The system estimates return probability and helps prioritize reviews under user-entered cost and capacity scenarios. It cannot establish that review prevents returns or increases profit. Cost inputs are simulations, and feature importance is not causal evidence.")
    st.caption(f"ReturnWise version {__version__} · Prediction timing is audited as pre-shipment by default.")
