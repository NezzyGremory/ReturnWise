import streamlit as st

from returnwise import __version__
from returnwise.storage import load_model_artifact
from returnwise.ui import active_data


frame, manifest = active_data()
artifact = load_model_artifact()

st.caption("RETURNWISE / DECISION SUPPORT")
st.title("Prioritize return-risk reviews by capacity.")
st.write(
    "ReturnWise estimates return probabilities and ranks transactions so a team can focus a limited review queue. "
    "A probability is an estimate, and a review priority does not guarantee that a return will be prevented."
)

action, secondary = st.columns([1, 1], vertical_alignment="center")
with action:
    st.page_link("app_pages/04_risk_queue.py", label="Open risk queue", icon=":material/low_priority:")
with secondary:
    st.page_link("app_pages/06_research.py", label="Review methods and evidence", icon=":material/science:")

st.divider()
left, right = st.columns([1.3, 0.9], gap="large")
with left:
    st.subheader("A review list shaped by the time you have")
    st.write(
        "Import an unlabeled transaction file, check its fields, and score it with the retained research artifact. "
        "The queue sorts by estimated return probability and marks the transactions that fit your review capacity."
    )
    st.page_link("app_pages/02_data.py", label="Set up a dataset", icon=":material/database:")
    st.page_link("app_pages/03_analysis.py", label="Inspect dataset patterns", icon=":material/query_stats:")

with right:
    with st.container(border=True):
        st.caption("Current workspace")
        if frame is None:
            st.info(
                "Select a Kaggle dataset or import a CSV to inspect its schema and quality.",
                title="No active dataset",
            )
            st.page_link("app_pages/02_data.py", label="Open dataset setup", icon=":material/upload_file:")
        else:
            data_label = (manifest or {}).get("source", "unknown").title()
            st.write(f"**{data_label} dataset**")
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Research rows", f"{len(frame):,}")
            metric_b.metric("Fields", f"{len(frame.columns):,}")
            if "returned" in frame.columns:
                labels = frame["returned"].dropna()
                if not labels.empty:
                    st.caption(f"Observed return share in this labeled dataset: {labels.mean():.1%}")
            st.caption("Dataset counts describe research input. They are not scored orders.")
        if artifact:
            st.warning(
                "The available model is a historical research artifact. Corrected-pipeline validation and a deployment refit are not complete.",
                title="Scoring status: experimental",
            )
        else:
            st.info("No model artifact is available. Risk scoring stays unavailable until a real model is exported.", title="Model unavailable")

st.divider()
st.subheader("What ReturnWise supports")
one, two = st.columns(2, gap="medium")
with one:
    with st.container(border=True):
        st.markdown("#### Risk estimation")
        st.write("Score unlabeled transactions with the retained artifact and keep its experimental status visible.")
        st.page_link("app_pages/04_risk_queue.py", label="Score transactions", icon=":material/arrow_forward:")
with two:
    with st.container(border=True):
        st.markdown("#### Capacity-based ranking")
        st.write("Sort by estimated probability, choose a review count, and see which rows fit that limit.")
        st.page_link("app_pages/04_risk_queue.py", label="Set review capacity", icon=":material/arrow_forward:")

three, four = st.columns(2, gap="medium")
with three:
    with st.container(border=True):
        st.markdown("#### Scenario analysis")
        st.write("Compare review strategies under cost and capacity assumptions you provide.")
        st.page_link("app_pages/05_decision_simulator.py", label="Open simulator", icon=":material/arrow_forward:")
with four:
    with st.container(border=True):
        st.markdown("#### Data and model evidence")
        st.write("Inspect feature timing, dataset quality, evaluation protocols, and study limitations.")
        st.page_link("app_pages/06_research.py", label="Read research status", icon=":material/arrow_forward:")

st.caption(f"ReturnWise {__version__} · Local research workspace. No claim of return prevention or business impact.")
