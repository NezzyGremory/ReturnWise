import numpy as np
import pandas as pd
import streamlit as st

from returnwise.decision import priority_queue
from returnwise.explain import local_shap_explanation
from returnwise.export import predict_with_artifact
from returnwise.storage import active_dataset_fingerprint, load_model_artifact


st.caption("DECISION SUPPORT / RISK QUEUE")
st.title("Prioritize transactions for review")
st.write(
    "Upload an unlabeled CSV to estimate return probabilities, rank transactions, and select a review list within your team's capacity."
)
artifact = load_model_artifact()

if not artifact:
    st.info(
        "Run the research notebooks on real labeled data, select a candidate using a valid holdout, and export an artifact before scoring.",
        title="Prediction unavailable",
    )
else:
    metadata = artifact.get("metadata", {})
    st.warning(
        "The retained artifact predates the corrected raw-feature pipeline. Scores can be inspected for research use, but this is not a validated production model.",
        title="Experimental scoring only",
    )
    st.caption(
        f"Artifact: {metadata.get('selected_model', 'unknown')} · Scenario: {metadata.get('prediction_scenario', 'unspecified')} · Exported: {metadata.get('exported_at', 'unknown')}"
    )

    current_fingerprint = active_dataset_fingerprint()
    model_fingerprint = metadata.get("dataset_fingerprint")
    stale_model = bool(current_fingerprint and model_fingerprint and current_fingerprint != model_fingerprint)
    if stale_model:
        st.error(
            "The active research dataset differs from the dataset used for this artifact. Select the matching dataset before research scoring.",
            icon=":material/error:",
        )

    st.info(
        "Upload unlabeled transactions. Remove `returned` before scoring. Missing model fields are handled by the saved pipeline and shown in a warning.",
        title="Input requirements",
    )
    uploaded = st.file_uploader("Transactions to score", type=["csv"], key="risk_upload", disabled=stale_model)
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            if frame.empty:
                st.error("The uploaded CSV has no rows.")
            elif "returned" in frame.columns:
                st.error("Remove the `returned` target before scoring. Use a held-out labeled split in the research notebooks for evaluation.")
            else:
                missing = [column for column in artifact["feature_columns"] if column not in frame.columns]
                if missing:
                    st.warning("Missing model inputs will be passed through the trained imputer: " + ", ".join(missing))
                raw, calibrated = predict_with_artifact(artifact, frame)
                queue = priority_queue(frame, raw, calibrated)
                st.session_state["scored_queue"] = queue
                st.session_state["scored_model_metadata"] = metadata
                st.success(f"Scored {len(queue):,} transactions using the exported model.")

                st.subheader("Set review capacity")
                capacity_choice = st.selectbox(
                    "Transactions your team can review",
                    options=[10, 25, 50, 100, "Custom count"],
                    index=1,
                    key="risk_review_capacity_choice",
                )
                if capacity_choice == "Custom count":
                    capacity = int(
                        st.number_input(
                            "Custom review count",
                            min_value=1,
                            max_value=len(queue),
                            value=min(25, len(queue)),
                            step=1,
                            key="risk_review_capacity_custom",
                        )
                    )
                else:
                    capacity = min(int(capacity_choice), len(queue))
                priority_set = queue.head(capacity).copy()
                view = queue.copy()
                view["review_status"] = np.where(
                    view["priority_rank"] <= capacity,
                    "Within selected capacity",
                    "Outside selected capacity",
                )
                view.insert(0, "review_status", view.pop("review_status"))

                metric_a, metric_b, metric_c = st.columns(3)
                metric_a.metric("Transactions scored", f"{len(queue):,}")
                metric_b.metric("Review capacity", f"{capacity:,}")
                metric_c.metric("Transactions prioritized", f"{len(priority_set):,}")
                st.caption("Priority rank sorts estimated calibrated return probabilities from highest to lowest. The selected list is a review aid, not a return-prevention guarantee.")

                search = st.text_input("Search transaction fields", key="risk_search")
                if search:
                    mask = view.astype(str).apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
                    view = view[mask]
                if len(view) > 1_000:
                    st.caption("Showing the first 1,000 matching rows. The download includes every scored transaction.")
                st.dataframe(
                    view.head(1_000),
                    hide_index=True,
                    height=500,
                    alt="Ranked transactions showing estimated return probability and whether each row fits the selected review capacity",
                    column_config={
                        "priority_rank": st.column_config.NumberColumn("Priority rank", format="%d"),
                        "calibrated_probability": st.column_config.ProgressColumn(
                            "Estimated return probability", min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "return_probability": st.column_config.NumberColumn("Raw model score", format="percent"),
                    },
                )
                download_a, download_b = st.columns(2)
                with download_a:
                    st.download_button(
                        "Download priority set",
                        data=priority_set.to_csv(index=False).encode("utf-8"),
                        file_name="returnwise_priority_review.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                with download_b:
                    st.download_button(
                        "Download full risk queue",
                        data=queue.to_csv(index=False).encode("utf-8"),
                        file_name="returnwise_risk_queue.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                with st.expander("Historical evaluation and model status"):
                    validation = metadata.get("validation_metrics", {})
                    key_metrics = {
                        key: validation.get(key)
                        for key in ["roc_auc", "pr_auc", "brier_score", "log_loss", "expected_calibration_error"]
                        if validation.get(key) is not None
                    }
                    if key_metrics:
                        st.caption(
                            "These results are from the original random internal holdout. They predate the corrected raw-feature pipeline and do not validate the current scoring artifact for production."
                        )
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Historical holdout ROC-AUC", f"{key_metrics['roc_auc']:.3f}")
                        m2.metric("Historical average precision", f"{key_metrics['pr_auc']:.3f}")
                        m3.metric("Historical Brier score", f"{key_metrics['brier_score']:.3f}")
                    reliability = validation.get("reliability_curve")
                    if reliability:
                        reliability_frame = pd.DataFrame(reliability)
                        st.line_chart(
                            reliability_frame,
                            x="mean_predicted_probability",
                            y="observed_return_rate",
                            alt="Historical holdout return rate by predicted-probability bin for the original model pipeline",
                        )
                    if not metadata.get("probability_is_calibrated", False):
                        st.warning("The artifact metadata does not confirm a selected calibrator. Treat its outputs as scores, not calibrated probabilities.")

                importance = artifact.get("permutation_importance")
                if importance:
                    st.subheader("Saved feature reliance")
                    st.caption("Historical permutation importance shows which fields affected this model's holdout score. It is not causal evidence and may not transfer to a corrected model.")
                    st.dataframe(
                        pd.DataFrame(importance),
                        hide_index=True,
                        alt="Historical permutation feature importance measured on the original model holdout",
                    )
                else:
                    st.info("No validation-set permutation importance was saved with this artifact.")

                st.subheader("On-demand transaction explanation")
                st.caption("SHAP is calculated only after you request it. Contributions describe model output relative to a background sample, not causes of returns.")
                row_index = st.number_input("Transaction row to explain", min_value=0, max_value=max(0, len(frame) - 1), value=0, step=1)
                if st.button("Generate SHAP explanation", icon=":material/psychology:"):
                    try:
                        row = frame.iloc[[int(row_index)]].reindex(columns=artifact["feature_columns"])
                        background = frame.reindex(columns=artifact["feature_columns"]).drop(index=int(row_index), errors="ignore").head(50)
                        local = local_shap_explanation(artifact["model"], background, row)
                        st.dataframe(
                            local,
                            hide_index=True,
                            alt="SHAP contributions for the selected transaction relative to the displayed background sample",
                        )
                        chart = local.head(12).sort_values("contribution")
                        st.bar_chart(chart, x="feature", y="contribution", alt="SHAP contribution by feature for the selected transaction")
                    except (RuntimeError, ValueError, ImportError) as exc:
                        st.info(f"A SHAP explanation could not be generated: {exc}. The saved permutation analysis remains available above.")
        except (ValueError, OSError, pd.errors.ParserError, KeyError) as exc:
            st.error(f"Scoring failed: {exc}")
