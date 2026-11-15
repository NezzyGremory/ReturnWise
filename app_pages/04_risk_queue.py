import streamlit as st
import pandas as pd

from returnwise.data.profile import quality_report
from returnwise.decision import priority_queue
from returnwise.export import predict_with_artifact
from returnwise.explain import local_shap_explanation
from returnwise.storage import active_dataset_fingerprint, load_model_artifact

st.title("Return risk queue")
artifact = load_model_artifact()
if not artifact:
    st.info("No exported model artifact is available. Run the research notebooks on real labeled data, select a model using validation results, then export it from notebook 20.", title="Prediction unavailable")
else:
    metadata = artifact.get("metadata", {})
    st.caption(f"Model: {metadata.get('selected_model', 'unknown')} · Scenario: {metadata.get('prediction_scenario', 'unspecified')} · Exported: {metadata.get('exported_at', 'unknown')}")
    current_fingerprint = active_dataset_fingerprint()
    model_fingerprint = metadata.get("dataset_fingerprint")
    stale_model = bool(current_fingerprint and model_fingerprint and current_fingerprint != model_fingerprint)
    if stale_model:
        st.error("The active dataset fingerprint differs from the dataset used to train this model. Rerun the research sequence and export a matching artifact before scoring.")
    st.warning("Upload unlabeled transactions for operational scoring. Labeled training data is not accepted here as an operational queue.")
    uploaded = st.file_uploader("Transactions to score", type=["csv"], key="risk_upload", disabled=stale_model)
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            if frame.empty:
                st.error("The uploaded CSV has no rows.")
            elif "returned" in frame.columns:
                st.error("Remove the `returned` target before operational scoring. Use a held-out labeled split in the research notebooks for evaluation.")
            else:
                missing = [c for c in artifact["feature_columns"] if c not in frame.columns]
                if missing:
                    st.warning("Missing model inputs will be passed through the trained imputer: " + ", ".join(missing))
                raw, calibrated = predict_with_artifact(artifact, frame)
                queue = priority_queue(frame, raw, calibrated)
                st.session_state["scored_queue"] = queue
                st.session_state["scored_model_metadata"] = metadata
                st.success(f"Scored {len(queue):,} transactions using the exported model.")
                search = st.text_input("Search displayed values", key="risk_search")
                view = queue.copy()
                if search:
                    mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                    view = view[mask]
                st.dataframe(view.head(1000), hide_index=True, height=520, alt="Transactions ordered by calibrated return probability with uncertainty flags and risk bands")
                st.download_button("Download full risk queue", data=queue.to_csv(index=False).encode("utf-8"), file_name="returnwise_risk_queue.csv", mime="text/csv", width="content")
                st.caption("Predictions are estimates, not guarantees. Risk bands and uncertainty flags describe model outputs, not business outcomes.")
                with st.expander("Prediction reliability and model status"):
                    validation = metadata.get("validation_metrics", {})
                    key_metrics = {k: validation.get(k) for k in ["roc_auc", "pr_auc", "brier_score", "log_loss", "expected_calibration_error"] if validation.get(k) is not None}
                    if key_metrics:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Validation ROC-AUC", f"{key_metrics['roc_auc']:.3f}" if key_metrics.get('roc_auc') is not None else "—")
                        m2.metric("Validation PR-AUC", f"{key_metrics['pr_auc']:.3f}" if key_metrics.get('pr_auc') is not None else "—")
                        m3.metric("Brier score", f"{key_metrics['brier_score']:.3f}" if key_metrics.get('brier_score') is not None else "—")
                    reliability = validation.get("reliability_curve")
                    if reliability:
                        reliability_frame = pd.DataFrame(reliability)
                        st.line_chart(reliability_frame, x="mean_predicted_probability", y="observed_return_rate", alt="Observed return rate by predicted-probability bin on the internal final holdout")
                    else:
                        st.json(validation)
                    if not metadata.get("probability_is_calibrated", False):
                        st.warning("The exported model metadata does not confirm a selected calibrator. Treat its probabilities as uncalibrated scores.")
                importance = artifact.get("permutation_importance")
                if importance:
                    st.subheader("Global risk factors")
                    st.dataframe(pd.DataFrame(importance), hide_index=True, alt="Permutation feature importance measured on validation data")
                else:
                    st.info("No validation-set permutation importance was saved with this model artifact; feature attribution is unavailable.")
                st.subheader("Local risk explanation")
                row_index = st.number_input("Transaction row to explain", min_value=0, max_value=max(0, len(frame) - 1), value=0, step=1)
                if st.button("Explain selected transaction", icon=":material/psychology:"):
                    try:
                        row = frame.iloc[[int(row_index)]].reindex(columns=artifact["feature_columns"])
                        background = frame.reindex(columns=artifact["feature_columns"]).drop(index=int(row_index), errors="ignore").head(50)
                        local = local_shap_explanation(artifact["model"], background, row)
                        st.dataframe(local, hide_index=True, alt="Local SHAP contributions for the selected transaction; positive contributions raise the model return score relative to the background")
                        chart = local.head(12).sort_values("contribution")
                        st.bar_chart(chart, x="feature", y="contribution", alt="Local feature contribution to the positive-class model score for the selected transaction")
                        st.caption("SHAP values explain model output relative to the selected background. They do not establish causality.")
                    except (RuntimeError, ValueError, ImportError) as exc:
                        st.info(f"A local SHAP explanation could not be generated: {exc}. The saved validation-set permutation importance remains the model-agnostic alternative.")
        except (ValueError, OSError, pd.errors.ParserError, KeyError) as exc:
            st.error(f"Scoring failed: {exc}")
