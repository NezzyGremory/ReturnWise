import streamlit as st

from returnwise.decision import evaluate_review_strategies


st.caption("DECISION SUPPORT / CAPACITY SIMULATION")
st.title("See how review capacity changes the queue")
st.write(
    "Choose a review count to see the highest-ranked transactions and compare decision strategies under explicit cost assumptions. "
    "Expected counts come from model probabilities because the scored file has no observed return labels."
)
queue = st.session_state.get("scored_queue")
if queue is None:
    st.info(
        "Upload and score an unlabeled transaction CSV on the Risk queue page first. No sample predictions are generated here.",
        title="No scored queue",
    )
    st.page_link("app_pages/04_risk_queue.py", label="Open risk queue", icon=":material/low_priority:")
else:
    if "return_probability" not in queue or "calibrated_probability" not in queue:
        st.error("The scored queue does not contain the probability fields required for capacity simulation.")
    else:
        with st.form("scenario"):
            capacity_choice = st.selectbox(
                "Transactions your team can review",
                options=[10, 25, 50, 100, "Custom count"],
                index=1,
                key="sim_review_capacity_choice",
            )
            if capacity_choice == "Custom count":
                capacity_count = int(
                    st.number_input(
                        "Custom review count",
                        min_value=1,
                        max_value=len(queue),
                        value=min(25, len(queue)),
                        step=1,
                        key="sim_review_capacity_custom",
                    )
                )
            else:
                capacity_count = min(int(capacity_choice), len(queue))

            a, b, c = st.columns(3)
            false_positive_cost = a.number_input("False-positive cost assumption", min_value=0.0, value=1.0, step=0.25)
            false_negative_cost = b.number_input("False-negative cost assumption", min_value=0.0, value=5.0, step=0.5)
            review_cost = c.number_input("Review cost assumption", min_value=0.0, value=0.25, step=0.05)
            d, e = st.columns(2)
            threshold = d.slider("Fixed threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            uncertainty = e.slider("Uncertainty flag confidence ceiling", min_value=0.5, max_value=1.0, value=0.6, step=0.01)
            submitted = st.form_submit_button("Run capacity simulation", type="primary")

        if submitted:
            top = queue.head(capacity_count)
            probability = top["calibrated_probability"].astype(float)
            expected_total = float(queue["calibrated_probability"].astype(float).sum())
            expected_selected = float(probability.sum())
            coverage = expected_selected / expected_total if expected_total else 0.0

            st.subheader("Selected review list")
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Transactions in capacity", f"{len(top):,}")
            metric_b.metric("Expected return cases", f"{expected_selected:,.1f}")
            metric_c.metric("Share of estimated cases", f"{coverage:.1%}")
            st.caption("Expected cases are the sum of estimated probabilities, not observed outcomes. Priority rank does not show that review prevents a return.")
            st.download_button(
                "Download selected review list",
                data=top.to_csv(index=False).encode("utf-8"),
                file_name="returnwise_capacity_review.csv",
                mime="text/csv",
                width="content",
            )

            results = evaluate_review_strategies(
                queue["return_probability"],
                calibrated_probabilities=queue["calibrated_probability"],
                costs={"false_positive": false_positive_cost, "false_negative": false_negative_cost, "review": review_cost},
                threshold=threshold,
                uncertainty_threshold=uncertainty,
            )
            st.subheader("Strategy comparison")
            st.caption("The table compares the saved research strategies at 5%, 10%, 20%, and 30% of this queue. Costs are scenario inputs, not measured business costs.")
            st.dataframe(
                results,
                hide_index=True,
                alt="Expected costs and return counts for six strategies across four review-capacity percentages",
            )
            at_ten = results[results["capacity_percent"] == 10]
            if not at_ten.empty:
                st.bar_chart(
                    at_ten,
                    x="strategy",
                    y="expected_decision_cost",
                    alt="Expected scenario cost at 10 percent review capacity",
                )
