import streamlit as st

from returnwise.decision import evaluate_review_strategies

st.title("Decision cost simulator")
st.write("Compare review strategies under explicit scenario assumptions. Costs are simulated inputs; they are not measured business costs or evidence of profit impact.")
queue = st.session_state.get("scored_queue")
if queue is None:
    st.info("Score an unlabeled transaction CSV from the Risk queue page first. Expected-cost outputs require model probabilities.")
else:
    with st.form("scenario"):
        a, b, c = st.columns(3)
        false_positive_cost = a.number_input("False-positive cost", min_value=0.0, value=1.0, step=0.25)
        false_negative_cost = b.number_input("False-negative cost", min_value=0.0, value=5.0, step=0.5)
        review_cost = c.number_input("Review cost per transaction", min_value=0.0, value=0.25, step=0.05)
        d, e = st.columns(2)
        threshold = d.slider("Fixed threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        uncertainty = e.slider("Uncertainty flag confidence ceiling", min_value=0.5, max_value=1.0, value=0.6, step=0.01)
        submitted = st.form_submit_button("Evaluate scenarios", type="primary")
    if submitted:
        data = queue
        if "return_probability" not in data or "calibrated_probability" not in data:
            st.error("The scored queue does not contain the probability columns required for simulation.")
        else:
            results = evaluate_review_strategies(
                data["return_probability"],
                calibrated_probabilities=data["calibrated_probability"],
                costs={"false_positive": false_positive_cost, "false_negative": false_negative_cost, "review": review_cost},
                threshold=threshold,
                uncertainty_threshold=uncertainty,
            )
            st.dataframe(results, hide_index=True, alt="Expected review cost, precision-at-capacity, recall-at-capacity, and expected return cases by strategy")
            top = results[results["capacity_percent"] == 10]
            if not top.empty:
                st.bar_chart(top, x="strategy", y="expected_decision_cost", alt="Expected decision cost at 10 percent review capacity under current simulated cost assumptions")
            st.caption("Counts are expectations from model probabilities because this operational CSV has no labels. For observed false-positive and false-negative counts, use an untouched labeled validation holdout in the notebooks.")
