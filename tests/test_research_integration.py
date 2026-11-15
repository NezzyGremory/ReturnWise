from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from returnwise.decision import evaluate_review_strategies, priority_queue
from returnwise.export import make_competition_submission, predict_with_artifact
from returnwise.research_state import load_state
from returnwise.storage import load_model_artifact


ROOT = Path(__file__).resolve().parents[1]


def _research_payload():
    payload = load_model_artifact()
    state = load_state()
    if payload is None or "X_final" not in state:
        pytest.skip("A completed local research run and exported artifact are required for integration checks.")
    return payload, state


def test_exported_model_reloads_with_holdout_prediction_parity():
    payload, state = _research_payload()
    features = state["X_final"].head(128)
    raw, calibrated = predict_with_artifact(payload, features)

    assert raw.shape == calibrated.shape == (len(features),)
    assert np.isfinite(raw).all() and np.isfinite(calibrated).all()
    assert ((raw >= 0) & (raw <= 1)).all()
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    np.testing.assert_allclose(raw, np.asarray(state["final_raw_probabilities"])[: len(features)], rtol=0, atol=1e-12)
    np.testing.assert_allclose(calibrated, np.asarray(state["final_probabilities"])[: len(features)], rtol=0, atol=1e-12)
    assert payload["metadata"]["dataset_fingerprint"] == state["dataset_fingerprint"]


def test_actual_model_scores_feed_ordered_capacity_queue():
    payload, state = _research_payload()
    features = state["X_final"].head(256)
    labels = state["y_final"].head(256)
    raw, calibrated = predict_with_artifact(payload, features)

    queue = priority_queue(features, raw, calibrated)
    decisions = evaluate_review_strategies(raw, y_true=labels, calibrated_probabilities=calibrated)

    assert len(queue) == len(features)
    assert queue["calibrated_probability"].is_monotonic_decreasing
    assert decisions["capacity_percent"].nunique() == 4
    assert set(decisions["capacity_percent"]) == {5.0, 10.0, 20.0, 30.0}
    assert decisions["count_type"].eq("observed on labeled validation data").all()
    assert np.isfinite(decisions["expected_decision_cost"]).all()


def test_all_registered_streamlit_routes_render_without_exceptions():
    routes = [
        "app_pages/01_overview.py",
        "app_pages/02_data.py",
        "app_pages/03_analysis.py",
        "app_pages/04_risk_queue.py",
        "app_pages/05_decision_simulator.py",
        "app_pages/06_research.py",
    ]
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception

    for route in routes:
        app.switch_page(route).run()
        assert not app.exception, f"Streamlit route {route} raised: {[exc.message for exc in app.exception]}"


def test_unlabeled_upload_scores_queue_and_runs_simulator():
    _, state = _research_payload()
    csv_bytes = state["X_final"].head(64).to_csv(index=False).encode("utf-8")
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.switch_page("app_pages/04_risk_queue.py").run()
    app.file_uploader(key="risk_upload").upload("transactions.csv", csv_bytes, "text/csv").run()

    assert not app.exception
    assert any("Scored 64 transactions" in element.value for element in app.success)
    assert len(app.session_state["scored_queue"]) == 64

    app.switch_page("app_pages/05_decision_simulator.py").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 24


def test_competition_submission_uses_sample_schema_without_labels():
    output = ROOT / "data" / "kaggle" / "returnwise_submission.csv"
    sample_path = ROOT / "data" / "kaggle" / "sample_submission.csv"
    test_path = ROOT / "data" / "kaggle" / "test.csv"
    if not output.is_file() or not sample_path.is_file():
        pytest.skip("The local Kaggle submission files are unavailable.")

    submission = pd.read_csv(output)
    sample = pd.read_csv(sample_path)
    competition_test = pd.read_csv(test_path, usecols=["order_id"])
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample) == 50_000
    assert "returned" in submission
    assert submission["ID"].reset_index(drop=True).equals(competition_test["order_id"].reset_index(drop=True))
    assert np.isfinite(submission["returned"]).all()
    assert submission["returned"].between(0, 1).all()


def test_competition_submission_rejects_misaligned_sample_identifiers():
    payload, _ = _research_payload()
    competition_test = pd.read_csv(ROOT / "data" / "kaggle" / "test.csv").head(16)
    sample = pd.read_csv(ROOT / "data" / "kaggle" / "sample_submission.csv").head(16)
    sample.loc[0, "ID"] += 1

    with pytest.raises(ValueError, match="identifiers"):
        make_competition_submission(payload, competition_test, sample)
