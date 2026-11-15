# ReturnWise Final Experiment Log

Date: 2026-11-15. Scope: data and modeling work only. No Streamlit, UI, dashboard, or application changes were made in this finalization turn.

## Completed and reused evidence

| Work | Method / actual result | Status |
|---|---|---|
| Dataset integrity and target | Full local CSV verification; 200,000 training rows, 50,000 unlabeled test rows; target prevalence 47.4595%; no missing values or exact duplicate rows. | Reused from validated acquisition and audit. |
| Numeric anomalies and sentinel screen | Existing anomaly rates retained in `FINAL_DATA_QUALITY.md`; new streaming check of common exact sentinel constants across all 200,000 rows. | Complete; raw data unchanged. |
| Feature timing | Feature matrix separates plausible candidates, unverified snapshots, post-event fields, identifier, and label. | Complete as an evidence/limitation audit; no verified strict pre-shipment feature set. |
| Feature interactions | Matched 3-fold CV, raw vs three deterministic interactions; differences small and mixed across LR, HGB, LightGBM. | Complete; raw default retained. |
| Feature selection | Fold-local L1 `SelectFromModel`; ROC-AUC 0.583493 and PR-AUC 0.548963 vs no-selection 0.585437 and 0.551053. | Complete; selection rejected. |
| Corrected model comparison | Raw-feature LR, HGB, LightGBM on 3 stratified folds; best mean ROC-AUC 0.586576 (LightGBM), fold SD 0.003231. | Complete, exploratory CV only. |
| OOF ensemble | Raw OOF blends; best ROC-AUC 0.587648 (LR + HGB), best PR-AUC 0.553849 (LR + LightGBM). | Complete, no repeated-CV evidence. |
| Optuna pilot | LightGBM, persistent SQLite, TPE, 3-fold outer CV, two completed trials; scores 0.586433 and 0.587598; about 20.36 seconds. | Smoke test only, not optimal tuning. |
| Historical calibration and decision analysis | Reused original CV calibration, 40,000-row holdout scores, separate development decision-study holdout, and simulated cost scenarios. | Complete historically; not retuned for corrected candidates. |

## Not run in this finalization

- No additional CV, repeated CV, holdout scoring, or tuning. The finalization preflight measured 0.463 GiB RAM and stopped before scientific-library imports or data load; a subsequent check measured 0.414 GiB RAM and 0.239 GiB free disk. Seven Python processes were idle at 0% CPU, with about 0.053 GiB combined working set.
- No model refit on all 200,000 rows, no final artifact/metadata/feature/preprocessing files, no test prediction file, and no new submission. `scripts/finalize_model_submission.py` is prepared with a 0.55 GiB minimum reserve and exact registry-parameter checks; it exited safely before any model fit or output write. The only existing model is the 98,250-byte historical artifact; it was not copied or renamed.
- No new holdout split. The original 40,000-row holdout has already been scored for the historical model and is not reused to select or report a corrected candidate.
- No calibration of the corrected LightGBM or Logistic Regression candidate. Existing beta/sigmoid/isotonic comparisons belong to the historical soft vote.
- No source correction, clipping, zero replacement, or deletion for anomalous values. Their meaning requires the dataset owner's definition.
- No additional dependencies installed. Optuna 4.9.0 and LightGBM 4.7.0 were installed during the prior authorized stage; core scientific-library versions were not upgraded. `pip check` reported an unrelated existing `scikit-image` / `networkx` version mismatch.
- No full integration test rerun in this turn. The last recorded suite result is 6 passed in 34.88 seconds after the main model changes; the later small SQLite URL-rendering correction was exercised by a successful persistent Optuna pilot.

## Decision log

1. Preserve raw rows and use the raw 11-field expanded candidate for modeling; discard feature interactions as a default because CV effects were inconsistent.
2. Keep all 11 candidate input fields; do not use L1 selection because metrics worsened.
3. Exclude `returned` from predictors, `order_id` as an identifier, `delivery_delay_days` as post-event, and `past_purchase_count` / `past_return_rate` until their as-of cutoffs are verified. Keep `product_rating` in the expanded candidate only with an explicit snapshot caveat.
4. Treat default raw LightGBM as the CV-leading candidate and Logistic Regression as the simpler close baseline; do not declare either a deployment-final winner without a fresh, valid finalization process.
5. Do not claim pre-shipment leakage safety, test-set ROC-AUC, business savings, prevented returns, global novelty, or an Optuna optimum.
6. Stop model fitting because the actual preflight fell below the configured RAM threshold; keep the point-in-time caveat explicit.

## Reproducibility pointers

Dataset details: [FINAL_DATA_QUALITY.md](FINAL_DATA_QUALITY.md). Evaluation: [FINAL_MODEL_EVALUATION.md](FINAL_MODEL_EVALUATION.md). Full prior experiment notes remain in [advanced_experiment_results.md](advanced_experiment_results.md), [hyperparameter_optimization.md](hyperparameter_optimization.md), and [deep_research_audit.md](deep_research_audit.md). The existing historical artifact is `artifacts/returnwise_model.joblib` with `artifacts/metadata.json`; it was not overwritten.
