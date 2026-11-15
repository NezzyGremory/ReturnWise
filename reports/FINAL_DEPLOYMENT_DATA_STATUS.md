# ReturnWise Final Deployment Data Status

**Status: blocked before refit by the RAM preflight.** The model-selection and artifact-generation path is prepared, but no final model, predictions, or submission were written.

## 1. Intended final candidate

LightGBM was selected as the CV-leading single-model candidate for the requested final refit. The matching raw-feature Quick configuration recorded by the prior 3-fold experiment is `n_estimators=48`, `learning_rate=0.05`, `num_leaves=31`, `class_weight="balanced"`, `n_jobs=1`, and `random_state=42`; remaining estimator defaults are preserved from the existing registry. This is the untuned configuration associated with ROC-AUC 0.5865765 and fold SD 0.0032310, not the separate two-trial Optuna pilot. Features are the ordered 11-field expanded candidate in `FINAL_DATA_QUALITY.md`, with feature engineering and feature selection disabled.

## 2. Data and preprocessing plan

The intended fit uses all 200,000 labeled rows from the validated `train.csv`, target `returned`, and no test labels. Pipeline preprocessing is fitted on training data: numeric median imputation; categorical most-frequent imputation and one-hot encoding (`handle_unknown="ignore"`, `min_frequency=2`). No clipping, anomaly correction, scaling, calibration, or test-fitted transform is applied. Probabilities would be raw LightGBM outputs. The selected feature fields are not verified as point-in-time safe; no pre-shipment safety claim is made.

## 3. CV evidence and evaluation limit

The basis for selecting this candidate is 3-fold stratified CV on 160,000 development rows, seed 42: mean ROC-AUC **0.5865765**, fold SD **0.0032310**. This is a CV result, not a Kaggle test score. The prior 40,000-row holdout was scored for the historical model and is not used as an independent evaluation for the corrected pipeline. No new holdout result is available for this candidate. The historical holdout metrics belong to the old HGB + Logistic Regression artifact, not to a new LightGBM fit.

## 4. Preflight and execution outcome

The one-process finalization script, `scripts/finalize_model_submission.py`, checks for existing outputs, RAM before imports/data loading/training/prediction, exact model parameters, target/schema integrity, test/sample ID order, and serialized-model reload parity. It stages outputs and refuses to overwrite existing final filenames.

At the first execution, RAM measured **0.463 GiB**, below the script's **0.55 GiB** threshold (the runner floor is 0.45 GiB). The script stopped before importing scientific libraries, loading CSVs, fitting, or writing outputs. A later check measured **0.414 GiB RAM** and **0.239 GiB free disk**. Seven Python processes were idle (0% CPU) with about 0.053 GiB combined working set; they were left untouched. No training run was attempted after the failed preflight.

## 5. Artifact and submission locations

The following requested outputs do **not** exist because refit did not start:

- `artifacts/final_model.joblib`
- `artifacts/final_metadata.json`
- `artifacts/final_feature_list.json`
- `artifacts/final_preprocessing_config.json`
- `artifacts/final_test_predictions.csv`
- `artifacts/final_submission.csv`

The existing `artifacts/returnwise_model.joblib` and `artifacts/metadata.json` were preserved and not relabeled as the new LightGBM model. No Kaggle test ROC-AUC is available because `test.csv` has no labels.

## 6. Validation and limitations

No final artifact load test, 50,000-row prediction validation, or template comparison was possible because no new artifact was produced. The guard script contains those validations for a future run when resources are safely above threshold. The current project therefore has an evidence-based LightGBM candidate and exact configuration but not a final refit, test prediction, or submission.
