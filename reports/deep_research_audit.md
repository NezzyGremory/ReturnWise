# ReturnWise Deep Research Audit

Audit date: 2026-11-15  
Scope: repository source, notebooks 00-20, saved experiment state and history, dataset cache metadata, model artifact and metadata, reports, tests, and environment manifest.

## Executive summary

The saved Quick Mode run is internally reproducible from the current local state and dataset fingerprint. The competition CSVs pass the recorded full-file schema and checksum checks; the binary target is present and consistent; the exported model metadata matches the saved research state. No direct target column or delivery outcome is included in the selected 11-column model schema.

The current artifact is best described as a **conditional order-time candidate**. The CSVs have no event timestamp, feature capture timestamp, or historical aggregate cutoff. The non-post-event fields therefore cannot be certified as available at the exact pre-shipment decision point. The data also contain sizeable numbers of physically questionable negative values whose semantics are undocumented. These limitations prevent a verified operational or temporal-performance claim.

The previous run completed only in Quick Mode. The follow-up added a corrected raw-versus-engineered comparison, fold-local L1 selection, raw-feature OOF blends, and a persistent two-trial LightGBM/Optuna pilot. The Optuna run is a smoke pilot, not meaningful optimization. SHAP, several model families, repeated CV, confidence intervals, class-imbalance alternatives, final deployment fitting, and application artifact integration remain incomplete. No comprehensive search result is claimed.

## 1. Project and evidence inventory

Reviewed:

- `README.md`, `docs/implementation_plan.md`, `docs/references.md`, and `reports/returnwise_research_report.md`.
- All Python modules under `returnwise/`, including data acquisition/profiling, feature timing, preprocessing, modeling, calibration, decisions, explanation, export, storage, and reporting.
- All 21 notebooks, `notebooks/00_environment_setup.ipynb` through `notebooks/20_returnwise_model_export.ipynb`. Every notebook has two executed code cells and saved output; the markdown cell is not executed.
- `experiments/research_state.joblib`, all 12 lines in `experiments/history.jsonl`, `artifacts/returnwise_model.joblib`, `artifacts/metadata.json`, the data metadata, `requirements.txt`, `scripts/run_research_pipeline.py`, six Streamlit page registrations, and `tests/test_research_integration.py`.

The working tree already contained uncommitted edits to README, the implementation plan, research notebooks 18-20, the report generator, the EDA profile, the research page, the research runner, a report, and tests. Those edits were preserved. This follow-up also changed the model pipeline and submission identifier validation, added regression coverage, installed the already-declared optional Optuna and LightGBM dependencies, and added audit/experiment documentation and dataset figures. The original exported artifact and raw CSVs were not replaced.

## 2. Dataset identity and integrity

- Dataset fingerprint in the local acquisition metadata, research state, and model metadata: `2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56`.
- `train.csv`: 200,000 rows, 16 columns including `returned`.
- `test.csv`: 50,000 rows, 15 feature columns and no label.
- `sample_submission.csv`: 50,000 rows, columns `ID` and `returned`.
- Train and test feature schemas match in name and order after removing the training target. The target contains only 0 and 1.
- Local SHA-256 hashes and full-file row counts are recorded in `data/kaggle/dataset_metadata.json`. Kaggle's current file listing byte sizes match. The local ZIP is an incomplete 1 MiB `BadZipFile` with a partial marker; it was not extracted. The three CSVs were validated independently, so the archive problem does not invalidate the local CSV cache.

The official [Kaggle data page](https://www.kaggle.com/competitions/retail-return-risk-modeling/data) describes the listed customer, product, session, delivery, and transaction fields. The [competition overview](https://www.kaggle.com/competitions/retail-return-risk-modeling/overview/citation) describes transaction records from 2022-2024, but no timestamp column exists in the supplied CSV schema. A date range in the competition description does not create a usable temporal split.

## 3. Implementation and experiment audit

### Data and feature pipeline

`returnwise.data.kaggle` validates required files, target values, train/test schemas, local file hashes, and archive extraction paths. The state fingerprint is consistent with the acquisition metadata. The training target is defined as the observed binary `returned` column.

`returnwise.features.temporal_feature_audit` uses column names and competition descriptions to classify feature timing. It correctly excludes `returned`, `order_id`, `delivery_delay_days`, `past_purchase_count`, and `past_return_rate` from the current pre-shipment candidate set. The 11 included fields are treated as likely order-time inputs, not verified point-in-time records. The audit is heuristic and does not inspect source-system timestamps.

Preprocessing is inside sklearn pipelines. Numeric fields use median imputation; categorical fields use most-frequent imputation and one-hot encoding for the linear, forest, and LightGBM candidates. The deterministic feature transformer creates price x discount, views per session minute, and price x rating, then converts infinite values to missing. It does not determine whether questionable negative source values are valid.

**Implementation defect found during this audit:** the generic pipeline originally built its `ColumnTransformer` from the raw 11-column frame and then added the interactions in a preceding step. Because the column transformer selects named columns and drops the remainder, it silently discarded the three new columns for Logistic Regression, Random Forest, Extra Trees, LightGBM, and XGBoost. HistGradientBoosting separately discovered its columns from the engineered frame and did receive them; CatBoost's cleaner also returned the engineered frame. The saved soft-voting model therefore blended Logistic Regression on raw inputs with HistGradientBoosting on engineered inputs. The prior logistic/forest/LightGBM metrics are raw-feature results, not feature-engineered results. The LightGBM pilot logged during this continuation was also measured before this fix and is a raw-feature baseline.

The generic pipeline has now been corrected to build preprocessing from the same engineered schema it receives at fit/predict time. Raw-versus-engineered CV has been run with the explicit switch. Its effects are small and inconsistent, so raw inputs remain the default. The selected exported artifact has not been retrained and still represents the original mixed raw/engineered soft vote. HistGradientBoosting currently uses ordinal codes for nominal categories, which imposes an arbitrary order; one-hot or native-categorical comparison remains a separate opportunity. The earlier saved benchmark rows and holdout metrics predate this correction and must remain labeled as historical results.

### Splitting, model selection, and calibration

- Modeling uses stratified random splits with seed 42; no time-ordered split is available. No temporal validation is claimed.
- The saved state records 160,000 development rows and a 40,000-row internal holdout. Candidate comparisons used three-fold stratified CV in Quick Mode.
- Calibration CV reserves an inner calibration subset within each outer development fold. The final holdout was not used to choose the model or calibration method.
- Notebook 19 fits its final evaluation model on 120,000 rows, fits the selected calibrator on a separate 40,000-row calibration subset, and scores the untouched 40,000-row holdout. Notebook 20 exports that same evaluation model. Thus the exported model is valid for reproducing the reported holdout result, but it does not use all 200,000 labeled rows for deployment fitting. A deployment refit after model selection remains to be designed and documented.
- The selected candidate is `Soft voting: HistGradientBoosting + Logistic regression`, with beta calibration. Holdout metrics are ROC-AUC 0.589481, average precision 0.556060, Brier 0.243089, log loss 0.679141, and ECE 0.001805. Constant-prevalence Brier and log loss are 0.249355 and 0.691856. Discrimination is modest; the random holdout does not establish future-period performance.

Saved three-fold CV discrimination and probability scores for the selected candidates:

| Candidate | ROC-AUC | PR-AUC | ROC-AUC fold SD | PR-AUC fold SD | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Soft voting: HGB + Logistic Regression | 0.587039 | 0.553303 | 0.004070 | 0.002602 | 0.243406 | 0.679781 |
| Logistic Regression | 0.585251 | 0.550937 | 0.003710 | 0.002145 | 0.243699 | 0.680391 |
| HistGradientBoosting | 0.585704 | 0.551442 | 0.003985 | 0.002219 | 0.243738 | 0.680466 |
| Extra Trees | 0.563745 | 0.530606 | 0.000440 | 0.000245 | 0.246107 | 0.685297 |

The original saved OOF blend combines HGB with Logistic Regression and predates the pipeline correction. A new raw-feature OOF comparison is recorded below; its best blend has ROC-AUC 0.587648 and PR-AUC 0.553586 for LR + HGB. Gains are small and have no repeated-CV or confidence-interval evidence.

### Coverage of requested methods

| Area | What the saved run did | Audit conclusion |
|---|---|---|
| Baselines | Dummy prior and Logistic Regression | Completed in 3-fold Quick Mode. |
| Tree models | Random Forest, Extra Trees, HistGradientBoosting | Completed with small Quick Mode budgets; HGB used 48 iterations, forests used 32 trees. |
| Other requested families | Decision Tree, Gradient Boosting, linear/RBF SVM, CatBoost, LightGBM, XGBoost | Not evaluated. Optional boosting libraries were absent; Decision Tree and SVM are not in the current registry. |
| Imbalance | Logistic Regression and forest candidates use class weights | No paired comparison against unweighted models, random oversampling, or SMOTE. The target is near balanced at 47.46% positive. |
| Feature engineering | After fixing a pipeline schema bug, raw vs three interactions was compared for Logistic Regression, HGB, and LightGBM on the same three folds | Changes are small and mixed; raw fields remain the default. See `feature_engineering_analysis.md`. |
| Feature selection | No selection vs fold-local L1 `SelectFromModel` on Logistic Regression | L1 reduced ROC-AUC and PR-AUC and increased variability; no selection is retained. Other selectors were not run because there are only 11 source fields and L1 did not help. See `feature_selection_analysis.md`. |
| Hyperparameter search | Persistent Optuna TPE study with median pruning, SQLite storage, and two LightGBM trials | Pilot only: best 3-fold ROC-AUC 0.587598 across two completed trials; no pruned or failed trials. This is not enough to claim a tuned optimum. |
| Ensemble | New raw-feature OOF equal-weight blends of LR, HGB, and LightGBM | LR + HGB scored ROC-AUC 0.587648 / PR-AUC 0.553586; no stacking or weight search. Gains are small and unconfirmed. |
| Calibration | Uncalibrated, sigmoid, isotonic, and beta in nested CV | Beta was selected using training-side CV and fitted on a separate calibration partition for final scoring. No beta-calibration dependency is needed because it is implemented locally. |
| Explainability | Permutation importance | SHAP was not attempted; permutation importance is a model-reliance measure, not a causal effect. Importance stability across folds was not measured. |
| Decision analysis | Random, probability, fixed threshold, cost-sensitive, calibration-aware, and uncertainty-aware strategies at 5-30% capacity | Costs are hypothetical. Lift@K and net benefit are not reported. Fixed costs make the expected cost-sensitive score a monotone transform of probability, so identical ranking is expected. Review effectiveness is not empirically established. |
| Robustness and uncertainty | Three-fold variability and a single stratified holdout | No repeated CV, bootstrap confidence intervals, paired model intervals, seed sensitivity, or fold sensitivity. |

The saved code uses `StratifiedKFold`, pipelines, seeded estimators, and OOF probabilities. `tune_with_optuna` now supports a persistent SQLite study with TPE, median pruning, incremental trial budgets, and fold-local early stopping. `experiments/optuna_studies.db` contains two completed LightGBM trials (best 3-fold ROC-AUC 0.587598); no failed or pruned trials occurred. The two-trial result is a smoke test only. Calibration reports ECE using ten equal-width bins and a quantile reliability curve, but no calibration slope/intercept or bootstrap interval is recorded. The beta optimizer accepts a finite fitted result even when its optimizer status is unsuccessful; this failure-handling path has not been observed in the saved run and should be tightened before expanded calibration experiments.

### Calibration and decision implementation

`ProbabilityCalibrator` implements sigmoid, isotonic, and beta calibration. The final beta map is monotonic under its nonnegative coefficient constraints, so it preserves ranking. The decision cost score for constant false-positive, false-negative, and review costs is `p * FN - (review + (1-p) * FP)`, which is strictly increasing in `p` when costs are nonnegative. Under fixed review capacity, cost-sensitive ranking therefore cannot improve on probability ranking solely by changing this score; any claim must come from a different decision rule or empirically supported nonconstant assumptions.

On labeled validation data, the simulator counts unreviewed returns as false negatives and reviewed non-returns as false positives. This encodes a perfect review-identification assumption for selected return cases. No measured intervention effectiveness or business cost is available, so expected cost is a scenario calculation, not realized savings or evidence that reviews prevent returns.

### Export and application

The original artifact schema, selected model, selected calibrator, dataset fingerprint, and holdout metrics agree with the original exported research run. `predict_with_artifact` reindexes to the exported feature schema; model preprocessing is included in the serialized estimator. The risk queue and decision simulator load that artifact, and `streamlit_app.py` registers six pages. The artifact was not retrained after the model-pipeline correction or new OOF comparisons, so those newer results are not deployed in the app.

`make_competition_submission` checks row count, matching identifiers when column names match, output schema, and probability output. Here `sample_submission.csv` calls the identifier `ID` while `test.csv` calls it `order_id`; the original helper did not compare those differently named columns. The current files do match exactly in row order (50,000 aligned identifiers). This audit added the cross-name comparison and an integration regression check so future misalignment fails explicitly. The test file remains unlabeled and is not used for supervised evaluation.

The integration suite contains six checks. It passed after the pipeline, feature-selection, and tuning implementation changes: **6 passed in 34.88 seconds**. This run preceded a small Optuna SQLite URL-rendering correction; the subsequent two-trial persistent pilot exercised that correction successfully. The suite was not rerun after that small change because the resource stop threshold was reached. Coverage includes app/artifact/queue/submission behavior and explicit rejection of mismatched sample/test identifiers.

## 4. Discrepancies and documentation risks

- `docs/implementation_plan.md` contains earlier statements that the workspace was empty, Notebook 02 had not run, research had not been executed, and tests were unavailable. Later sections report the completed run and tests. The newer status explains the sequence, but the full document should be consolidated so historical snapshots cannot be mistaken for current status.
- The current research report accurately marks performance and cost assumptions as limited, but it is a generated run summary rather than a deep dataset audit. The two audit reports added in this task cover the missing dataset and method review.
- `requirements.txt` declares optional boosting, tuning, and SHAP packages. Optuna 4.9.0 and LightGBM 4.7.0 are now installed in the existing Python 3.12.10 environment; CatBoost, XGBoost, SHAP, and Kaggle Python remain absent. The CSV cache permits local analysis without Kaggle CLI.
- The declared version ranges accept the observed core versions (NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1, Streamlit 1.66.1). No dependency upgrade was made.
- The package resolver selected Optuna 4.9.0 and LightGBM 4.7.0 without proposing core dependency upgrades. Both packages installed in the existing `.venv` and imported successfully. `pip check` reported an unrelated existing mismatch: `scikit-image 0.26.0` requires `networkx>=3.0`, while `networkx 2.8.8` is present; ReturnWise does not use either package, and neither was changed.
- A sequential, three-fold Quick Mode LightGBM pilot completed on 160,000 development rows before the feature-schema fix: ROC-AUC 0.586576 (fold SD 0.003231), PR-AUC 0.553064 (fold SD 0.002638), mean fit time 2.04 seconds per fold. This is a raw-feature result and must not be interpreted as the corrected engineered pipeline. Its per-fold outputs and run configuration were checkpointed in `research_state.joblib` and `history.jsonl`.
- After the completed two-trial Optuna pilot, observed headroom fell to approximately 0.471 GiB available RAM and 0.348 GiB free disk, close to the notebook runner's 0.45 GiB RAM floor. Further model fitting, tuning, calibration, or deployment refit was stopped under the prompt's resource rule. The two-trial pilot and preceding comparisons remain saved; larger research budgets are incomplete.

## 5. Leakage and feature availability decision

No direct target or post-delivery column is in the exported model schema. This removes the obvious direct leakage paths. It does not prove that every remaining value was captured before the prediction time. See the per-feature matrix and scenario definitions in [`dataset_deep_analysis.md`](dataset_deep_analysis.md).

The available columns support three distinct research views, but not three comparable operational claims:

1. **Strict verified pre-shipment:** no column has a source timestamp or immutable as-of guarantee in the CSV, so this dataset alone cannot certify a strictly point-in-time feature set.
2. **Expanded order-time candidate:** the 11 current model fields are plausible from their descriptions, subject to provenance and snapshot verification.
3. **Diagnostic all-available-fields:** add delivery delay and historical aggregates only as sensitivity analysis. Keep `returned` out and keep `order_id` excluded as an identifier. Do not present this scenario as pre-shipment.

The official competition page defines `delivery_delay_days` using actual delivery and identifies the customer-history fields as historical aggregates, but it does not publish the cutoff construction needed to establish leakage safety. The CSV has no timestamps with which to reconstruct that cutoff.

## 6. Priority actions

1. Obtain the feature-generation rules and as-of timestamps for product rating, customer aggregates, session fields, and order-time fields. If they are unavailable, retain the current candidate-only wording and do not call the artifact operationally validated.
2. Ask the dataset owner to define the negative price, session duration, view count, out-of-range discount/rating, and invalid historical return-rate values. Preserve raw data until meaning is known; evaluate explicit invalid-value indicators and robust alternatives as separate pipelines.
3. Complete a separate strict/expanded/diagnostic feature-scenario evaluation with identical split design, clearly labeled scenario outputs, and no temporal claim unless timestamps are obtained.
4. Resolve source semantics before testing additional price transforms or invalid-value handling. If feature count expands, compare further fold-local selectors; consider a nominal-category encoding sensitivity for HistGradientBoosting if resources permit.
5. Expand benchmark coverage only after the resource estimate, then tune promising candidates using training folds only and checkpoint trial state.
6. Add uncertainty estimates and decision metrics (including Lift@K) with clearly specified review-effect assumptions and cost sensitivity. Do not claim cost savings or intervention impact.
7. After selection and once-only holdout reporting, fit a distinct deployment model using the chosen data/calibration strategy; record that it differs from the evaluation estimator.
8. Reconcile the historical status blocks in `docs/implementation_plan.md` after the deep research sequence is complete.

## 7. Audit boundary

This audit establishes repository and dataset facts from local code, outputs, metadata, and the official competition description. It does not establish causation, production data availability, business cost, review effectiveness, temporal generalization, global novelty, or private leaderboard performance. No raw dataset, prior notebook output, saved model, or existing research result was modified by this audit.
