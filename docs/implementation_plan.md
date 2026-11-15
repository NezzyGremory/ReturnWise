# Workspace audit and implementation record

## Current verified run — 2026-11-15

All notebooks 00–20 completed sequentially in the existing `.venv` using Quick Mode. The runner verified the three local CSVs against Notebook 01 metadata and SHA-256 hashes before research execution. Notebook 14 first stopped on an ROC-AUC scorer shape mismatch; the explanation helper was corrected to score the positive-class probability explicitly, then Notebook 14 and later notebooks completed. A temporary low-disk guard stop after Notebook 17 cleared after the kernel exited; no project files were removed, and the runner resumed from saved checkpoints.

The ZIP remains an incomplete 1 MiB `BadZipFile` with a resume marker. It was not extracted or modified. The complete train/test/sample-submission CSV files were independently verified and reused.

### Recorded outcomes

- Dataset: 200,000 labeled training rows, 50,000 unlabeled competition-test rows, 16 training columns, 15 test columns; train/test feature schemas match after excluding `returned`.
- Training SHA-256: `1d44901b7a1f1aaacb5a899909c31b68cb999106d789fd8153552e0e9107025b`; test SHA-256: `261d13cc3bc3f7f4baf4b39113a02787043739439a0234573be08df4685d90cf`; sample-submission SHA-256: `9b2340d344297d4d047542db825d612cbf0325b660ae300679f6742d8432d733`.
- Training target prevalence was 47.4595%. The data-quality scan found no missing values or exact duplicate rows. It flagged negative prices (27,285 rows), negative session duration (1,487), negative product views (21,443), discounts outside 0–100 (1,686), and ratings outside the common 0–5 range (3,831). Raw rows were preserved; these questionable values were not silently clipped or removed.
- The original pre-shipment modeling scenario used 11 plausible order-time fields. The later dataset audit found that their point-in-time capture is not verified from the CSV metadata. `order_id`, `delivery_delay_days`, and historical aggregates `past_purchase_count` and `past_return_rate` were excluded. No temporal validation is claimed.
- The selected model is `Soft voting: HistGradientBoosting + Logistic regression`; selected calibration is beta. Final internal holdout (40,000 stratified rows): ROC-AUC 0.589481, average precision 0.556060, Brier 0.243089, log loss 0.679141, ECE 0.001805. Constant-prevalence references: Brier 0.249355 and log loss 0.691856. Discrimination is modest and this random split is not a future-period guarantee.
- Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting completed Quick Mode comparisons; Random Forest and Extra Trees used 32 trees, HistGradientBoosting used 48 boosting iterations, and the selected soft-voting blend averaged its member probabilities. Native libraries were restricted to one thread. The blend was evaluated from out-of-fold probabilities. CatBoost, LightGBM, XGBoost, TabPFN, and Optuna tuning were skipped because their optional packages were absent. No tuned optimum is claimed.
- Permutation importance completed after the positive-class scorer fix. SHAP was not installed. Cost-sensitive ranking matched probability ranking under constant costs; the saved 10% ablation did not show benefit from uncertainty-aware prioritization. Costs remain simulation assumptions.
- Notebook 20 exported `artifacts/returnwise_model.joblib` and metadata, reloaded the artifact, and verified prediction equality on the holdout. It also created the competition submission from unlabeled test rows; this file is not used for labeled evaluation.
- The research report is `reports/returnwise_research_report.md`. The application smoke test and automated suite are recorded after execution below.

### Follow-up audit and resource-bounded experiments (2026-11-15)

The deep audit added a dataset feature-timing/anomaly analysis, corrected the generic model pipeline so preprocessing sees the engineered schema, and changed new benchmark defaults to raw features after matched CV showed no consistent interaction benefit. Fold-local L1 selection reduced both ROC-AUC and PR-AUC. New raw-feature three-fold comparisons and OOF blends are exploratory; they do not replace or update the original holdout result or exported artifact. LightGBM 4.7.0 and Optuna 4.9.0 were installed without changing core dependencies. A persistent two-trial Optuna pilot completed (best ROC-AUC 0.587598), which is only a smoke test, not a reliable optimum.

The current saved artifact remains the historical soft vote and has not been retrained or re-exported after the pipeline correction. Available RAM after the pilot was about 0.471 GiB, near the runner's 0.45 GiB safety floor; further fitting and tuning stopped under the research prompt's resource rule. The remaining requested final-model refit, corrected holdout evaluation, deployment export, and app integration are incomplete. The full twelve-file research documentation is indexed by `reports/final_research_findings.md`.

### Current reproducibility entry point

Use the existing `.venv`, validated Kaggle CSV cache, and kernel `Python (ReturnWise)`. From the project root run `.venv\Scripts\python.exe scripts\run_research_pipeline.py`. The runner executes notebooks serially, saves outputs, checks handoff state and dataset fingerprint, and stops before the next notebook when its RAM or disk thresholds are crossed. It does not download data or install packages.

## Initial audit

The workspace was empty: no application framework, Python modules, notebooks, dependency manifest, database, data files, or Git repository were present. There were no existing features to preserve or replace. The environment provides Python 3.12, Streamlit 1.66, common scientific packages (pandas, NumPy, scikit-learn, SciPy), and a Kaggle CLI executable installed for another Python environment.

The Kaggle CLI executable is present (version 2.2.4), but its default config location `C:\Users\Lenovo\.kaggle` is not writable in this environment. A second check using a temporary writable `KAGGLE_CONFIG_DIR` confirmed that Kaggle authentication is not configured. The CLI instructed the user to run `kaggle auth login`; the competition file listing therefore stopped safely and no dataset download was attempted. The application reports this condition and gives the local-terminal remediation. The official Kaggle competition page describes the requested dataset schema, but the implementation still reads and validates the actual CSVs at acquisition time instead of treating that page as a substitute for CSV inspection.

## Implementation plan

1. Establish a Python 3.11+ modular project, reproducible dependencies, VS Code settings, and a `Python (ReturnWise)` Jupyter kernel path.
2. Implement the official Kaggle CLI connector with access checks, safe extraction, integrity checks, SHA-256 fingerprints, dataset metadata, and a seven-day cache refresh policy. Keep manual CSV import available.
3. Implement schema profiling, quality checks, temporal leakage audit, EDA, statistical tests, model benchmarking, calibration, explanation fallbacks, uncertainty flags, cost scenarios, ablation helpers, model export, and factual reporting.
4. Build a Streamlit workspace whose predictions depend on a real exported artifact and whose unlabeled cost mode reports expected counts.
5. Create the 21 sequential research notebooks with reusable package calls and no fabricated saved outputs.
6. Verify the code paths that are possible in this environment. Actual research runs remain dependent on obtaining the competition CSVs or importing a documented manual dataset and installing optional research libraries.

## Implementation delivered

Steps 1 through 5 have been created. The app has Dataset and quality, EDA and statistics, Risk queue, Decision simulator, and Models and research pages. The 21 notebooks call modular functions, persist notebook handoff state locally, and contain no cached results presented as findings. Credentials and downloaded data are not included in source control.

## Notebook 01 acquisition update (recorded before the full research run)

Notebook 01 completed on 2026-11-15 using the already-present Kaggle CSV cache. A successful authenticated competition listing returned `train.csv` (33,115,892 bytes), `test.csv` (8,180,207 bytes), and `sample_submission.csv` (478,039 bytes). Each local file matched its listed byte size and passed a full streaming CSV parse. The training target is binary, the test split has no target, and train/test feature columns match in name and order. Local SHA-256 hashes and the dataset fingerprint are recorded in `data/kaggle/dataset_metadata.json` and the acquisition entry in `experiments/history.jsonl`; Kaggle's listing did not provide remote SHA-256 values.

The existing `retail-return-risk-modeling.zip` is an incomplete 1 MiB partial with a Kaggle resume marker. ZIP validation returned `BadZipFile`; it was left untouched and not extracted. The three complete CSVs were verified independently and reused without a second download. Notebook 01 records this distinction. Notebook 02 has not been run.

The installed Kaggle CLI 2.2.4 source provides five automatic download retries with exponential backoff and marker-validated HTTP range resume when the server supplies matching object identity metadata. Its competition download help has no separate resume flag; `--force` disables cross-invocation resume. ReturnWise avoids `--force` for ordinary cache synchronization and gives the CLI sufficient time for its bounded retries. No additional retry loop or credential change was made.

## Application and test status

The live artifact, model metadata, risk queue, and decision simulator were exercised against the actual exported model and a small sample of competition rows. Streamlit AppTest opened all six registered routes through `streamlit_app.py` without exceptions. The initial suite had **5 passed in 29.53 seconds**. A follow-up added cross-name submission identifier validation and its negative regression case; the expanded suite had **6 passed in 34.88 seconds** after the main model changes. A subsequent small Optuna SQLite URL-rendering correction was exercised by the successful persistent pilot, but the suite was not rerun after that correction because available RAM had approached the safety floor. Kaggle competition notebooks were not inspected; no global novelty claim is made.
