# ReturnWise

**Intelligent E-Commerce Return Risk Management and Decision Support System**

ReturnWise is a local Python research workspace and Streamlit application for studying e-commerce return risk. It separates dataset acquisition, research notebooks, saved model artifacts, and operational scoring. Probabilities and costs remain estimates; the project does not claim that a review prevents returns or increases profit.

## Current status

As of 2026-11-15, the validated Kaggle CSV cache is available and Notebooks 00–20 have completed in serial Quick Mode. The exported pre-shipment model is `Soft voting: HistGradientBoosting + Logistic regression` with beta calibration. On the untouched internal holdout it recorded ROC-AUC 0.5895, average precision 0.5561, Brier score 0.2431, and log loss 0.6791. These are modest predictive results on a random internal split, not evidence of temporal or operational impact.

The Streamlit risk queue and decision simulator use the exported artifact. Automated smoke tests cover the app routes, artifact scoring, and queue decisions. The report draft at `reports/returnwise_research_report.md` is based on saved experiment state and verified local data. See `docs/implementation_plan.md` for resource limits, skipped optional models, data-quality findings, and study limitations.

## Local setup (Windows PowerShell)

```powershell
.\scripts\setup.ps1
```

In VS Code, choose the `.venv` interpreter and the `Python (ReturnWise)` notebook kernel.

Authenticate with Kaggle in a local terminal. Credentials must not be copied into source files, notebooks, chat, or the app:

```powershell
kaggle auth login
kaggle competitions files -c retail-return-risk-modeling
```

If the account has not joined the competition, open the [competition page](https://www.kaggle.com/competitions/retail-return-risk-modeling), sign in, accept its rules if required, then retry. If Kaggle CLI cannot write its default config directory, set `KAGGLE_CONFIG_DIR` to a writable local directory before authenticating and starting ReturnWise.

For example, in PowerShell:

```powershell
$env:KAGGLE_CONFIG_DIR = Join-Path $env:LOCALAPPDATA 'ReturnWise\kaggle'
kaggle auth login
```

## Run the application

```powershell
python -m streamlit run streamlit_app.py
```

The Dataset and quality page supports Kaggle Connected Mode and Manual CSV Mode. Imports are stored under `data/manual/`; Kaggle files and version metadata are stored under `data/kaggle/`. These local data files are git-ignored.

## Research workflow

Run the notebooks in numeric order from `notebooks/00_environment_setup.ipynb` through `notebooks/20_returnwise_model_export.ipynb`. Each notebook saves handoff state locally under `experiments/research_state.joblib` so the sequence can use separate Jupyter kernels. The state and generated datasets are ignored by Git. After dataset validation, `python scripts/run_research_pipeline.py` executes the sequence serially, skips complete notebooks, checks state handoffs and resource thresholds, and stops on errors.

The first notebook checks the environment. The acquisition notebook downloads only the requested competition via Kaggle CLI; it does not substitute another dataset. A manual CSV can be selected on the application page when Kaggle is unavailable. The competition `test.csv` has no labels and is never used for labeled evaluation.

The experiment code provides a Quick mode and a Full mode. Optional libraries (CatBoost, LightGBM, XGBoost, Optuna, SHAP, and TabPFN) are recorded in `requirements.txt`; a model family is skipped with an explanation if it is missing or fails its runtime eligibility check. TabPFN is gated by a conservative row and feature limit.

## Temporal validity

The default research scenario is **pre-shipment**. Identifiers, target-derived or potentially post-event fields, `delivery_delay_days`, and customer historical aggregates without a verifiable time cutoff are excluded. No temporal validation claim is made because an event-order timestamp is not established in the supplied specification. Review `temporal_feature_audit` before changing this policy.

## Cost and interpretation limits

The decision simulator exposes false-positive cost, false-negative cost, review cost, and review capacity as assumptions. Its unlabeled operational mode reports expected counts derived from model probabilities. Observed false-positive and false-negative counts are reported only on labeled validation data. Explanations describe model associations and are not causal effects.

## Project layout

- `returnwise/`: data connectors, profiling, statistics, feature handling, model experiments, calibration, decisions, exports, and reporting.
- `app_pages/`: Streamlit application pages.
- `notebooks/`: sequential research notebooks.
- `data/`: ignored local CSV cache and manual uploads.
- `artifacts/`: ignored exported model and metadata.
- `experiments/`: ignored state and run history.
- `docs/implementation_plan.md`: workspace audit and implementation record.
- `docs/references.md`: method references and literature-review boundaries.
