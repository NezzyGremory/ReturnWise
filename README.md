# ReturnWise

**A research workspace for e-commerce return-risk review prioritization.**

ReturnWise includes a six-page Streamlit research application and a separate static Netlify frontend. The interface helps inspect data and fit a ranked review list to available team capacity. A probability is an estimate. The project does not show that a review prevents returns or increases profit.

## Current research status

The verified competition files contain 200,000 labeled training rows and 50,000 unlabeled test rows. The target is `returned`; the competition metric is ROC-AUC. The supplied specification does not establish an event-order timestamp or a feature snapshot cutoff, so temporal generalization has not been evaluated.

The corrected raw-feature blend has an exploratory three-fold out-of-fold ROC-AUC of 0.587648. This is not a holdout result. The retained soft-vote artifact predates the corrected pipeline. Its earlier random 40,000-row internal holdout reported ROC-AUC 0.589481, average precision 0.556060, Brier score 0.243089, and log loss 0.679141. No corrected candidate has a fresh holdout evaluation, deployment refit, or integrated replacement artifact. The existing artifact is experimental and must not be described as production validated. See [research notes](site/research-notes.html) and [product audit](docs/product_audit.md).

## Run the Streamlit workspace

```powershell
python -m streamlit run streamlit_app.py
```

The Dataset and quality page supports Kaggle Connected Mode and Manual CSV Mode. Imports are stored under `data/manual/`; Kaggle files and version metadata are stored under `data/kaggle/`. These local data files are git-ignored. The risk queue uses the saved historical artifact for research scoring and labels its status in the interface.

For a local setup on Windows PowerShell:

```powershell
.\scripts\setup.ps1
```

In VS Code, choose the `.venv` interpreter and the `Python (ReturnWise)` notebook kernel. Kaggle credentials belong in the local Kaggle configuration, not in source files, notebooks, chat, or the app. If required, join the [competition](https://www.kaggle.com/competitions/retail-return-risk-modeling) before downloading its data.

## Static Netlify frontend

`site/` is a dependency-free static frontend configured by `netlify.toml`. It supports local CSV profiling and ranking only when the uploaded file already contains a valid `calibrated_probability` or `return_probability` column. It does not execute Python or load the local joblib artifact, and it has no hosted inference endpoint. A labeled target column disables risk ranking. Demo rows are explicitly illustrative.

To publish an existing, linked Netlify site, first check the account and site context with `netlify status`. Then create a preview with `netlify deploy` and publish production with `netlify deploy --prod`. The Netlify site must be linked and authenticated before deploy; this project does not create a new site automatically.

## Research workflow

Run notebooks in numeric order from `notebooks/00_environment_setup.ipynb` through `notebooks/20_returnwise_model_export.ipynb`. Each notebook saves handoff state locally under `experiments/research_state.joblib` so the sequence can use separate Jupyter kernels. After dataset validation, `python scripts/run_research_pipeline.py` executes the sequence serially, skips complete notebooks, checks state handoffs and resource thresholds, and stops on errors.

The acquisition notebook downloads only the requested competition through Kaggle CLI; it does not substitute another dataset. A manual CSV can be selected when Kaggle is unavailable. The competition `test.csv` has no labels and is not used for labeled evaluation. Optional model families are skipped with an explanation when unavailable or ineligible for the configured runtime.

## Interpretation limits

The default research scenario is **pre-shipment**. Identifiers, target-derived or potentially post-event fields, `delivery_delay_days`, and customer history without a verifiable time cutoff are excluded from the corrected raw-feature pipeline. This policy is not proof that every retained feature was available at the exact prediction moment.

The decision simulator exposes false-positive cost, false-negative cost, review cost, and review capacity as assumptions. Expected counts in its unlabeled mode come from model probabilities. Observed classification counts are reported only on labeled validation data. Explanations describe associations and are not causal effects.

## Project layout

- `returnwise/`: data connectors, profiling, statistics, feature handling, experiments, calibration, decisions, exports, and reporting.
- `app_pages/`: Streamlit application pages.
- `site/`: static Netlify frontend and research notes.
- `notebooks/`: sequential research notebooks.
- `data/`, `artifacts/`, `experiments/`: ignored local data, model files, and run state.
- `DESIGN.md`: visual and interaction direction.
- `docs/product_audit.md`: product, research, and deployment audit.
- `docs/references.md`: method references and literature-review boundaries.
