# ReturnWise product and deployment audit

Audit date: 2026-11-15

## Application today

The repository is a Python 3.11+ research workspace with a six-page Streamlit application. The pages cover overview, Kaggle or manual CSV setup, exploratory analysis, a risk queue, a decision simulator, and experiment reporting. Research and scoring logic lives in the `returnwise/` package. Local CSVs, saved experiment state, and model files are kept under ignored workspace directories.

## Existing behavior

- Dataset setup supports local manual CSVs and a cached Kaggle connector. It reports schema, missingness, duplicate rows, candidate identifiers, and domain or timing flags.
- Analysis uses the selected local dataset for descriptive statistics and labeled feature associations.
- The risk queue accepts an unlabeled CSV, scores it with the saved artifact, ranks calibrated probabilities, offers search, and exports the queue.
- The decision simulator compares review strategies with explicit cost assumptions.
- Research reports expose dataset provenance, experiment history, artifact metadata, and a downloadable report.
- The integration tests exercise the registered routes, artifact scoring, queue ranking, simulation, and submission shape.

## UI and workflow gaps

The original pages were functionally useful but retained the stock Streamlit visual treatment, had no shared brand mark, and made the initial workflow and model readiness hard to scan. The queue ranked records but did not let an operator set an exact review count or see the selected capacity alongside the ranked list. Historical artifact metrics appeared under generic validation wording.

## Research state and model limits

The verified local competition files contain 200,000 labeled training rows and 50,000 unlabeled test rows with `returned` as the binary target. The data do not establish transaction timestamps or feature snapshot cutoffs.

The retained soft-vote artifact is historical. Its random 40,000-row internal holdout result was ROC-AUC 0.589481, average precision 0.556060, Brier score 0.243089, and log loss 0.679141. This predates the corrected raw-feature pipeline. The corrected three-fold OOF blend result of ROC-AUC 0.587648 is exploratory and is not a holdout result. The reports state that no corrected candidate has a fresh holdout evaluation, deployment refit, or integrated replacement artifact. The existing artifact can still run for research, but the app must label its output experimental and must not call it production validated.

## Architecture and deployment risk

The current product UI and inference runtime are Streamlit and Python. Netlify can publish a static frontend, but this repository has no HTTP scoring backend for Netlify to call and no Netlify configuration. A static site cannot load the local joblib artifact or execute the Streamlit process. The Netlify site therefore needs to state that live inference is unavailable, while supporting local CSV inspection and ranking of an explicitly supplied probability column. The existing Streamlit workspace remains the local Python scoring path.

## Implemented redesign

The Streamlit app now uses the ReturnWise visual system and parcel mark. Its overview makes the research status visible. The risk queue exposes an exact review capacity, prioritizes rows from the saved experimental artifact, and labels that scoring path as historical research. The research page separates corrected OOF exploration from the old random internal holdout.

The new `site/` frontend is a separate, responsive static site configured to publish through Netlify. It supports local browser CSV profiling and queue ranking only when valid probabilities are present in an unlabeled file. It does not load the Python artifact or make a scoring request. The demo rows are invented for interface preview and are labeled as such. `site/research-notes.html` keeps dataset limits, model protocols, feature policy, and deployment boundaries with the public interface.

The package, training artifacts, and experiments were not changed. The implementation does not add a hosted inference API or claim that model output is production validated. Netlify account linkage and deploy state must be checked separately before a preview or production publish.
