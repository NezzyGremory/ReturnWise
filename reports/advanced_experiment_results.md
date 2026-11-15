# ReturnWise Advanced Experiment Results

Date: 2026-11-15. Dataset fingerprint: `2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56`.

## Objective and hypotheses

Check whether deterministic feature interactions, fold-local feature selection, additional boosting, OOF blending, and a small tuning pilot improve the current expanded order-time candidate. The hypotheses were that interactions or L1 selection might improve generalization, and that a LightGBM blend or tuned LightGBM might outperform the existing quick candidates.

## Method and parameters

All model comparisons used the 160,000-row development split, seed 42, and three stratified folds; the 40,000-row holdout was not used for these continuation experiments. Raw and engineered variants used identical folds and model settings. The three interactions were `product_price * discount_percent`, `num_product_views / session_length_minutes`, and `product_price * product_rating`. Feature selection used fold-local `SelectFromModel` with Logistic Regression (`C=0.1`, L1, `saga`, balanced weights, median threshold). LightGBM used a single-thread quick configuration. The Optuna pilot used persistent SQLite storage and two TPE trials, with early stopping inside each outer training fold.

## Results

| Model | Raw ROC-AUC / PR-AUC | Engineered ROC-AUC / PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.585437 / 0.551053 | 0.585517 / 0.551199 |
| HistGradientBoosting | 0.586347 / 0.552064 | 0.585704 / 0.551442 |
| LightGBM | 0.586576 / 0.553064 | 0.586535 / 0.553219 |

| Logistic Regression selector | ROC-AUC | PR-AUC | Mean fit sec/fold |
|---|---:|---:|---:|
| No selection | 0.585437 | 0.551053 | 1.49 |
| Fold-local L1 | 0.583493 | 0.548963 | 2.59 |

Raw-feature OOF comparisons:

| Candidate | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.585378 | 0.550909 |
| HistGradientBoosting | 0.586270 | 0.551878 |
| LightGBM | 0.586492 | 0.552728 |
| LR + HGB equal-weight mean | **0.587648** | 0.553586 |
| LR + LightGBM equal-weight mean | 0.587528 | **0.553849** |
| HGB + LightGBM equal-weight mean | 0.587210 | 0.553152 |

The two Optuna trial scores were 0.586433 and 0.587598; both completed. The best pilot configuration was `n_estimators=132`, `learning_rate=0.10198`, `num_leaves=16`, `max_depth=10`, `min_child_samples=30`, `subsample=0.7550`, `subsample_freq=2`, `colsample_bytree=0.8574`, `reg_alpha=0.0000772`, and `reg_lambda=0.00000418`.

## Interpretation and conclusion

Feature interactions had small, mixed effects. L1 selection reduced both ranking metrics and increased fit time. The raw OOF blends show small gains over their component models, but no repeated-CV or interval analysis establishes that these gains are stable. The two-trial Optuna result is a smoke pilot and cannot establish an optimum. Raw inputs remain the default for new comparisons; no new final model is selected from these exploratory results.

## Limitations

All scores use random stratified folds and candidate features with unverified capture times. OOF aggregate metrics and mean-fold CV results are distinct summaries and are not directly interchangeable. The original exported artifact and holdout metrics predate the pipeline correction. Larger benchmark budgets, repeated CV, confidence intervals, other libraries, and a deployment refit were stopped when available RAM approached the runner's safety floor. See [feature engineering](feature_engineering_analysis.md), [feature selection](feature_selection_analysis.md), and [model comparison](model_comparison.md).
