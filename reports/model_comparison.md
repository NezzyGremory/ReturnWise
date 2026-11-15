# ReturnWise Model Comparison

Date: 2026-11-15. The dataset and feature-timing boundaries are detailed in [dataset_deep_analysis.md](dataset_deep_analysis.md).

## Objective and hypothesis

Compare the current fast baseline families and new raw-feature candidates using ROC-AUC and PR-AUC. The working hypothesis was that nonlinear models or equal-weight OOF blends would improve on Logistic Regression.

## Method

The original saved benchmark used 160,000 development rows, three-fold stratified CV, seed 42, and Quick Mode estimator budgets. Its internal holdout has 40,000 rows. A subsequent comparison used raw-feature pipelines on the same three-fold design after correcting the generic preprocessing schema. The old artifact was not retrained. The new OOF results are reported separately from the old run.

## Historical quick benchmark

| Candidate | Mean CV ROC-AUC | Mean CV PR-AUC | ROC-AUC SD | PR-AUC SD |
|---|---:|---:|---:|---:|
| Soft voting: HGB + Logistic Regression | 0.587039 | 0.553303 | 0.004070 | 0.002602 |
| Logistic Regression | 0.585251 | 0.550937 | 0.003710 | 0.002145 |
| HistGradientBoosting | 0.585704 | 0.551442 | 0.003985 | 0.002219 |
| Extra Trees | 0.563745 | 0.530606 | 0.000440 | 0.000245 |

The historical holdout result for the then-selected soft vote with beta calibration was ROC-AUC 0.589481, average precision 0.556060, Brier 0.243089, log loss 0.679141, and ECE 0.001805. The constant-prevalence Brier and log loss were 0.249355 and 0.691856.

## Corrected raw-feature CV and OOF blend

| Candidate | 3-fold mean ROC-AUC | 3-fold mean PR-AUC | OOF ROC-AUC | OOF PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.585437 | 0.551053 | 0.585378 | 0.550909 |
| HistGradientBoosting | 0.586347 | 0.552064 | 0.586270 | 0.551878 |
| LightGBM | 0.586576 | 0.553064 | 0.586492 | 0.552728 |
| LR + HGB mean | — | — | **0.587648** | 0.553586 |
| LR + LightGBM mean | — | — | 0.587528 | **0.553849** |
| HGB + LightGBM mean | — | — | 0.587210 | 0.553152 |

## Interpretation

The original soft vote was the best among its historical Quick Mode comparisons. Corrected raw-feature comparisons put LightGBM slightly ahead of HGB and Logistic Regression individually, while the new LR + HGB OOF mean has the strongest ROC-AUC and LR + LightGBM has the strongest PR-AUC. These are small differences in one three-fold design. The historical model selected by a holdout-aware workflow remains the recorded evaluation model; the new exploratory OOF scores do not justify changing the final or deployed model.

## Limitations and conclusion

The original generic pipeline dropped engineered columns for some estimators, so its historical metric table reflects mixed feature schemas. The corrected comparisons use raw features consistently, but they do not have a new untouched holdout evaluation. No confidence intervals, repeated folds, or temporal validation are available. There is no statistically established winner among the corrected raw candidates. See [advanced experiment results](advanced_experiment_results.md) and [calibration analysis](calibration_analysis.md).
