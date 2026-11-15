# ReturnWise Calibration Analysis

## Objective and hypothesis

Evaluate whether probability calibration improves reliability and proper scoring rules for the historical selected soft-voting model, without using the final holdout for method selection. The hypothesis was that beta, sigmoid, or isotonic calibration might improve probability quality while preserving useful ranking.

## Method and parameters

The original study compared uncalibrated, sigmoid, isotonic, and beta maps through training-side nested calibration within three-fold stratified CV. The final calibrator was fitted on a separate 40,000-row calibration subset, and scoring used a separate 40,000-row internal holdout. The beta map is implemented locally with nonnegative coefficient constraints. ECE uses ten equal-width bins; the reliability plot uses quantile bins.

## Results

| CV map | ROC-AUC | PR-AUC | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Beta | 0.5870 | 0.5533 | 0.2434 | 0.6798 | 0.004762 |
| Isotonic | 0.5866 | 0.5476 | 0.2435 | 0.6809 | 0.006373 |
| Sigmoid | 0.5870 | 0.5533 | 0.2434 | 0.6798 | 0.004481 |
| Uncalibrated | 0.5870 | 0.5533 | 0.2436 | 0.6802 | 0.013260 |

| Holdout map | ROC-AUC | Average precision | Brier | Log loss |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.5895 | 0.5561 | 0.2433 | 0.6796 |
| Selected beta | 0.5895 | 0.5561 | 0.2431 | 0.6791 |

The holdout prevalence was 0.4746. Constant-prevalence Brier and log loss were 0.2494 and 0.6919.

## Interpretation and conclusion

Beta calibration was selected in the original workflow and made a small improvement in Brier and log loss without changing the rounded ranking scores. Sigmoid had slightly lower reported CV ECE. The evidence does not show a material calibration advantage for beta over sigmoid. The selected map's holdout proper scores were better than the constant-prevalence reference, but no confidence interval or future-period assessment is available.

## Limitations

These calibration and holdout results belong to the original artifact, whose underlying feature pipeline predates the correction documented in [deep_research_audit.md](deep_research_audit.md). They were not rerun on the corrected raw candidates. No bootstrap intervals, calibration slope/intercept, or groupwise calibration analysis was produced. Calibration cannot resolve the lack of point-in-time feature provenance.
