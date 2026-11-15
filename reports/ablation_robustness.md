# ReturnWise Ablation and Robustness Results

## Objective and hypotheses

Test whether candidate interactions, L1 feature selection, calibration, cost-sensitive scoring, and uncertainty flags contribute measurable benefit. Check whether reported performance stability is supported by the available folds. The hypotheses were that some additions could improve ranking, proper scores, or review utility.

## Methods

Feature ablations used matched three-fold stratified CV on the 160,000-row development split with seed 42. Calibration and decision ablations use the original saved evaluation workflow. No new random seeds or repeated CV were run. Details and parameterization are in [feature_engineering_analysis.md](feature_engineering_analysis.md), [feature_selection_analysis.md](feature_selection_analysis.md), [calibration_analysis.md](calibration_analysis.md), and [decision_optimization.md](decision_optimization.md).

## Results

| Ablation | Baseline | Variant | Observed result |
|---|---|---|---|
| Feature interactions, Logistic Regression | Raw: ROC 0.585437 / PR 0.551053 | Engineered: 0.585517 / 0.551199 | Tiny positive change |
| Feature interactions, HGB | Raw: 0.586347 / 0.552064 | Engineered: 0.585704 / 0.551442 | Both metrics lower |
| Feature interactions, LightGBM | Raw: 0.586576 / 0.553064 | Engineered: 0.586535 / 0.553219 | Mixed, negligible change |
| L1 selection, Logistic Regression | No selection: ROC 0.585437 / PR 0.551053 | L1: 0.583493 / 0.548963 | Worse metrics, higher fold variability and fit time |
| Calibration, original soft vote | Uncalibrated holdout Brier 0.2433 / log loss 0.6796 | Beta: 0.2431 / 0.6791 | Small proper-score change; ranking unchanged |
| Decision logic, 10% capacity | Probability ranking cost 68,140 | Cost-sensitive 68,140; uncertainty-aware 70,010 | Constant-cost ranking ties; uncertainty variant worse |

The original CV reports three-fold standard deviations, but no repeated-CV distribution, bootstrap interval, or paired confidence interval was computed. Ensemble OOF pairwise correlations were high: LR-HGB 0.9094, LR-LightGBM 0.9181, and HGB-LightGB 0.9592.

## Interpretation and conclusion

No feature or decision addition produced a consistent, material gain in the tested small comparisons. The modest OOF blend gains warrant further validation only if resources and feature provenance permit. Robustness is not established by one seed and three folds. The supported conclusion is that the interactions and L1 selector should not be enabled by default based on these observations; this does not prove they never help on other populations.

## Limitations

Comparisons concern the expanded candidate fields with unresolved point-in-time provenance. Some historical calibration/decision results predate the pipeline correction. No temporal split, subgroup robustness, seed sensitivity, bootstrap uncertainty, label-noise analysis, or source-approved anomaly handling was performed. The research sequence was stopped when available RAM approached the configured safety floor.
