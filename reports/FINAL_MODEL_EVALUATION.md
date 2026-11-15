# ReturnWise Final Model Evaluation Record

**Evaluation status:** historical holdout results are reportable for the originally exported model. The corrected raw-feature candidates have CV results only; no corrected-candidate holdout score is available.

## Validation design and holdout use

The original split used seed 42 and stratified sampling: 160,000 rows for development and a 40,000-row final holdout. Candidate comparison and calibration selection used development-side CV. Notebook 19 then fit the historical evaluation model on a 120,000-row development subset, fit beta calibration on a separate 40,000-row calibration subset, and scored the 40,000-row holdout. The decision-study notebooks describe a separate decision holdout created from the 160,000-row development partition.

The original 40,000-row holdout was evaluated for the historical model, so its metrics are no longer an unseen result for any new model-selection decision. It was not used in the recorded corrected raw/engineered CV or two-trial Optuna pilot, but it must not be reused to choose among those corrected candidates. No new split was created. There are no event timestamps, so no temporal validation is possible.

## Corrected raw-feature CV candidates

Three-fold stratified CV on the 160,000-row development set, seed 42; each fold has approximately 106,667 fit rows and 53,333 validation rows. These are development CV summaries, not holdout scores.

| Candidate | Mean ROC-AUC | Fold SD | Configuration |
|---|---:|---:|---|
| Logistic Regression | 0.585437 | 0.003905 | Balanced class weights, max_iter 2000, scaled numeric features; raw 11 fields. |
| HistGradientBoosting | 0.586347 | 0.003246 | 48 iterations, learning rate 0.08; raw inputs with ordinal-coded categories. |
| LightGBM | **0.586576** | 0.003231 | Quick default, 48 estimators, learning rate 0.05, 31 leaves, one thread; raw 11 fields. |

LightGBM is the CV-leading corrected single model by mean ROC-AUC, but its margin over HGB is 0.000229 and no repeated-CV or paired interval is available. Logistic Regression is a simpler, close baseline (0.001139 lower than LightGBM); observed CV dispersion is of similar order to these gaps. The OOF LR + HGB blend reached ROC-AUC 0.587648, but it has no independently evaluated holdout or repeated-CV result. No corrected candidate is declared a final deployment winner from these small differences.

## Historical holdout metrics

These metrics belong to the original exported soft vote (HistGradientBoosting + Logistic Regression) with beta calibration. Its generic pipeline had mixed feature schemas: Logistic Regression used raw fields while HGB received interactions. The original artifact was evaluated once on the 40,000-row holdout and has not been replaced.

| Metric | Holdout result | Reference |
|---|---:|---:|
| ROC-AUC | 0.589481 | Random-ranking reference was reported as 0.50. |
| Precision (threshold 0.50) | 0.554684 | No constant-prevalence precision comparison was recorded. |
| Recall (threshold 0.50) | 0.435735 | — |
| F1 (threshold 0.50) | 0.488067 | — |
| Brier score | 0.243089 | Constant-prevalence Brier: 0.249355. |
| Log loss | 0.679141 | Constant-prevalence log loss: 0.691856. |

Threshold 0.50 confusion matrix (TN, FP, FN, TP): 14,375; 6,641; 10,712; 8,272. These scores show modest discrimination and small proper-score gains over the constant-prevalence predictor. They do not establish future-period or deployment performance.

## Calibration record

For the historical model, training-side CV selected beta calibration. On the holdout, the calibrated model had the values above; the uncalibrated version had ROC-AUC 0.5895, average precision 0.5561, Brier 0.2433, and log loss 0.6796. The change in probability scores was small and ranking was unchanged at reported precision. This is not a calibration result for corrected raw LightGBM or Logistic Regression. No corrected candidate calibration is selected.

## Baseline and interpretation limits

The constant-prevalence predictor is available as a proper-score baseline only; no candidate-specific CV precision, recall, F1, Brier, or log-loss table was saved for the corrected candidates. Do not infer those values from ROC-AUC or from a different split. Three folds provide fold variability but not confidence intervals. Current output uses only scores actually recorded; no corrected-model holdout metric is invented.

**Conclusion:** LightGBM raw is the CV-leading candidate selected for the requested refit; the refit was stopped by the RAM preflight before model training. Logistic Regression remains a simpler close baseline. No corrected candidate has an independent holdout score, and the historical artifact remains the only saved model.
