# ReturnWise Final Research Findings (Current Status)

Date: 2026-11-15. This is a status report for the bounded work completed in this workspace; the full requested definition of done is **not met**.

## A. Dataset

The local verified training CSV contains 200,000 rows, 16 columns, and 94,919 returns (47.4595%). Test has 50,000 unlabeled rows. Files have no missing values or exact duplicate rows, and train/test IDs do not overlap. However, large counts of negative prices and product views, plus out-of-range values in several numeric columns, have undocumented meaning. The source ZIP is partial, while the cached CSVs pass local hash and schema checks. No row-level timestamps or feature snapshot rules are available.

Do not alter raw values. The current 11 model fields are an expanded order-time candidate, not a verified pre-shipment feature set. `returned` is label-only; `order_id`, `delivery_delay_days`, `past_purchase_count`, and `past_return_rate` are excluded from the current candidate due identifier, post-event, or unresolved cutoff risks. See [dataset_deep_analysis.md](dataset_deep_analysis.md).

## B. Modeling and evaluation

Historically evaluated: Dummy prior, Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, and a soft vote of HGB + Logistic Regression. LightGBM was installed and evaluated later with corrected raw/engineered CV and a two-trial Optuna smoke pilot. CatBoost, XGBoost, SHAP, Decision Tree, SVM, TabPFN, SMOTE, and repeated CV were not run.

Corrected raw three-fold CV ROC-AUC / PR-AUC: Logistic Regression 0.585437 / 0.551053; HGB 0.586347 / 0.552064; LightGBM 0.586576 / 0.553064. Raw OOF LR + HGB averaged 0.587648 ROC-AUC / 0.553586 PR-AUC. Optuna's best of two trials scored 0.587598 ROC-AUC, but this is not a tuned optimum.

The historical internal holdout result remains the only completed holdout evaluation: ROC-AUC 0.589481, average precision 0.556060, Brier 0.243089, log loss 0.679141, ECE 0.001805 for the original soft vote + beta calibrator. It predates the pipeline correction and must not be presented as the holdout score of the corrected candidates. No confidence intervals are available.

## C. Feature, calibration, explanation, and decision findings

The three deterministic interactions had small, mixed CV effects; raw fields remain the default. Fold-local L1 selection reduced ROC-AUC and PR-AUC and increased variability and fit time; keep all 11 source fields. Beta calibration gave a small proper-score improvement on the original holdout; sigmoid had marginally lower CV ECE. Permutation importance placed product category, shipping method, coupon use, payment method, and device type highest, but this is historical model reliance, not causality or stable importance. At fixed capacity, constant-cost ranking tied probability ranking; uncertainty-aware ranking was weaker in the saved 10% ablation. Costs and perfect review identification are assumptions.

## D. Final model and project status

No new final model is selected. The old artifact is retained and still loads in the existing app, but was not retrained after the pipeline fix. No corrected candidate has been calibrated on a separate calibration set, evaluated once on a fresh holdout, exported, and integrated. New corrected OOF results are exploratory, not deployable evidence.

The generic model pipeline was fixed so that engineered input columns are no longer silently dropped by preprocessing, and submission validation now checks IDs even when sample and test identifier column names differ. Six integration tests passed in 34.88 seconds after the main model changes; the later small Optuna storage-URL change was exercised by the successful persistent pilot but the suite was not rerun afterward. The original artifact was not overwritten.

Optuna 4.9.0 and LightGBM 4.7.0 were installed in the existing environment without upgrading core dependencies. `pip check` reported an unrelated pre-existing scikit-image/networkx version mismatch. No notebooks beyond the prior set were added. New reports cover dataset, experiments, and limitations; four dataset plots remain in `reports/figures/`.

## E. Stop conditions and next work

The last resource reading after the two-trial pilot was about 0.471 GiB available RAM and 0.348 GiB free disk. RAM was close to the runner's 0.45 GiB floor, so more fitting, tuning, calibration, and export work was stopped as required. The remaining point-in-time provenance questions and unexplained raw value ranges also prevent a verified operational claim.

Before resuming, obtain source timestamps/cutoff rules and anomaly semantics, then estimate resources. If the machine has safe headroom, run a larger sequential CV/tuning budget, repeated or bootstrap uncertainty analysis, select a candidate without touching holdout, fit a distinct calibration/deployment model, export it, and run the full regression suite. A clean model comparison and deployment refit must not reuse the old holdout as a tuning set.

## Conclusion

The dataset is structurally usable for descriptive, stratified research but has substantial semantic and point-in-time limitations. The current evidence supports modest predictive signal, not operational effectiveness, temporal generalization, global novelty, or a final-model improvement. The prior artifact is preserved; the new pipeline fix and exploratory comparisons are documented, while model selection, corrected holdout validation, deployment export, and final integration remain outstanding.
