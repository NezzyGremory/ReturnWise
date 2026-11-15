# ReturnWise: Adaptive Calibration-Aware Cost-Sensitive Return Prioritization

Research report draft · generated 2026-11-15T07:45:30+00:00 UTC

> This document is an evidence-based report draft assembled from the local ReturnWise run. Observed validation results, simulated cost assumptions, and methodological limits are identified separately. It does not claim causation, field impact, or global novelty.

> **Status note (2026-11-15):** the report below records the original Quick Mode run and its exported artifact. Follow-up audits found a generic feature-pipeline schema defect and corrected the code. The original metrics and artifact were not regenerated after that correction. New raw-feature CV, OOF blending, and a two-trial Optuna pilot are documented in the linked continuation reports; they are exploratory and do not replace the original holdout evaluation.

## Abstract

The study evaluates a pre-shipment return-risk ranking and capacity-constrained review workflow on 40,000 rows in a stratified internal holdout. The cross-validated selection procedure chose **Soft voting: HistGradientBoosting + Logistic regression** with **beta** probability calibration. On the untouched holdout, ROC-AUC was **0.589**, average precision was **0.556**, Brier score was **0.243**, and log loss was **0.679**. These results indicate modest ranking signal and a small improvement in probability scores over the constant-prevalence reference; they do not establish operational impact.

## 1. Research question and scope

The operational question is whether order-time fields can rank transactions for a limited human review queue while using calibrated probabilities and explicit error-cost assumptions. The declared prediction point is pre-shipment. The primary outcome is the binary competition field `returned`.

The work uses a Kaggle competition dataset and an internal stratified random holdout. There is no validated event-order timestamp in the supplied data, so the study does not make a temporal-validation claim. The competition test file has no target and is excluded from all labeled evaluation.

### Research questions

1. Do the conservatively selected pre-shipment fields provide discrimination above the prior and random-ranking references?
2. Do probability maps improve reliability and proper scoring rules on held-out data?
3. At fixed review capacity, do cost-sensitive or uncertainty-aware ranking rules improve observed holdout decision metrics under stated assumptions?

### Claims boundary

The analysis is predictive and observational. Statistical association, feature importance, and model scores do not show that a customer, product, or action causes returns. Simulated expected costs are not observed financial costs. A reduction in returns, improved profit, or successful operational deployment has not been tested.

## 2. Dataset and acquisition validation

The active research frame contains **200,000 rows and 16 columns**. Source recorded by the application: `kaggle`.

Notebook 01 recorded 200,000 training rows, 50,000 unlabeled competition-test rows, and a matching train/test feature schema: **True**.

Dataset fingerprint: `2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `train.csv` | 33115892 | `1d44901b7a1f1aaacb5a899909c31b68cb999106d789fd8153552e0e9107025b` |
| `test.csv` | 8180207 | `261d13cc3bc3f7f4baf4b39113a02787043739439a0234573be08df4685d90cf` |
| `sample_submission.csv` | 478039 | `9b2340d344297d4d047542db825d612cbf0325b660ae300679f6742d8432d733` |

The cached ZIP was marked invalid (`BadZipFile`) and was not extracted. The complete CSV files were verified independently; the incomplete archive was left intact and not used.

### Dataset schema

| column | role | dtype | missing | unique_values | description |
| --- | --- | --- | --- | --- | --- |
| customer_age | numeric feature | int64 | 0 | 78 | Customer age (competition description). |
| product_price | numeric feature | float64 | 0 | 199800 | Product price at purchase (competition description). |
| discount_percent | numeric feature | float64 | 0 | 199907 | Discount applied to the product (competition description). |
| product_rating | numeric feature | float64 | 0 | 199936 | Average product rating (competition description). |
| past_purchase_count | numeric feature | int64 | 0 | 26 | Number of the customer's past purchases (competition description; timing needs verification). |
| past_return_rate | numeric feature | float64 | 0 | 199934 | Historical customer return rate (competition description; leakage safety needs verification). |
| delivery_delay_days | numeric feature | float64 | 0 | 199947 | Delay against estimated delivery (competition description; likely post-shipment). |
| session_length_minutes | numeric feature | float64 | 0 | 199921 | Duration of the browsing session in minutes (competition description). |
| num_product_views | numeric feature | int64 | 0 | 50 | Products viewed during the session (competition description). |
| device_type | categorical feature | str | 0 | 3 | Device category used for the session (competition description). |
| product_category | categorical feature | str | 0 | 6 | Product category (competition description). |
| shipping_method | categorical feature | str | 0 | 3 | Selected shipping option (competition description). |
| payment_method | categorical feature | str | 0 | 4 | Payment type (competition description). |
| used_coupon | numeric feature | int64 | 0 | 2 | Whether a coupon was used (competition description). |
| returned | target | int64 | 0 | 2 | Binary competition target: 1 returned, 0 kept. |
| order_id | identifier candidate | int64 | 0 | 200000 | Meaning not documented; inferred from observed values only. |

### Data quality findings

Exact duplicate rows: **0** (0.000%). Missing cells: **0**. No rows were removed by the quality audit.

Observed training target distribution:

| class | rows | percent |
| --- | --- | --- |
| 0 | 105081 | 52.54 |
| 1 | 94919 | 47.46 |

No missing values were detected in the training CSV.

Domain screening flagged values outside conventional ranges. These counts are quality flags, not automatic deletion rules; the raw competition CSV was retained and preprocessing did not silently clip or discard them.

| field | screening rule | rows |
| --- | --- | --- |
| product_price | Negative value where the documented quantity is non-negative | 27285 |
| session_length_minutes | Negative value where the documented quantity is non-negative | 1487 |
| num_product_views | Negative value where the documented quantity is non-negative | 21443 |
| discount_percent | Discount percent outside 0–100 | 1686 |
| product_rating | Rating outside the common 0–5 screening range; confirm source scale | 3831 |

IQR screening counts (flags only; no automatic outlier treatment):

| feature | 1.5×IQR flags |
| --- | --- |
| past_purchase_count | 23432 |
| product_price | 11689 |
| delivery_delay_days | 9053 |
| past_return_rate | 7260 |
| customer_age | 0 |
| discount_percent | 0 |
| product_rating | 0 |
| session_length_minutes | 0 |
| num_product_views | 0 |
| used_coupon | 0 |
| returned | 0 |
| order_id | 0 |

## 3. Exploratory analysis and feature associations

The target is near balanced in the supplied training sample (47.46% returned, 52.54% kept). Numerical summaries below describe the observed file and include raw fields that are excluded from modeling for timing or identifier reasons.

| feature | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| customer_age | 2e+05 | 56.48 | 17.95 | 10 | 44 | 58 | 72 | 87 |
| product_price | 2e+05 | 37.98 | 45.63 | -183.3 | 6.877 | 24.46 | 53.36 | 855.1 |
| discount_percent | 2e+05 | 42.87 | 21.51 | -7.541 | 25.85 | 42.95 | 62.64 | 78.79 |
| product_rating | 2e+05 | 2.813 | 1.22 | 0.4453 | 1.789 | 2.819 | 3.81 | 5.453 |
| past_purchase_count | 2e+05 | 9.62 | 3.14 | 1 | 8 | 9 | 10 | 26 |
| past_return_rate | 2e+05 | 0.2151 | 0.1372 | -0.06088 | 0.1183 | 0.1914 | 0.282 | 0.8664 |
| delivery_delay_days | 2e+05 | 0.6251 | 2.331 | -9.05 | -0.7519 | 0.519 | 1.897 | 7.961 |
| session_length_minutes | 2e+05 | 77.41 | 36.39 | -11.01 | 49 | 81.56 | 108.2 | 137.7 |
| num_product_views | 2e+05 | 14.65 | 12.66 | -5 | 3 | 13 | 25 | 44 |
| used_coupon | 2e+05 | 0.6154 | 0.4865 | 0 | 0 | 1 | 1 | 1 |
| returned | 2e+05 | 0.4746 | 0.4994 | 0 | 0 | 0 | 1 | 1 |
| order_id | 2e+05 | 1.249e+05 | 7.215e+04 | 0 | 6.239e+04 | 1.249e+05 | 1.873e+05 | 2.5e+05 |

Categorical coverage (largest group is reported by count; its rate is descriptive):

| feature | levels | largest group rows | largest-group return rate |
| --- | --- | --- | --- |
| device_type | 3 | 109371 | 0.4512 |
| product_category | 6 | 48287 | 0.5031 |
| shipping_method | 3 | 131311 | 0.4551 |
| payment_method | 4 | 56667 | 0.4418 |

Feature-target tests used two-sided Mann–Whitney U with rank-biserial effect sizes for numeric fields and Pearson chi-square with Cramér’s V for categorical fields. Benjamini–Hochberg correction was applied over the eligible feature tests. P-values are sensitive to the 200,000-row sample size; effect sizes are included to show magnitude.

| feature | test | effect_size | effect_label | p_adjusted_bh | significant_fdr_0_05 |
| --- | --- | --- | --- | --- | --- |
| product_category | Pearson chi-square | 0.07646 | Cramer's V | 1.543e-249 | True |
| shipping_method | Pearson chi-square | 0.06961 | Cramer's V | 1.97e-210 | True |
| payment_method | Pearson chi-square | 0.06081 | Cramer's V | 2.017e-159 | True |
| device_type | Pearson chi-square | 0.05671 | Cramer's V | 5.76e-140 | True |
| used_coupon | Pearson chi-square | 0.05495 | Cramer's V | 5.246e-133 | True |
| product_price | Mann-Whitney U (two-sided) | 0.04622 | rank-biserial correlation | 3.17e-71 | True |
| num_product_views | Mann-Whitney U (two-sided) | -0.02316 | rank-biserial correlation | 5.081e-19 | True |
| session_length_minutes | Mann-Whitney U (two-sided) | -0.02048 | rank-biserial correlation | 3.18e-15 | True |
| discount_percent | Mann-Whitney U (two-sided) | 0.01298 | rank-biserial correlation | 6.286e-07 | True |
| customer_age | Mann-Whitney U (two-sided) | -0.01268 | rank-biserial correlation | 1.02e-06 | True |
| product_rating | Mann-Whitney U (two-sided) | 0.004899 | rank-biserial correlation | 0.05813 | False |

10 of 11 eligible feature tests passed adjusted p < 0.05. This is evidence of association within this dataset, not causal evidence or proof of transportability.

## 4. Prediction-time policy and preparation

The prediction scenario is pre-shipment. The audit excluded the label and `order_id`; it also excluded `delivery_delay_days` as a post-shipment outcome and `past_purchase_count` / `past_return_rate` because the event cutoff and leakage-safe construction are undocumented.

| feature | availability_assessment | decision |
| --- | --- | --- |
| customer_age | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| product_price | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| discount_percent | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| product_rating | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| past_purchase_count | historical aggregate; event order and cutoff unavailable | exclude until leakage-safe construction is verified |
| past_return_rate | historical aggregate; event order and cutoff unavailable | exclude until leakage-safe construction is verified |
| delivery_delay_days | post-shipment outcome | exclude from pre-shipment prediction |
| session_length_minutes | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| num_product_views | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| device_type | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| product_category | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| shipping_method | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| payment_method | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| used_coupon | likely available by order placement; verify with data documentation | eligible for pre-shipment candidate set |
| returned | label; never a feature | exclude |
| order_id | identifier; not a transaction behavior | exclude unless a documented use case justifies it |

The fitted inference schema contains 11 raw fields: `customer_age`, `product_price`, `discount_percent`, `product_rating`, `session_length_minutes`, `num_product_views`, `device_type`, `product_category`, `shipping_method`, `payment_method`, `used_coupon`.

The feature transformer creates three deterministic interactions: price × discount, views per session minute, and price × rating. A source audit found that the original generic pipeline built its column list from the raw inputs and silently dropped these new columns. HistGradientBoosting did receive the interactions; Logistic Regression, Random Forest, and Extra Trees did not. The saved soft-voting model therefore mixed raw Logistic Regression inputs with engineered HistGradientBoosting inputs. A post-audit code correction now aligns the preprocessing schema, but the corrected raw-versus-engineered comparison has not yet run. The saved metrics below describe the original pipelines.

Preprocessing is fit within each training fold: numeric values use median imputation, categorical values use most-frequent imputation and one-hot encoding (with an ordinal representation for HistGradientBoosting), and Logistic Regression uses scaling. The final export retains preprocessing as part of the fitted pipeline. No oversampling was applied. The existing exported artifact has not been retrained with the source correction.

### Internal split design

Notebook 07 used a stratified 80/20 split with random seed 42: 160,000 development rows and 40,000 final holdout rows. Model selection and calibration were performed on development folds; the final holdout was reserved for one-time model evaluation. This is a random internal split, not temporal validation.

## 5. Model development and comparison

Quick Mode was used to bound CPU and memory use. Available candidates were evaluated with three-fold stratified cross-validation. ROC-AUC and average precision were accompanied by threshold metrics and runtime/memory observations; accuracy was not the sole selection criterion.

The resource-bounded registry used 32 trees for Random Forest and Extra Trees and 48 boosting iterations for HistGradientBoosting. The selected soft-voting model averages the component probabilities with equal weight. Native model libraries were restricted to one thread for this run. These are Quick Mode settings, not a tuned optimum.

| model | folds | roc_auc | pr_auc | brier_score | log_loss | mean_fit_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HistGradientBoosting | 3 | 0.5857 | 0.5514 | — | — | 2.331 | completed |
| Logistic regression | 3 | 0.5854 | 0.5511 | — | — | 2.356 | completed |
| Extra trees | 3 | 0.5637 | 0.5306 | — | — | 7.821 | completed |
| Random forest | 3 | 0.5529 | 0.5208 | — | — | 9.267 | completed |
| Dummy (prior) | 3 | 0.5 | 0.4746 | — | — | 1.488 | completed |

An additional raw-feature LightGBM pilot was run after the audit but before the pipeline correction: three-fold ROC-AUC 0.5866 (fold SD 0.0032), PR-AUC 0.5531 (fold SD 0.0026), and mean fit time 2.04 seconds per fold. This is a preliminary raw-feature result; it is not directly comparable to the corrected engineered pipeline.

Out-of-fold probability blends:

| blend | roc_auc | pr_auc | fit_seconds_total | inference_seconds_total | rss_delta_mb_total |
| --- | --- | --- | --- | --- | --- |
| OOF mean: HistGradientBoosting + Logistic regression | 0.5873 | 0.5533 | 15.78 | 1.826 | 47.42 |
| HistGradientBoosting | 0.5856 | 0.5512 | 10.22 | 1.289 | 29.6 |
| Logistic regression | 0.5854 | 0.5509 | 5.562 | 0.5372 | 17.82 |

Selection table (development data only):

| model | cv_roc_auc | cv_pr_auc | selected_calibrator | brier_score | log_loss | selection_score |
| --- | --- | --- | --- | --- | --- | --- |
| Soft voting: HistGradientBoosting + Logistic regression | 0.587 | 0.5533 | beta | 0.2434 | 0.6798 | 0.33 |
| Logistic regression | 0.5853 | 0.5509 | beta | 0.2437 | 0.6804 | 0.4825 |
| HistGradientBoosting | 0.5845 | 0.5503 | beta | 0.2437 | 0.6805 | 0.6875 |
| Extra trees | 0.5641 | 0.531 | sigmoid | 0.2461 | 0.6853 | 1 |

The selected development candidate was **Soft voting: HistGradientBoosting + Logistic regression** and the selected probability map was **beta**. The composite selection score combines CV ROC-AUC (weight 0.28), CV PR-AUC (0.18), Brier score (0.24), log loss (0.14), ROC-AUC variability (0.07), fit time (0.05), and memory delta (0.04), using normalized ranks. This makes efficiency and probability quality explicit parts of selection.

Optional candidates not run:

| model family | reason |
| --- | --- |
| CatBoost | Optional dependency `catboost` is not installed. |
| XGBoost | Optional dependency `xgboost` is not installed. |
| TabPFN | Optional dependency `tabpfn` is not installed. |

The original generated report had no hyperparameter search result. A later two-trial LightGBM Optuna pilot is recorded in `hyperparameter_optimization.md`; it is a smoke test, not a tuned optimum, and the original Quick Mode candidate settings remain untuned.

## 6. Calibration and final holdout evaluation

Cross-validated probability-map comparison for the selected model:

| model | method | roc_auc | pr_auc | brier_score | log_loss | expected_calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| Soft voting: HistGradientBoosting + Logistic regression | beta | 0.587 | 0.5533 | 0.2434 | 0.6798 | 0.004762 |
| Soft voting: HistGradientBoosting + Logistic regression | isotonic | 0.5866 | 0.5476 | 0.2435 | 0.6809 | 0.006373 |
| Soft voting: HistGradientBoosting + Logistic regression | sigmoid | 0.587 | 0.5533 | 0.2434 | 0.6798 | 0.004481 |
| Soft voting: HistGradientBoosting + Logistic regression | uncalibrated | 0.587 | 0.5533 | 0.2436 | 0.6802 | 0.01326 |

Final-holdout ranking and proper scores before and after the CV-selected map:

| probability map | ROC-AUC | average precision | Brier score | Log loss |
| --- | --- | --- | --- | --- |
| Uncalibrated | 0.5895 | 0.5561 | 0.2433 | 0.6796 |
| Selected calibrator | 0.5895 | 0.5561 | 0.2431 | 0.6791 |

| Final holdout measure | Result |
| --- | ---: |
| Rows | 40000 |
| Returned prevalence | 0.4746 |
| ROC-AUC | 0.5895 |
| Average precision (PR-AUC) | 0.5561 |
| Precision / recall / F1 at 0.50 | 0.5547 / 0.4357 / 0.4881 |
| Brier / log loss / ECE | 0.2431 / 0.6791 / 0.001805 |
| Constant-prevalence Brier / log loss | 0.2494 / 0.6919 |
| Confusion matrix TN / FP / FN / TP at 0.50 | 14375 / 6641 / 10712 / 8272 |

ROC-AUC exceeds 0.50 and PR-AUC exceeds the holdout prevalence, but both margins are modest. Brier score and log loss improve over a constant-prevalence prediction. The selected calibrator changed probability quality only slightly; calibration does not materially change rank ordering for the selected monotonic map.

## 7. Explainability and uncertainty

Permutation importance was measured by ROC-AUC decrease on a validation subset of development data. The values describe model reliance under this sample distribution; they are not causal effects. Correlated inputs may share or mask importance.

| feature | importance_mean_roc_auc_decrease | importance_std |
| --- | --- | --- |
| product_category | 0.02415 | 0.001088 |
| shipping_method | 0.0217 | 0.001481 |
| used_coupon | 0.01633 | 0.001495 |
| payment_method | 0.01193 | 0.0006765 |
| device_type | 0.01113 | 0.0007496 |
| product_price | 0.003178 | 0.000694 |
| session_length_minutes | 0.001784 | 0.0001046 |
| num_product_views | 0.001249 | 0.000364 |
| discount_percent | 0.001098 | 0.0001557 |
| customer_age | 0.0001329 | 0.0001073 |
| product_rating | -1.875e-05 | 0.0001733 |

SHAP status: `not attempted`. When SHAP was unavailable, the project retained model-agnostic permutation importance as the global explanation. No SHAP result is invented.

The uncertainty workflow uses distance from 0.50 as a margin-based flag. It is a review heuristic, not a calibrated confidence interval or a guarantee of error. No conformal coverage claim is made.

## 8. Capacity-constrained and cost-sensitive decision analysis

Review capacities were evaluated at 5%, 10%, 20%, and 30%. The low, medium, and high false-positive, false-negative, and review costs are configurable simulation assumptions; none was supplied as an observed business cost. Labeled holdout counts below are observed on the internal decision-validation split. The unlabelled operational queue instead reports expected counts from predicted probabilities.

Final decision-evaluation sample at each review capacity (medium scenario is false-positive cost 1, false-negative cost 5, review cost 0.25):

| strategy | capacity_percent | expected_decision_cost | precision_at_k | recall_at_k | false_positives | false_negatives |
| --- | --- | --- | --- | --- | --- | --- |
| random | 5 | 9.166e+04 | 0.48 | 0.05057 | 1040 | 18024 |
| probability_only | 5 | 8.976e+04 | 0.6385 | 0.06727 | 723 | 17707 |
| fixed_threshold | 5 | 8.976e+04 | 0.6385 | 0.06727 | 723 | 17707 |
| cost_sensitive | 5 | 8.976e+04 | 0.6385 | 0.06727 | 723 | 17707 |
| calibration_aware_cost_sensitive | 5 | 8.976e+04 | 0.6385 | 0.06727 | 723 | 17707 |
| uncertainty_aware | 5 | 9.154e+04 | 0.49 | 0.05162 | 1020 | 18004 |
| random | 10 | 8.848e+04 | 0.4768 | 0.1005 | 2093 | 17077 |
| probability_only | 10 | 8.509e+04 | 0.6178 | 0.1302 | 1529 | 16513 |
| fixed_threshold | 10 | 8.509e+04 | 0.6178 | 0.1302 | 1529 | 16513 |
| cost_sensitive | 10 | 8.509e+04 | 0.6178 | 0.1302 | 1529 | 16513 |
| calibration_aware_cost_sensitive | 10 | 8.509e+04 | 0.6178 | 0.1302 | 1529 | 16513 |
| uncertainty_aware | 10 | 8.82e+04 | 0.4883 | 0.1029 | 2047 | 17031 |
| random | 20 | 8.192e+04 | 0.4793 | 0.202 | 4166 | 15150 |
| probability_only | 20 | 7.663e+04 | 0.5894 | 0.2484 | 3285 | 14269 |
| fixed_threshold | 20 | 7.663e+04 | 0.5894 | 0.2484 | 3285 | 14269 |
| cost_sensitive | 20 | 7.663e+04 | 0.5894 | 0.2484 | 3285 | 14269 |
| calibration_aware_cost_sensitive | 20 | 7.663e+04 | 0.5894 | 0.2484 | 3285 | 14269 |
| uncertainty_aware | 20 | 8.103e+04 | 0.4976 | 0.2097 | 4019 | 15003 |
| random | 30 | 7.552e+04 | 0.4778 | 0.302 | 6266 | 13250 |
| probability_only | 30 | 6.908e+04 | 0.5672 | 0.3585 | 5194 | 12178 |
| fixed_threshold | 30 | 6.908e+04 | 0.5672 | 0.3585 | 5194 | 12178 |
| cost_sensitive | 30 | 6.908e+04 | 0.5672 | 0.3585 | 5194 | 12178 |
| calibration_aware_cost_sensitive | 30 | 6.908e+04 | 0.5672 | 0.3585 | 5194 | 12178 |
| uncertainty_aware | 30 | 7.451e+04 | 0.4918 | 0.3108 | 6099 | 13083 |

Ablation comparison at 10% capacity:

| experiment | strategy | expected_decision_cost | precision_at_k | recall_at_k | runtime_seconds |
| --- | --- | --- | --- | --- | --- |
| A: probability-ranking baseline | probability_only | 6.814e+04 | 0.6144 | 0.1295 | 1.304 |
| B: calibration | probability_only | 6.814e+04 | 0.6144 | 0.1295 | 0.8687 |
| C: cost-sensitive | cost_sensitive | 6.814e+04 | 0.6144 | 0.1295 | 0.7984 |
| D: calibration + cost-sensitive | calibration_aware_cost_sensitive | 6.814e+04 | 0.6144 | 0.1295 | 0.7858 |
| E: calibration + cost + uncertainty review | uncertainty_aware | 7.001e+04 | 0.5169 | 0.1089 | 0.7537 |

At fixed constant costs, the cost-sensitive score is monotonic in return probability. In this run, probability-only, fixed-threshold-under-capacity, cost-sensitive, and calibration-aware cost-sensitive strategies selected the same ordering and had identical results. The uncertainty-aware strategy did not improve the measured expected cost at 10% capacity in the saved ablation. Therefore the experiments do not support a claim that the full adaptive combination outperforms probability ranking on this data.

## 9. Export and application integration

The model artifact was exported at `2026-11-15T07:25:48+00:00` for scenario `pre_shipment` with calibrated probabilities: `True`.

The saved artifact contains the fitted preprocessing/model pipeline, selected calibrator, raw feature schema, holdout metrics, decision configuration, and permutation importance. Notebook 20 reloaded the artifact and compared raw and calibrated probabilities on the same rows against in-memory predictions at absolute tolerance 1e-12.

The competition submission file, when present, is generated from unlabeled `test.csv` rows using the sample submission schema. Its predictions are not evaluated because no labels are provided.

The Streamlit application loads only the exported artifact for operational scoring. The risk queue accepts unlabeled CSV input; the decision simulator labels outputs as expected or simulated; the research view exposes saved metrics and this report.

Artifact paths: `artifacts/returnwise_model.joblib`, `artifacts/metadata.json`. Experiment state: `experiments/research_state.joblib`. Run history: `experiments/history.jsonl`.

## 10. Limitations and threats to validity

1. **Random split rather than time split.** No event-order timestamp supports temporal validation or a claim about future-period performance.
2. **Questionable raw values.** Quality screening found negative product prices, negative session duration and product-view counts, discount values outside 0–100, and ratings outside the common 0–5 range. The raw file was preserved; no undocumented correction was applied. These values may materially affect model behavior and require source-owner clarification before production use.
3. **Historical aggregate timing.** Past purchase count and past return rate were excluded because their as-of cutoffs cannot be verified. Their exclusion could remove potentially useful information, but including them without provenance would risk leakage.
4. **Limited candidate coverage and pipeline correction.** CatBoost, XGBoost, and TabPFN were not installed. Optuna and LightGBM were installed later; LightGBM has corrected raw/engineered CV and a two-trial Optuna smoke pilot. The saved metrics predate the feature-pipeline correction. The selected candidate is best only among those compared under the original Quick Mode setup.
5. **Modest predictive signal.** Holdout discrimination is above baseline but not strong. Performance is not evidence of broadly reliable individual-level predictions.
6. **Cost assumptions.** Expected costs depend on illustrative configurable values and do not represent measured economics or intervention effectiveness.
7. **Dataset provenance.** The competition CSVs were checksum-verified locally; the source archive remained a partial invalid ZIP. The dataset appears competition-generated, and its semantics should be checked before operational use.
8. **No causal or field evaluation.** No experiment tested whether a review action changes returns, customer outcomes, or profit.
9. **Literature scope.** The references below motivate established methods; they are not a systematic literature review or a comparison against all competition submissions.

## 11. Conclusion and contribution boundary

This run provides a reproducible comparison of baseline and tree-based tabular pipelines, cross-validated probability calibration, permutation importance, and capacity-constrained decision rules on a competition dataset. The selected soft-voting model achieved modest discrimination and small proper-score gains over a constant-prevalence reference on a random internal holdout. The calibration-aware and cost-sensitive strategies did not outperform probability ranking at fixed capacity in the observed comparisons; uncertainty-aware ranking was weaker in the saved 10% ablation. The defensible contribution is this dataset-specific empirical evaluation and transparent decision simulation, not a claim of a novel algorithm or demonstrated business impact.

## 12. Reproducibility record

- Dataset fingerprint: `2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56`.
- Experiment mode: `quick`.
- Selected model/calibrator: `Soft voting: HistGradientBoosting + Logistic regression` / `beta`.
- Split: stratified 80/20 internal holdout, random seed 42; development evaluation used stratified three-fold CV in Quick Mode.
- Pipeline: `python scripts/run_research_pipeline.py` from the project root after the validated dataset and `.venv` kernel are available.
- The runner verifies Notebook 01 hashes and schema, runs notebooks serially, checks each state handoff, and saves notebook outputs. It does not download data or install dependencies.
- Core dependencies are listed in `requirements.txt`; optional model packages are skipped with recorded reasons when unavailable.

### Recorded experiment runs

| run | created_at | dataset fingerprint |
| --- | --- | --- |
| Dataset acquisition | 2026-11-15T06:05:11+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Baseline machine learning benchmark | 2026-11-15T07:03:01+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Advanced model benchmark | 2026-11-15T07:17:35+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| OOF probability blend benchmark | 2026-11-15T07:18:16+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Probability calibration and model selection | 2026-11-15T07:19:19+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Explainable AI feature importance | 2026-11-15T07:22:52+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Uncertainty analysis | 2026-11-15T07:23:07+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Cost-sensitive decision analysis | 2026-11-15T07:23:28+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Adaptive priority sensitivity analysis | 2026-11-15T07:23:59+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Ablation study | 2026-11-15T07:25:19+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| One-time internal final holdout evaluation | 2026-11-15T07:25:36+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |
| Model export | 2026-11-15T07:25:51+00:00 | 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56 |

## References and method sources

These references support method descriptions; they do not establish novelty of the combined workflow.

- Platt, J. (1999). [Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Platt1999.pdf).
- Kull, M., Silva Filho, T., & Flach, P. (2017). [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html).
- Lundberg, S. M. & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
- Ke, G. et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
- Chen, T. & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
- [Kaggle competition dataset](https://www.kaggle.com/competitions/retail-return-risk-modeling).

---
Prepared from local CSVs, saved notebook outputs, experiment state, and model metadata. No external or fabricated performance result was added.
