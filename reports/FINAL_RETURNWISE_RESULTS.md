# ReturnWise Final Data Science Results

**Finalization status: partial, blocked before refit by the RAM preflight.** Dataset findings and evidence-based decisions are finalized. A corrected-pipeline holdout evaluation, refit, test predictions, and submission were not produced. The prior artifact remains intact. See [deployment data status](FINAL_DEPLOYMENT_DATA_STATUS.md).

## Problem and dataset

ReturnWise ranks orders by probability of return for a capacity-constrained review workflow. The supplied competition data contain 200,000 labeled rows and 50,000 unlabeled test rows. The target is 47.4595% positive. The checked CSVs have no missing values or exact duplicate rows. Numeric anomalies include 27,285 negative prices, 21,443 negative product-view counts, 1,487 negative session lengths, 1,686 out-of-range discounts, and 3,831 ratings outside the common 0–5 range. A shortlist scan found no common extreme sentinel constants such as -999/-99/999/9999; the negative view counts span the ordinary integer range and their meaning is unresolved.

## Features and preprocessing

The 11-field modeling set is an **expanded order-time candidate**, not a verified pre-shipment feature contract: `customer_age`, `product_price`, `discount_percent`, `product_rating`, `session_length_minutes`, `num_product_views`, `device_type`, `product_category`, `shipping_method`, `payment_method`, and `used_coupon`. No timestamp or snapshot rule confirms all fields' availability. `order_id` is excluded as an identifier; `delivery_delay_days` is excluded as post-event; historical counts/rates are excluded until cutoff rules are proven; `returned` is label-only.

Raw values remain unchanged. Model preprocessing uses training-fold median numeric imputation and most-frequent categorical imputation plus one-hot encoding. Three candidate interactions produced small, mixed CV changes and are off by default. Fold-local L1 selection reduced both ROC-AUC and PR-AUC, so all 11 source fields are retained. Detailed decisions are in [FINAL_DATA_QUALITY.md](FINAL_DATA_QUALITY.md).

## Models and evidence

Corrected raw-feature three-fold CV on 160,000 development rows (seed 42) produced mean ROC-AUC 0.585437 (SD 0.003905) for Logistic Regression, 0.586347 (SD 0.003246) for HistGradientBoosting, and 0.586576 (SD 0.003231) for LightGBM. The raw OOF LR + HGB blend scored 0.587648 ROC-AUC, but this gain is from one three-fold setup and has no repeated-CV or confidence-interval support. A two-trial Optuna pilot reached 0.587598; it is not a reliable tuned optimum. LightGBM is the corrected-CV leading candidate, while Logistic Regression is a simpler close baseline. No corrected candidate is declared final for deployment.

The historical soft vote of HGB + Logistic Regression with beta calibration is the only candidate with a saved holdout evaluation. On the original 40,000-row random internal holdout, it recorded ROC-AUC 0.589481, precision 0.554684, recall 0.435735, F1 0.488067 at threshold 0.50, Brier 0.243089, and log loss 0.679141. Constant-prevalence Brier and log loss were 0.249355 and 0.691856. The historical artifact's generic pipeline had mixed feature schemas; these metrics are not the holdout result for corrected raw candidates. The holdout is not reused here to select a corrected model. Full score provenance and limits are in [FINAL_MODEL_EVALUATION.md](FINAL_MODEL_EVALUATION.md).

## Calibration and decision analysis

Beta calibration was selected for the historical model through development-side CV and produced only a small proper-score change; its ranking remained unchanged. No corrected candidate was calibrated. Review ranking by probability and the fixed-constant-cost score produced identical selections at fixed capacities, as the cost score is monotone in probability. Uncertainty-aware ranking was weaker in the saved 10% ablation. Costs are scenario assumptions, and the simulator assumes reviewed returns can be correctly identified; no causal reduction in returns or profit impact is demonstrated.

## Finalization decision and limitations

The requested LightGBM refit script checked resources and stopped before scientific-library imports when available RAM was 0.463 GiB, below its 0.55 GiB safety threshold. A subsequent reading was 0.414 GiB RAM and 0.239 GiB free disk. The dataset also lacks point-in-time provenance and anomaly semantics. Therefore no model was fit, no final artifact was copied or renamed, and no 50,000-row test prediction or submission was generated. See [deployment data status](FINAL_DEPLOYMENT_DATA_STATUS.md).

## Definition of done

| Requirement | Status |
|---|---|
| Data quality and anomaly decisions documented | Complete; raw CSVs preserved. |
| Feature set and availability limits documented | Complete; expanded candidate only, timing unverified. |
| Reproducible preprocessing pipeline available | Available in research code; not locked into a newly refit deployment artifact. |
| Model selected from valid corrected-pipeline evaluation | LightGBM selected as the requested CV-leading refit candidate; no independent corrected holdout evaluation. |
| Holdout metrics and limitations reported | Complete for historical artifact only; not transferable to corrected candidates. |
| Calibration evaluated | Complete for historical model; not for corrected candidates. |
| Model refit/export and metadata | Incomplete; RAM preflight stopped before fit. |
| Exactly 50,000 test probabilities and template submission | Incomplete; no final refit artifact available. |
| Final reports | Complete: [data quality](FINAL_DATA_QUALITY.md), [model evaluation](FINAL_MODEL_EVALUATION.md), [experiment log](FINAL_EXPERIMENT_LOG.md), and this summary. |
| Unsupported claims avoided | Complete. No temporal, operational impact, test ROC-AUC, or novelty claim is made. |
