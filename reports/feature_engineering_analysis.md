# Feature Engineering Analysis

## Question and design

Test whether the three deterministic candidate interactions improve ranking beyond the 11 raw fields:

- `product_price * discount_percent`
- `num_product_views / session_length_minutes`
- `product_price * product_rating`

The comparison used the same 160,000-row development frame, seed 42, three stratified folds, estimator settings, and fold assignments for the raw and engineered pipelines. Feature construction is row-local and uses no target. Infinite ratios are changed to missing and then handled by the training-fold numeric imputer. The 40,000-row final holdout was not used.

## Audit correction

The original generic pipeline created the interactions before a `ColumnTransformer` built from the raw columns. Since the transformer retained only named raw columns and dropped the remainder, Logistic Regression, Random Forest, Extra Trees, LightGBM, and XGBoost silently lost the engineered fields. HistGradientBoosting and CatBoost used a separately constructed engineered schema. The selected historical soft vote consequently combined Logistic Regression on raw inputs with HistGradientBoosting on engineered inputs.

`returnwise/models.py` now builds the preprocessing schema from the engineered frame and exposes an explicit feature-engineering switch. The comparison below was rerun after this correction. The deployment artifact was not retrained.

## Results

| Model | Feature set | ROC-AUC | ROC-AUC SD | PR-AUC | PR-AUC SD | Mean fit seconds/fold |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Raw | 0.585437 | 0.003905 | 0.551053 | 0.002281 | 1.49 |
| Logistic Regression | Engineered | 0.585517 | 0.003789 | 0.551199 | 0.002165 | 2.23 |
| HistGradientBoosting | Raw | 0.586347 | 0.003246 | 0.552064 | 0.002376 | 1.52 |
| HistGradientBoosting | Engineered | 0.585704 | 0.003985 | 0.551442 | 0.002219 | 1.98 |
| LightGBM | Raw | 0.586576 | 0.003231 | 0.553064 | 0.002638 | 2.49 |
| LightGBM | Engineered | 0.586535 | 0.003175 | 0.553219 | 0.002359 | 2.32 |

The changes are small and inconsistent. Logistic Regression improves by 0.00008 ROC-AUC and 0.00015 PR-AUC. HistGradientBoosting loses 0.00064 ROC-AUC and 0.00062 PR-AUC. LightGBM changes by -0.00004 ROC-AUC and +0.00016 PR-AUC. None is a material or consistent improvement in this three-fold Quick Mode comparison.

## Decision

Use the raw 11-field pipeline as the default candidate. Keep the interactions as a reproducible optional comparison, not as an assumed enhancement. Do not retain them in the deployment model based on these results. No log/robust price transform, clipping, invalid-value correction, or interaction search was run because the source semantics for negative values remain unresolved.

## Limits

Three folds provide a quick paired comparison, not a definitive confidence interval. The values are from the expanded pre-shipment candidate set, whose capture times have not been proven. Only the specified three interactions were tested; the result does not rule out other transformations after data provenance is clarified.
