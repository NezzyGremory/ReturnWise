# ReturnWise Explainability Analysis

## Objective and hypothesis

Describe which inputs the historical model relies on, using a model-agnostic measure. Permutation importance was used as an exploratory ranking; the hypothesis was that product and fulfillment categories would have stronger validation reliance than weakly associated numeric fields.

## Method and parameters

Permutation importance was calculated on a validation subset of development data, with ROC-AUC decrease as the scoring measure. SHAP was not attempted. The resulting importance is predictive model reliance under this sample distribution, not an effect estimate or causal explanation.

## Results

| Feature | Mean ROC-AUC decrease | SD |
|---|---:|---:|
| product_category | 0.02415 | 0.001088 |
| shipping_method | 0.02170 | 0.001481 |
| used_coupon | 0.01633 | 0.001495 |
| payment_method | 0.01193 | 0.000676 |
| device_type | 0.01113 | 0.000750 |
| product_price | 0.003178 | 0.000695 |
| session_length_minutes | 0.001784 | 0.000105 |
| num_product_views | 0.001249 | 0.000364 |
| discount_percent | 0.001098 | 0.000156 |
| customer_age | 0.000133 | 0.000107 |
| product_rating | -0.000019 | 0.000173 |

## Interpretation and conclusion

The largest recorded permutation drops were for product category and shipping method, followed by coupon use, payment method, and device type. Product rating's mean was effectively zero. Correlated predictors can divide or mask importance, and the low-order rank should not be read as causal or universal. No stability comparison across folds was run, so the table is descriptive of the original saved model only.

## Limitations

The importance output predates the generic pipeline correction and was not recomputed for corrected raw-only candidates. SHAP and local explanations were not run. Feature timing is unverified, and categories may be proxies for other dataset construction factors. The numeric/categorical screens in [dataset_deep_analysis.md](dataset_deep_analysis.md) are unadjusted descriptive evidence, not a substitute for model explanation.
