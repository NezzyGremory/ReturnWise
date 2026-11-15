# Feature Selection Analysis

## Question and design

The current expanded pre-shipment candidate contains 11 raw columns and 23 one-hot/numeric model inputs. This is already a small feature set, so selection must show a measurable CV benefit before being kept.

The no-selection Logistic Regression result is the raw-feature row in `feature_engineering_analysis.md`. The comparison adds `SelectFromModel` with an L1 Logistic Regression selector (`C=0.1`, `solver=saga`, balanced class weights, median coefficient threshold) inside the sklearn pipeline. Each selector was fitted only on that fold's training rows; its selected names were recorded before scoring that fold's validation rows. The folds, seed, preprocessor, and final Logistic Regression candidate match the no-selection baseline.

## Results

| Configuration | ROC-AUC | ROC-AUC SD | PR-AUC | PR-AUC SD | Mean fit seconds/fold |
|---|---:|---:|---:|---:|---:|
| No feature selection | 0.585437 | 0.003905 | 0.551053 | 0.002281 | 1.49 |
| Fold-local L1 selection | 0.583493 | 0.005381 | 0.548963 | 0.003794 | 2.59 |

L1 selection reduces ROC-AUC by 0.00194 and PR-AUC by 0.00209 while increasing mean fit time. It also increases fold variability. It does not improve the validation objective and is not retained.

The selector retained 12 encoded columns in each fold. Ten of those names were common across all three folds; `product_price` and one payment-level indicator varied. This is limited stability evidence from three folds, not a stable universal importance ranking.

## Decision and methods not run

Keep all 11 raw input columns. No input feature is removed by statistical selection. Correlation pruning, mutual information, RFE, sequential selection, tree-based selection, Boruta, and SHAP-based selection were not run. With only 11 source fields and no demonstrated L1 benefit, those extra searches are not justified in the current resource-bounded pass. If future feature growth warrants them, every selector must remain fitted inside each training fold.

## Limits

The model selection result is conditional on the expanded order-time candidate set and the supplied competition data. Three folds do not establish stability across time or production populations. The analysis does not convert model reliance into causal evidence.
