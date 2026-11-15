# Method references and review boundaries

This map names established methods relevant to ReturnWise. They motivate evaluation choices; they do not establish a global novelty claim for the combined application.

## Dataset and access

- Kaggle competition: [E-Commerce Returns: Who Will Send It Back?](https://www.kaggle.com/competitions/retail-return-risk-modeling)
- Official Kaggle CLI: [Kaggle/kaggle-cli](https://github.com/Kaggle/kaggle-cli) and [competition CLI reference](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/competitions.md)

## Calibration and probabilistic prediction

- Platt, J. (1999). [Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Platt1999.pdf).
- Wenger, J. et al. (2020). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v108/wenger20a.html).
- Kull, M., Silva Filho, T., & Flach, P. (2017). [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html).
- The research code compares uncalibrated probabilities with sigmoid, isotonic, and beta maps on held-out calibration partitions. Calibration selection is made from cross-validation results, not the final holdout.

## Explainability and model families

- Lundberg, S. M. & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
- Hollmann, N. et al. (2022). [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848). TabPFN is an optional candidate with a conservative resource gate, not an assumed winner.
- Ke, G. et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
- Chen, T. & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
- CatBoost authors. [CatBoost: Unbiased Boosting with Categorical Features](https://catboost.ai/docs/en/concepts/educational-materials-papers).

## Decision support and uncertainty

The decision simulator uses an explicit expected-cost comparison under user-entered assumptions and a capacity constraint. It tests whether calibration changes reliability and whether review selection changes decision cost at fixed capacity. Because expected cost-sensitive review benefit is monotonic in calibrated probability when costs are constant, cost-sensitive ranking can produce the same ordering as probability ranking; any claimed operational benefit must therefore come from threshold/capacity choices, calibrated estimates, or uncertainty handling and must be demonstrated on validation data.

Margin-based uncertainty flags are implemented as a review aid. Conformal prediction is not enabled in this baseline: coverage claims would require explicit exchangeability assumptions and a held-out conformal calibration design.

## Literature review status

The citations above cover standard methods and model families. They do not constitute a systematic literature review or a review of Kaggle notebooks. Kaggle competition notebooks must be inspected after account access is available before comparing this project with participant solutions. Until then, describe the work as an empirical evaluation of established components for this dataset and decision setting, and do not claim global novelty.
