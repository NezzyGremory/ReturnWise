# ReturnWise Decision Optimization

## Objective and hypothesis

Compare return-risk ranking rules under a fixed human-review capacity and stated false-positive, false-negative, and review costs. The hypothesis was that calibration-aware, cost-sensitive, or uncertainty-aware scoring could improve expected decision cost over probability ranking.

## Method and assumptions

The original workflow evaluated review capacities of 5%, 10%, 20%, and 30%, using labeled internal validation data. The medium scenario assigned costs 1 per false positive, 5 per false negative, and 0.25 per review. Review is assumed to correctly identify a return among selected cases; this is a simulation assumption, not measured intervention effectiveness. The unlabeled operational queue reports expected counts rather than observed outcomes.

## Results at 10% capacity

| Strategy | Expected cost (ablation) | Precision@K | Recall@K |
|---|---:|---:|---:|
| Probability ranking | 68,140 | 0.6144 | 0.1295 |
| Calibration then probability ranking | 68,140 | 0.6144 | 0.1295 |
| Cost-sensitive ranking | 68,140 | 0.6144 | 0.1295 |
| Calibration + cost-sensitive | 68,140 | 0.6144 | 0.1295 |
| Calibration + cost + uncertainty | 70,010 | 0.5169 | 0.1089 |

In the separate capacity table, probability ranking at 10% had expected cost 85,090, precision@K 0.6178, recall@K 0.1302, 1,529 false positives, and 16,513 false negatives. Random review had expected cost 88,480 and precision@K 0.4768. These tables use different saved evaluation subsets/configuration and should not be merged into one count series.

## Interpretation and conclusion

For fixed constant costs, the implementation's cost score is strictly increasing in predicted return probability. Thus its ranking matches probability-only selection at fixed capacity, consistent with the identical results. Calibration also preserves ranking for the monotonic map. Uncertainty-aware ranking underperformed in the saved 10% ablation. Lift@K and a measured review treatment effect were not established; no business savings claim is supported.

## Limitations

Costs are hypothetical and review effectiveness is assumed. The scores use the original model and split design. No cost uncertainty analysis, policy learning, causal treatment evaluation, or external validation was performed. The priority queue should be treated as a research simulation pending observed operational outcomes and verified feature availability.
