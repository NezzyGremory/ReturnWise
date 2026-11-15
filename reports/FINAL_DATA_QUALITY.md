# ReturnWise Final Data Quality and Feature Decision

**Status:** data audit complete for the supplied CSVs; feature point-in-time provenance is not available. No raw file was changed.

## Dataset and target

The locally cached, checksum-verified `train.csv` has 200,000 rows and 16 columns. `test.csv` has 50,000 rows and 15 feature columns, with no target. `sample_submission.csv` has 50,000 rows and columns `ID` and `returned`. The training target is binary with 94,919 returns (47.4595%) and 105,081 non-returns (52.5405%). Existing full-file checks found no missing cells, duplicate rows, schema mismatch, or train/test ID overlap. The incomplete 1 MiB ZIP was not extracted; validated CSVs were used. Fingerprint: `2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56`.

## Quality findings

| Field | Screen | Count | Share | Evidence / decision |
|---|---|---:|---:|---|
| `product_price` | below 0 | 27,285 | 13.64% | Values span a distribution rather than one documented invalid code; retain raw pending provenance. |
| `num_product_views` | below 0 | 21,443 | 10.72% | Integer values span -5 to 44; -1, 0, and 1 all occur frequently. No evidence that -1 alone is a sentinel. |
| `session_length_minutes` | below 0 | 1,487 | 0.74% | Meaning is unknown; retain raw and flag for source-owner clarification. |
| `discount_percent` | outside 0–100 | 1,686 | 0.84% | Do not clip without source semantics. |
| `product_rating` | outside common 0–5 | 3,831 | 1.92% | No documented alternate scale; retain raw. |
| `past_return_rate` | outside 0–1 | 2,220 | 1.11% | Excluded from current candidate due both range and cutoff ambiguity. |

### Sentinel screening

A streaming scan of all 200,000 training rows checked common exact codes `-999`, `-99`, `-9`, `-1`, `0`, `1`, `99`, `999`, and `9999` across numeric fields. It found no occurrences of the extreme codes `-999`, `-99`, `999`, or `9999`. Exact `-1`, `0`, and `1` occur in `num_product_views`, whose observed integer range is -5 to 44; the presence of several adjacent negative and nonnegative values does not establish a sentinel convention. `past_purchase_count=1` occurs 33 times and is within its observed count range. This screen cannot rule out undocumented encodings outside that shortlist.

Train/test numeric marginals were previously found to be close (maximum inspected numeric SMD approximately 0.0065); categorical share differences were also small (maximum approximately 0.0045). These are descriptive comparisons, not a formal drift test. See the [dataset analysis](dataset_deep_analysis.md) and its four actual CSV plots in `reports/figures/`.

## Feature availability decision

| Group | Fields | Decision |
|---|---|---|
| Plausible order/checkout/session candidates, capture time unverified | `customer_age`, `product_price`, `discount_percent`, `session_length_minutes`, `num_product_views`, `device_type`, `product_category`, `shipping_method`, `payment_method`, `used_coupon` | Candidate-only for research; no verified point-in-time guarantee in CSV or metadata. |
| Snapshot/cutoff unresolved | `product_rating`, `past_purchase_count`, `past_return_rate` | Exclude from the expanded candidate until as-of rules are obtained; rating may include later reviews, history may include current/future events. |
| Post-event / identifier | `delivery_delay_days`, `order_id` | Exclude from pre-shipment candidate; delay requires actual delivery, and ID has no supported time interpretation. |
| Label | `returned` | Target only, never predictor. |

There is no feature set that can be certified as strictly available at a defined pre-shipment instant using these CSVs alone. Do not describe the model as leakage-free or operationally validated until the data owner supplies event timestamps, feature snapshots, and aggregate cutoffs.

## Preprocessing decision

Preserve source values. Do not delete flagged rows, replace negatives with zero, clip ranges, or impute from the test distribution. The reproducible raw-feature pipeline uses median imputation for numeric inputs; most-frequent imputation and one-hot encoding for categorical inputs; and training-fold fit for all preprocessing. The optional three interactions were not retained by default because paired CV effects were small and mixed. L1 selection was rejected because both ROC-AUC and PR-AUC decreased. No source correction is defensible without data-owner semantics.

**Conclusion:** the CSVs are structurally usable for exploratory binary classification, but value semantics and point-in-time availability remain unresolved. The final research feature set is the 11-field expanded candidate listed above, not a verified pre-shipment contract.
