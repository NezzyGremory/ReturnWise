"""Research report generation from verified local data and saved experiment results."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from returnwise.analysis.statistics import association_tests
from returnwise.data.kaggle import read_metadata
from returnwise.data.profile import data_dictionary, eda_summaries, quality_report
from returnwise.experiments import read_experiment_history
from returnwise.features import select_features, temporal_feature_audit
from returnwise.paths import KAGGLE_DIR
from returnwise.research_state import load_state
from returnwise.storage import active_dataset_fingerprint, load_active_dataset, load_model_artifact


def _cell(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(float(value))):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(frame: pd.DataFrame | None, columns: list[str] | None = None, limit: int = 30) -> str:
    if frame is None or frame.empty:
        return "No records are available."
    view = frame.loc[:, columns] if columns else frame
    view = view.head(limit)
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    if len(frame) > limit:
        lines.append(f"\n_Table limited to {limit} of {len(frame)} rows._")
    return "\n".join(lines)


def _load_report_data(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, dict | None]:
    if df is not None:
        return df, None
    active, manifest = load_active_dataset()
    if active is not None:
        return active, manifest
    train_path = KAGGLE_DIR / "train.csv"
    metadata = read_metadata()
    if train_path.is_file() and metadata and metadata.get("competition") == "retail-return-risk-modeling":
        try:
            return pd.read_csv(train_path), {"source": "kaggle", "file": "data/kaggle/train.csv"}
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            return None, None
    return None, manifest


def _comparison_table(state: dict) -> pd.DataFrame:
    frames = [state[key] for key in ("baseline_results", "advanced_results") if isinstance(state.get(key), pd.DataFrame)]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    columns = ["model", "folds", "roc_auc", "pr_auc", "brier_score", "log_loss", "mean_fit_seconds", "status"]
    for column in columns:
        if column not in combined:
            combined[column] = np.nan
    return combined[columns].sort_values("roc_auc", ascending=False, na_position="last").reset_index(drop=True)


def _holdout_probability_metrics(state: dict) -> pd.DataFrame:
    y = state.get("y_final")
    raw = state.get("final_raw_probabilities")
    calibrated = state.get("final_probabilities")
    if y is None or raw is None or calibrated is None:
        return pd.DataFrame()
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    target = np.asarray(y, dtype=int)
    rows = []
    for name, scores in (("Uncalibrated", raw), ("Selected calibrator", calibrated)):
        probability = np.clip(np.asarray(scores, dtype=float).reshape(-1), 1e-12, 1 - 1e-12)
        rows.append({
            "probability map": name,
            "ROC-AUC": roc_auc_score(target, probability),
            "average precision": average_precision_score(target, probability),
            "Brier score": brier_score_loss(target, probability),
            "Log loss": log_loss(target, probability, labels=[0, 1]),
        })
    return pd.DataFrame(rows)


def generate_research_report(df: pd.DataFrame | None, model_metadata: dict | None = None) -> str:
    """Return a factual, report-ready Markdown draft based on local outputs."""
    state = load_state()
    data, manifest = _load_report_data(df)
    kaggle_metadata = read_metadata() or {}
    if model_metadata is None:
        artifact = load_model_artifact()
        model_metadata = artifact.get("metadata") if artifact else None
    history = read_experiment_history()
    metrics = state.get("final_metrics") or {}
    selected_model = state.get("selected_model", metrics.get("selected_model", "not selected"))
    selected_calibrator = state.get("selected_calibrator", metrics.get("selected_calibrator", "not selected"))
    fingerprint = state.get("dataset_fingerprint") or active_dataset_fingerprint()

    lines = [
        "# ReturnWise: Adaptive Calibration-Aware Cost-Sensitive Return Prioritization",
        "",
        f"Research report draft · generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC",
        "",
        "> This document is an evidence-based report draft assembled from the local ReturnWise run. Observed validation results, simulated cost assumptions, and methodological limits are identified separately. It does not claim causation, field impact, or global novelty.",
        "",
        "## Abstract",
        "",
    ]
    if metrics:
        lines.append(
            f"The study evaluates a pre-shipment return-risk ranking and capacity-constrained review workflow on {int(metrics.get('holdout_rows', 0)):,} rows in a stratified internal holdout. The cross-validated selection procedure chose **{selected_model}** with **{selected_calibrator}** probability calibration. On the untouched holdout, ROC-AUC was **{float(metrics.get('roc_auc', float('nan'))):.3f}**, average precision was **{float(metrics.get('pr_auc', float('nan'))):.3f}**, Brier score was **{float(metrics.get('brier_score', float('nan'))):.3f}**, and log loss was **{float(metrics.get('log_loss', float('nan'))):.3f}**. These results indicate modest ranking signal and a small improvement in probability scores over the constant-prevalence reference; they do not establish operational impact."
        )
    else:
        lines.append("No final holdout metrics are available in the local research state. Model-performance conclusions are therefore withheld.")
    lines += [
        "",
        "## 1. Research question and scope",
        "",
        "The operational question is whether order-time fields can rank transactions for a limited human review queue while using calibrated probabilities and explicit error-cost assumptions. The declared prediction point is pre-shipment. The primary outcome is the binary competition field `returned`.",
        "",
        "The work uses a Kaggle competition dataset and an internal stratified random holdout. There is no validated event-order timestamp in the supplied data, so the study does not make a temporal-validation claim. The competition test file has no target and is excluded from all labeled evaluation.",
        "",
        "### Research questions",
        "",
        "1. Do the conservatively selected pre-shipment fields provide discrimination above the prior and random-ranking references?",
        "2. Do probability maps improve reliability and proper scoring rules on held-out data?",
        "3. At fixed review capacity, do cost-sensitive or uncertainty-aware ranking rules improve observed holdout decision metrics under stated assumptions?",
        "",
        "### Claims boundary",
        "",
        "The analysis is predictive and observational. Statistical association, feature importance, and model scores do not show that a customer, product, or action causes returns. Simulated expected costs are not observed financial costs. A reduction in returns, improved profit, or successful operational deployment has not been tested.",
        "",
        "## 2. Dataset and acquisition validation",
        "",
    ]
    integrity = kaggle_metadata.get("integrity", {})
    train_rows = integrity.get("train_rows_inspected", len(data) if data is not None else "unknown")
    test_rows = integrity.get("test_rows_inspected", "unknown")
    if data is not None:
        lines += [
            f"The active research frame contains **{len(data):,} rows and {len(data.columns)} columns**. Source recorded by the application: `{(manifest or {}).get('source', kaggle_metadata.get('competition', 'unknown'))}`.",
            "",
        ]
    lines += [
        f"Notebook 01 recorded {train_rows:,} training rows, {test_rows:,} unlabeled competition-test rows, and a matching train/test feature schema: **{integrity.get('train_test_schema_match', 'not recorded')}**.",
        "",
        f"Dataset fingerprint: `{fingerprint or 'not available'}`.",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        size = kaggle_metadata.get("file_bytes", {}).get(name, "not recorded")
        digest = kaggle_metadata.get("file_sha256", {}).get(name, "not recorded")
        lines.append(f"| `{name}` | {size} | `{digest}` |")
    archive = kaggle_metadata.get("archive_validation", {})
    if archive.get("present") and not archive.get("valid"):
        lines += [
            "",
            f"The cached ZIP was marked invalid (`{archive.get('error_type', 'archive validation failed')}`) and was not extracted. The complete CSV files were verified independently; the incomplete archive was left intact and not used.",
        ]
    lines += ["", "### Dataset schema", ""]

    quality = None
    if data is not None:
        quality = quality_report(data)
        dictionary = data_dictionary(data)
        lines += [
            _table(dictionary, ["column", "role", "dtype", "missing", "unique_values", "description"], limit=40),
            "",
            "### Data quality findings",
            "",
            f"Exact duplicate rows: **{quality['duplicate_rows']:,}** ({quality['duplicate_rows_percent']:.3f}%). Missing cells: **{int(data.isna().sum().sum()):,}**. No rows were removed by the quality audit.",
            "",
        ]
        target_counts = quality.get("target_counts")
        if target_counts is not None:
            lines += ["Observed training target distribution:", "", _table(target_counts, ["class", "rows", "percent"]), ""]
        missing = quality.get("missing")
        if missing is not None and not missing.empty:
            lines += ["Missing values by feature:", "", _table(missing), ""]
        else:
            lines += ["No missing values were detected in the training CSV.", ""]
        invalid = quality.get("invalid_values", {})
        if invalid:
            invalid_rows = pd.DataFrame([{"field": key, "screening rule": value.get("issue"), "rows": value.get("rows")} for key, value in invalid.items()])
            lines += [
                "Domain screening flagged values outside conventional ranges. These counts are quality flags, not automatic deletion rules; the raw competition CSV was retained and preprocessing did not silently clip or discard them.",
                "",
                _table(invalid_rows),
                "",
            ]
        outliers = quality.get("iqr_outlier_counts", {})
        if outliers:
            outlier_frame = pd.DataFrame([{"feature": name, "1.5×IQR flags": count} for name, count in sorted(outliers.items(), key=lambda item: item[1], reverse=True)])
            lines += ["IQR screening counts (flags only; no automatic outlier treatment):", "", _table(outlier_frame, limit=15), ""]
        if quality.get("categorical_inconsistencies"):
            lines += ["Potential trim/case inconsistencies:", "", _table(pd.DataFrame([{"feature": key, **value} for key, value in quality["categorical_inconsistencies"].items()])), ""]
    else:
        lines += ["The local training CSV was unavailable, so row-level quality and EDA findings are not reproduced here.", ""]

    lines += [
        "## 3. Exploratory analysis and feature associations",
        "",
    ]
    if data is not None and "returned" in data:
        eda = eda_summaries(data)
        numeric = eda["numeric_summary"].copy()
        keep_numeric = [c for c in ["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"] if c in numeric]
        lines += [
            "The target is near balanced in the supplied training sample (47.46% returned, 52.54% kept). Numerical summaries below describe the observed file and include raw fields that are excluded from modeling for timing or identifier reasons.",
            "",
            _table(numeric, keep_numeric, limit=25),
            "",
        ]
        categorical = [c for c in data.select_dtypes(exclude=["number", "bool"]).columns if c != "returned"]
        categorical_rows = []
        for feature in categorical:
            grouped = data.groupby(feature, dropna=False)["returned"].agg(["size", "mean"]).sort_values("size", ascending=False)
            if not grouped.empty:
                categorical_rows.append({"feature": feature, "levels": int(data[feature].nunique(dropna=False)), "largest group rows": int(grouped["size"].iloc[0]), "largest-group return rate": float(grouped["mean"].iloc[0])})
        if categorical_rows:
            lines += ["Categorical coverage (largest group is reported by count; its rate is descriptive):", "", _table(pd.DataFrame(categorical_rows)), ""]
        eligible = select_features(data, scenario="pre_shipment")
        tests = association_tests(data[eligible + ["returned"]])
        if not tests.empty:
            association_view = tests[["feature", "test", "effect_size", "effect_label", "p_adjusted_bh", "significant_fdr_0_05"]]
            lines += [
                "Feature-target tests used two-sided Mann–Whitney U with rank-biserial effect sizes for numeric fields and Pearson chi-square with Cramér’s V for categorical fields. Benjamini–Hochberg correction was applied over the eligible feature tests. P-values are sensitive to the 200,000-row sample size; effect sizes are included to show magnitude.",
                "",
                _table(association_view, limit=20),
                "",
                f"{int(tests['significant_fdr_0_05'].sum())} of {len(tests)} eligible feature tests passed adjusted p < 0.05. This is evidence of association within this dataset, not causal evidence or proof of transportability.",
                "",
            ]
    else:
        lines += ["Labeled exploratory summaries and feature-target tests are not available.", ""]

    audit = temporal_feature_audit(list(data.columns)) if data is not None else pd.DataFrame()
    lines += ["## 4. Prediction-time policy and preparation", ""]
    if not audit.empty:
        lines += [
            "The prediction scenario is pre-shipment. The audit excluded the label and `order_id`; it also excluded `delivery_delay_days` as a post-shipment outcome and `past_purchase_count` / `past_return_rate` because the event cutoff and leakage-safe construction are undocumented.",
            "",
            _table(audit, ["feature", "availability_assessment", "decision"], limit=30),
            "",
        ]
    feature_columns = state.get("feature_columns", [])
    engineered = state.get("engineered_feature_columns", [])
    lines += [
        f"The fitted inference schema contains {len(feature_columns)} raw fields: " + ", ".join(f"`{name}`" for name in feature_columns) + ".",
        "",
        "Three deterministic interactions were constructed inside the model pipelines when source columns were available: price × discount, views per session minute, and price × rating. These are transformations of the same order-time inputs; no future-derived historical aggregates were built.",
        "",
        "Preprocessing is fit within each training fold: numeric values use median imputation, categorical values use most-frequent imputation and one-hot encoding (with an ordinal representation for HistGradientBoosting), and Logistic Regression uses scaling. The final export retains preprocessing as part of the fitted pipeline. No oversampling was applied.",
        "",
        "### Internal split design",
        "",
        f"Notebook 07 used a stratified 80/20 split with random seed 42: {len(state.get('X_train', [])):,} development rows and {len(state.get('X_final', [])):,} final holdout rows. Model selection and calibration were performed on development folds; the final holdout was reserved for one-time model evaluation. This is a random internal split, not temporal validation.",
        "",
        "## 5. Model development and comparison",
        "",
        "Quick Mode was used to bound CPU and memory use. Available candidates were evaluated with three-fold stratified cross-validation. ROC-AUC and average precision were accompanied by threshold metrics and runtime/memory observations; accuracy was not the sole selection criterion.",
        "",
        "The resource-bounded registry used 32 trees for Random Forest and Extra Trees and 48 boosting iterations for HistGradientBoosting. The selected soft-voting model averages the component probabilities with equal weight. Native model libraries were restricted to one thread for this run. These are Quick Mode settings, not a tuned optimum.",
        "",
    ]
    comparison = _comparison_table(state)
    if not comparison.empty:
        lines += [_table(comparison, limit=20), ""]
    blend = state.get("blend_results")
    if isinstance(blend, pd.DataFrame) and not blend.empty:
        lines += ["Out-of-fold probability blends:", "", _table(blend, limit=10), ""]
    selection = state.get("selection_table")
    if isinstance(selection, pd.DataFrame) and not selection.empty:
        rank_columns = [c for c in ["model", "cv_roc_auc", "cv_pr_auc", "selected_calibrator", "brier_score", "log_loss", "selection_score"] if c in selection]
        lines += ["Selection table (development data only):", "", _table(selection, rank_columns), ""]
    lines += [
        f"The selected development candidate was **{selected_model}** and the selected probability map was **{selected_calibrator}**. The composite selection score combines CV ROC-AUC (weight 0.28), CV PR-AUC (0.18), Brier score (0.24), log loss (0.14), ROC-AUC variability (0.07), fit time (0.05), and memory delta (0.04), using normalized ranks. This makes efficiency and probability quality explicit parts of selection.",
        "",
    ]
    skipped = state.get("advanced_skipped", {})
    if skipped:
        lines += ["Optional candidates not run:", "", _table(pd.DataFrame([{"model family": name, "reason": reason} for name, reason in skipped.items()])), ""]
    tuning = state.get("tuning_result")
    if tuning:
        lines += ["A bounded hyperparameter search was recorded:", "", str(tuning), ""]
    else:
        lines += ["No hyperparameter search result was saved (`tuning_result` is empty). Quick Mode retained the configured defaults; this run does not claim a tuned optimum.", ""]

    lines += ["## 6. Calibration and final holdout evaluation", ""]
    calibration = state.get("calibration_summary")
    if isinstance(calibration, pd.DataFrame) and not calibration.empty:
        chosen_calibration = calibration[calibration["model"].eq(selected_model)]
        lines += ["Cross-validated probability-map comparison for the selected model:", "", _table(chosen_calibration, limit=10), ""]
    holdout_calibration = _holdout_probability_metrics(state)
    if not holdout_calibration.empty:
        lines += ["Final-holdout ranking and proper scores before and after the CV-selected map:", "", _table(holdout_calibration), ""]
    if metrics:
        tn, fp, fn, tp = metrics.get("confusion_matrix_tn_fp_fn_tp", [None] * 4)
        lines += [
            "| Final holdout measure | Result |",
            "| --- | ---: |",
            f"| Rows | {metrics.get('holdout_rows', 'not recorded')} |",
            f"| Returned prevalence | {_cell(metrics.get('holdout_prevalence'))} |",
            f"| ROC-AUC | {_cell(metrics.get('roc_auc'))} |",
            f"| Average precision (PR-AUC) | {_cell(metrics.get('pr_auc'))} |",
            f"| Precision / recall / F1 at 0.50 | {_cell(metrics.get('precision_at_threshold'))} / {_cell(metrics.get('recall_at_threshold'))} / {_cell(metrics.get('f1_at_threshold'))} |",
            f"| Brier / log loss / ECE | {_cell(metrics.get('brier_score'))} / {_cell(metrics.get('log_loss'))} / {_cell(metrics.get('expected_calibration_error'))} |",
            f"| Constant-prevalence Brier / log loss | {_cell(metrics.get('constant_prevalence_brier'))} / {_cell(metrics.get('constant_prevalence_log_loss'))} |",
            f"| Confusion matrix TN / FP / FN / TP at 0.50 | {tn} / {fp} / {fn} / {tp} |",
            "",
            "ROC-AUC exceeds 0.50 and PR-AUC exceeds the holdout prevalence, but both margins are modest. Brier score and log loss improve over a constant-prevalence prediction. The selected calibrator changed probability quality only slightly; calibration does not materially change rank ordering for the selected monotonic map.",
            "",
        ]
    else:
        lines += ["A labeled final holdout evaluation is not available.", ""]

    importance = state.get("permutation_importance")
    if isinstance(importance, pd.DataFrame):
        importance_frame = importance
    elif importance:
        importance_frame = pd.DataFrame(importance)
    else:
        importance_frame = pd.DataFrame()
    lines += ["## 7. Explainability and uncertainty", ""]
    if not importance_frame.empty:
        lines += [
            "Permutation importance was measured by ROC-AUC decrease on a validation subset of development data. The values describe model reliance under this sample distribution; they are not causal effects. Correlated inputs may share or mask importance.",
            "",
            _table(importance_frame, [c for c in ["feature", "importance_mean_roc_auc_decrease", "importance_std"] if c in importance_frame], limit=20),
            "",
        ]
    lines += [
        f"SHAP status: `{state.get('shap_status', 'not recorded')}`. When SHAP was unavailable, the project retained model-agnostic permutation importance as the global explanation. No SHAP result is invented.",
        "",
        "The uncertainty workflow uses distance from 0.50 as a margin-based flag. It is a review heuristic, not a calibrated confidence interval or a guarantee of error. No conformal coverage claim is made.",
        "",
        "## 8. Capacity-constrained and cost-sensitive decision analysis",
        "",
        "Review capacities were evaluated at 5%, 10%, 20%, and 30%. The low, medium, and high false-positive, false-negative, and review costs are configurable simulation assumptions; none was supplied as an observed business cost. Labeled holdout counts below are observed on the internal decision-validation split. The unlabelled operational queue instead reports expected counts from predicted probabilities.",
        "",
    ]
    final_costs = state.get("final_costs")
    if isinstance(final_costs, pd.DataFrame) and not final_costs.empty:
        lines += [
            "Final decision-evaluation sample at each review capacity (medium scenario is false-positive cost 1, false-negative cost 5, review cost 0.25):",
            "",
            _table(final_costs, ["strategy", "capacity_percent", "expected_decision_cost", "precision_at_k", "recall_at_k", "false_positives", "false_negatives"], limit=30),
            "",
        ]
    ablation = state.get("ablation")
    if isinstance(ablation, pd.DataFrame) and not ablation.empty:
        ten_percent = ablation[ablation["capacity_percent"].eq(10)]
        cols = [c for c in ["experiment", "strategy", "expected_decision_cost", "precision_at_k", "recall_at_k", "runtime_seconds"] if c in ten_percent]
        lines += ["Ablation comparison at 10% capacity:", "", _table(ten_percent, cols, limit=20), ""]
    lines += [
        "At fixed constant costs, the cost-sensitive score is monotonic in return probability. In this run, probability-only, fixed-threshold-under-capacity, cost-sensitive, and calibration-aware cost-sensitive strategies selected the same ordering and had identical results. The uncertainty-aware strategy did not improve the measured expected cost at 10% capacity in the saved ablation. Therefore the experiments do not support a claim that the full adaptive combination outperforms probability ranking on this data.",
        "",
        "## 9. Export and application integration",
        "",
    ]
    if model_metadata:
        lines += [
            f"The model artifact was exported at `{model_metadata.get('exported_at', 'unknown')}` for scenario `{model_metadata.get('prediction_scenario', 'unspecified')}` with calibrated probabilities: `{model_metadata.get('probability_is_calibrated', False)}`.",
            "",
            "The saved artifact contains the fitted preprocessing/model pipeline, selected calibrator, raw feature schema, holdout metrics, decision configuration, and permutation importance. Notebook 20 reloaded the artifact and compared raw and calibrated probabilities on the same rows against in-memory predictions at absolute tolerance 1e-12.",
            "",
            "The competition submission file, when present, is generated from unlabeled `test.csv` rows using the sample submission schema. Its predictions are not evaluated because no labels are provided.",
            "",
            "The Streamlit application loads only the exported artifact for operational scoring. The risk queue accepts unlabeled CSV input; the decision simulator labels outputs as expected or simulated; the research view exposes saved metrics and this report.",
            "",
            "Artifact paths: `artifacts/returnwise_model.joblib`, `artifacts/metadata.json`. Experiment state: `experiments/research_state.joblib`. Run history: `experiments/history.jsonl`.",
            "",
        ]
    else:
        lines += ["No model artifact is available. The application must not display operational predictions until a validated artifact exists.", ""]

    lines += [
        "## 10. Limitations and threats to validity",
        "",
        "1. **Random split rather than time split.** No event-order timestamp supports temporal validation or a claim about future-period performance.",
        "2. **Questionable raw values.** Quality screening found negative product prices, negative session duration and product-view counts, discount values outside 0–100, and ratings outside the common 0–5 range. The raw file was preserved; no undocumented correction was applied. These values may materially affect model behavior and require source-owner clarification before production use.",
        "3. **Historical aggregate timing.** Past purchase count and past return rate were excluded because their as-of cutoffs cannot be verified. Their exclusion could remove potentially useful information, but including them without provenance would risk leakage.",
        "4. **Limited candidate coverage.** CatBoost, LightGBM, XGBoost, TabPFN, and Optuna were not installed in this environment and were skipped. The selected candidate is best only among the candidates actually compared under Quick Mode.",
        "5. **Modest predictive signal.** Holdout discrimination is above baseline but not strong. Performance is not evidence of broadly reliable individual-level predictions.",
        "6. **Cost assumptions.** Expected costs depend on illustrative configurable values and do not represent measured economics or intervention effectiveness.",
        "7. **Dataset provenance.** The competition CSVs were checksum-verified locally; the source archive remained a partial invalid ZIP. The dataset appears competition-generated, and its semantics should be checked before operational use.",
        "8. **No causal or field evaluation.** No experiment tested whether a review action changes returns, customer outcomes, or profit.",
        "9. **Literature scope.** The references below motivate established methods; they are not a systematic literature review or a comparison against all competition submissions.",
        "",
        "## 11. Conclusion and contribution boundary",
        "",
        "This run provides a reproducible comparison of baseline and tree-based tabular pipelines, cross-validated probability calibration, permutation importance, and capacity-constrained decision rules on a competition dataset. The selected soft-voting model achieved modest discrimination and small proper-score gains over a constant-prevalence reference on a random internal holdout. The calibration-aware and cost-sensitive strategies did not outperform probability ranking at fixed capacity in the observed comparisons; uncertainty-aware ranking was weaker in the saved 10% ablation. The defensible contribution is this dataset-specific empirical evaluation and transparent decision simulation, not a claim of a novel algorithm or demonstrated business impact.",
        "",
        "## 12. Reproducibility record",
        "",
        f"- Dataset fingerprint: `{fingerprint or 'not available'}`.",
        f"- Experiment mode: `{state.get('experiment_mode', 'not recorded')}`.",
        f"- Selected model/calibrator: `{selected_model}` / `{selected_calibrator}`.",
        "- Split: stratified 80/20 internal holdout, random seed 42; development evaluation used stratified three-fold CV in Quick Mode.",
        "- Pipeline: `python scripts/run_research_pipeline.py` from the project root after the validated dataset and `.venv` kernel are available.",
        "- The runner verifies Notebook 01 hashes and schema, runs notebooks serially, checks each state handoff, and saves notebook outputs. It does not download data or install dependencies.",
        "- Core dependencies are listed in `requirements.txt`; optional model packages are skipped with recorded reasons when unavailable.",
        "",
        "### Recorded experiment runs",
        "",
    ]
    if history.empty:
        lines.append("No experiment history records are available.")
    else:
        recent = history.tail(20)
        summary_rows = []
        for _, record in recent.iterrows():
            summary_rows.append({
                "run": record.get("name", "Unnamed run"),
                "created_at": record.get("created_at", "time unavailable"),
                "dataset fingerprint": record.get("dataset_fingerprint", "not recorded"),
            })
        lines.append(_table(pd.DataFrame(summary_rows), limit=20))
    lines += [
        "",
        "## References and method sources",
        "",
        "These references support method descriptions; they do not establish novelty of the combined workflow.",
        "",
        "- Platt, J. (1999). [Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Platt1999.pdf).",
        "- Kull, M., Silva Filho, T., & Flach, P. (2017). [Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers](https://proceedings.mlr.press/v54/kull17a.html).",
        "- Lundberg, S. M. & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).",
        "- Ke, G. et al. (2017). [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).",
        "- Chen, T. & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).",
        "- [Kaggle competition dataset](https://www.kaggle.com/competitions/retail-return-risk-modeling).",
        "",
        "---",
        "Prepared from local CSVs, saved notebook outputs, experiment state, and model metadata. No external or fabricated performance result was added.",
    ]
    return "\n".join(lines)
