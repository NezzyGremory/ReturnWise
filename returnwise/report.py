"""Factual Markdown report generation from dataset profiles and saved results."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from returnwise.data.profile import quality_report
from returnwise.experiments import read_experiment_history
from returnwise.storage import active_dataset_fingerprint


def generate_research_report(df: pd.DataFrame | None, model_metadata: dict | None = None) -> str:
    lines = [
        "# ReturnWise research report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "This report contains only dataset and experiment results available in the local workspace. It does not claim causal impact or business savings.",
        "",
    ]
    if df is None:
        lines += ["## Data availability", "", "No active dataset is selected. Import a CSV or synchronize the Kaggle competition files before running research analyses.", ""]
    else:
        quality = quality_report(df)
        lines += ["## Dataset profile", "", f"- Rows: {quality['rows']:,}", f"- Columns: {quality['columns']:,}", f"- Exact duplicate rows: {quality['duplicate_rows']:,} ({quality['duplicate_rows_percent']:.2f}%)"]
        if "returned" in df:
            rates = df["returned"].value_counts(normalize=True, dropna=False).to_dict()
            lines.append(f"- Observed target distribution: `{rates}`")
        else:
            lines.append("- Target `returned` is not present; this dataset cannot support labeled evaluation.")
        lines.append("")
        fingerprint = active_dataset_fingerprint()
        lines.append(f"- Dataset SHA-256 fingerprint: `{fingerprint or 'not available'}`")
        lines.append("")
    history = read_experiment_history()
    lines += ["## Experiment history", ""]
    if history.empty:
        lines.append("No experiment runs have been recorded yet.")
    else:
        for _, record in history.iterrows():
            lines.append(f"- {record.get('name', 'Unnamed run')} — {record.get('created_at', 'time unavailable')}; dataset fingerprint: {record.get('dataset_fingerprint', 'not recorded')}")
    lines.append("")
    if model_metadata:
        lines += ["## Exported model", "", f"- Model: {model_metadata.get('selected_model', 'unknown')}", f"- Prediction scenario: {model_metadata.get('prediction_scenario', 'unspecified')}", f"- Export time: {model_metadata.get('exported_at', 'unknown')}", ""]
        metrics = model_metadata.get("validation_metrics") or {}
        if metrics:
            lines += ["Validation metrics:", "", "```", pd.Series(metrics).to_string(), "```", ""]
    else:
        lines += ["## Exported model", "", "No trained model artifact is available. Prediction results are not reported.", ""]
    lines += ["## Limitations", "", "- The competition test split has no labels and is not used for labeled evaluation.", "- A post-shipment variable such as delivery delay is excluded from pre-shipment prediction.", "- Customer history aggregates require documented time cutoffs before they can be treated as leakage-safe.", "- Cost scenarios are user-specified simulations, not observed business costs.", "- Predictive associations and feature explanations are not causal evidence.", ""]
    return "\n".join(lines)
