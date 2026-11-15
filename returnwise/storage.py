"""Local data selection and model artifact loading for the application."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from returnwise.data.kaggle import METADATA_FILE, read_metadata, sha256_file
from returnwise.paths import ARTIFACT_DIR, KAGGLE_DIR, MANUAL_DIR

ACTIVE_FILE = MANUAL_DIR / "active_dataset.json"
MODEL_FILE = ARTIFACT_DIR / "returnwise_model.joblib"


def save_manual_csv(name: str, content: bytes) -> dict:
    safe_name = Path(name).name
    if not safe_name.lower().endswith(".csv"):
        raise ValueError("Manual import accepts CSV files only.")
    destination = MANUAL_DIR / safe_name
    destination.write_bytes(content)
    frame = pd.read_csv(destination)
    if frame.empty:
        destination.unlink(missing_ok=True)
        raise ValueError("The uploaded CSV contains no data rows.")
    manifest = {
        "source": "manual",
        "file": str(destination.relative_to(MANUAL_DIR.parent.parent)),
        "original_name": safe_name,
        "rows": len(frame),
        "columns": list(frame.columns),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "has_target": "returned" in frame.columns,
    }
    ACTIVE_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_active_manifest() -> dict | None:
    try:
        return json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def use_kaggle_training_data() -> dict:
    path = KAGGLE_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError("No validated Kaggle train.csv is cached. Synchronize it first.")
    frame = pd.read_csv(path)
    manifest = {
        "source": "kaggle",
        "file": "data/kaggle/train.csv",
        "rows": len(frame),
        "columns": list(frame.columns),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": (read_metadata() or {}).get("dataset_version"),
        "has_target": "returned" in frame.columns,
    }
    ACTIVE_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_active_dataset() -> tuple[pd.DataFrame | None, dict | None]:
    manifest = read_active_manifest()
    if not manifest:
        return None, None
    if manifest.get("source") == "manual":
        path = MANUAL_DIR / Path(manifest.get("file", "")).name
    else:
        path = KAGGLE_DIR / "train.csv"
    if not path.is_file():
        return None, manifest
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        manifest["load_error"] = str(exc)
        return None, manifest
    return frame, manifest


def load_model_artifact() -> dict | None:
    if not MODEL_FILE.is_file():
        return None
    try:
        payload = joblib.load(MODEL_FILE)
        if not isinstance(payload, dict) or "model" not in payload or "feature_columns" not in payload:
            return None
        return payload
    except (OSError, EOFError, ValueError, ImportError):
        return None


def active_dataset_fingerprint() -> str | None:
    manifest = read_active_manifest() or {}
    if manifest.get("source") == "kaggle":
        return (read_metadata() or {}).get("fingerprint_sha256")
    if manifest.get("source") == "manual":
        path = MANUAL_DIR / Path(manifest.get("original_name", "")).name
        if path.is_file():
            return sha256_file(path)
    return None
