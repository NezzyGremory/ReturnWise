"""Shared display helpers; business logic remains in the research package."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import pandas as pd

from returnwise.paths import KAGGLE_DIR, MANUAL_DIR
from returnwise.storage import load_model_artifact


def active_data():
    manifest = load_active_manifest()
    if manifest is None:
        return None, None
    path = MANUAL_DIR / Path(manifest.get("file", "")).name if manifest.get("source") == "manual" else KAGGLE_DIR / "train.csv"
    if not path.exists():
        return None, manifest
    frame = _read_dataset(str(path), path.stat().st_mtime_ns)
    return frame, manifest


@st.cache_data(ttl=300, max_entries=4)
def _read_dataset(path: str, modified_ns: int):
    return pd.read_csv(path)


def load_active_manifest():
    from returnwise.storage import read_active_manifest

    return read_active_manifest()


def app_status():
    frame, manifest = active_data()
    artifact = load_model_artifact()
    with st.container(border=True):
        st.caption("Workspace status")
        data_label = "No active dataset" if manifest is None else f"{manifest.get('source', 'unknown').title()} dataset"
        model_label = artifact.get("metadata", {}).get("selected_model", "No model artifact") if artifact else "No model artifact"
        st.write(f"**Data:** {data_label}  ·  **Model:** {model_label}")
        if frame is not None:
            st.caption(f"{len(frame):,} rows · {len(frame.columns):,} columns")
        elif manifest and manifest.get("load_error"):
            st.error(manifest["load_error"])


def require_dataset():
    frame, manifest = active_data()
    if frame is None:
        st.info("Import a CSV or synchronize the Kaggle competition data to begin.", title="No dataset selected")
        st.page_link("app_pages/02_data.py", label="Open dataset setup", icon=":material/database:")
        return None, None
    return frame, manifest
