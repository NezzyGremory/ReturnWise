"""Persistent, local-only handoff between sequential notebook kernels."""

from __future__ import annotations

import joblib

from returnwise.paths import EXPERIMENT_DIR

STATE_FILE = EXPERIMENT_DIR / "research_state.joblib"


def load_state() -> dict:
    if not STATE_FILE.is_file():
        return {}
    try:
        value = joblib.load(STATE_FILE)
        return value if isinstance(value, dict) else {}
    except (OSError, EOFError, ValueError, ImportError):
        return {}


def save_state(state: dict) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, STATE_FILE, compress=3)
