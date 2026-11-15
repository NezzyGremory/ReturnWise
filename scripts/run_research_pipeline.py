"""Execute the ReturnWise research notebooks serially with persisted outputs."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import nbformat
import psutil
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from returnwise.data.kaggle import EXPECTED_FILES, read_metadata, sha256_file, validate_competition_files
from returnwise.paths import KAGGLE_DIR
from returnwise.research_state import load_state

NOTEBOOK_DIR = ROOT / "notebooks"
KERNEL = "returnwise"
MIN_AVAILABLE_RAM_GB = 0.45
MIN_FREE_DISK_GB = 0.10
TIMEOUT_SECONDS = 3600
STATE_KEYS = {
    7: "X_train",
    8: "engineered_feature_columns",
    9: "baseline_results",
    10: "advanced_results",
    11: "tuning_result",
    12: "blend_results",
    13: "selected_model",
    14: "permutation_importance",
    15: "decision_predictions",
    16: "decision_results",
    17: "priority_sensitivity",
    18: "ablation",
    19: "final_metrics",
    20: "export_metadata",
}


def _check_resources(label: str) -> None:
    memory = psutil.virtual_memory()
    free_disk = psutil.disk_usage(str(ROOT)).free
    if memory.available < MIN_AVAILABLE_RAM_GB * 1024**3:
        raise RuntimeError(
            f"Stopped before {label}: available memory is {memory.available / 1024**3:.2f} GiB "
            f"(minimum {MIN_AVAILABLE_RAM_GB:.2f} GiB)."
        )
    if free_disk < MIN_FREE_DISK_GB * 1024**3:
        raise RuntimeError(
            f"Stopped before {label}: free disk space is {free_disk / 1024**3:.2f} GiB "
            f"(minimum {MIN_FREE_DISK_GB:.2f} GiB)."
        )


def _notebook_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return bool(code_cells) and all(
        cell.execution_count is not None
        and not any(output.output_type == "error" for output in cell.get("outputs", []))
        for cell in code_cells
    )


def _verify_dataset() -> dict:
    metadata = read_metadata()
    if not metadata or metadata.get("competition") != "retail-return-risk-modeling":
        raise RuntimeError("Verified Kaggle dataset metadata is missing or belongs to another competition.")
    files = {name: KAGGLE_DIR / name for name in EXPECTED_FILES}
    integrity = validate_competition_files(files)
    if not integrity.get("train_test_schema_match") or integrity.get("train_rows_inspected") != 200_000:
        raise RuntimeError("Dataset schema or row count differs from the verified Notebook 01 cache.")
    for name, path in files.items():
        if path.stat().st_size != int(metadata.get("file_bytes", {}).get(name, -1)):
            raise RuntimeError(f"Dataset size no longer matches Notebook 01 metadata: {name}.")
        if sha256_file(path) != metadata.get("file_sha256", {}).get(name):
            raise RuntimeError(f"Dataset SHA-256 no longer matches Notebook 01 metadata: {name}.")
    return metadata


def _verify_state(notebook_number: int, expected_fingerprint: str) -> None:
    state = load_state()
    if state.get("dataset_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Notebook {notebook_number:02d} changed or lost the verified dataset fingerprint.")
    required = STATE_KEYS.get(notebook_number)
    if required and required not in state:
        raise RuntimeError(f"Notebook {notebook_number:02d} did not persist required handoff state `{required}`.")
    if notebook_number == 9:
        results = state[required]
        if results.empty or not (results["status"] == "completed").any():
            raise RuntimeError("No baseline model completed validation; stopping the dependent pipeline.")
    if notebook_number == 10:
        results = state[required]
        if results.empty or not (results["status"] == "completed").any():
            raise RuntimeError("No advanced candidate completed validation; stopping the dependent pipeline.")


def _run_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=TIMEOUT_SECONDS,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        record_timing=True,
    )
    started = time.perf_counter()
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, path)
        raise
    nbformat.write(notebook, path)
    elapsed = time.perf_counter() - started
    failed_cells = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(output.output_type == "error" for output in cell.get("outputs", []))
    ]
    if failed_cells:
        raise RuntimeError(f"Notebook {path.name} has recorded errors in code cells {failed_cells}.")
    print(f"Completed {path.name} in {elapsed:.1f}s; outputs saved.", flush=True)


def main() -> int:
    # Keep native numeric/model libraries single-threaded for this CPU-first run.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    ipython_dir = Path(tempfile.gettempdir()) / "returnwise_ipython"
    jupyter_dir = Path(tempfile.gettempdir()) / "returnwise_jupyter"
    ipython_dir.mkdir(parents=True, exist_ok=True)
    (jupyter_dir / "config").mkdir(parents=True, exist_ok=True)
    (jupyter_dir / "runtime").mkdir(parents=True, exist_ok=True)
    os.environ["IPYTHONDIR"] = str(ipython_dir)
    os.environ["JUPYTER_CONFIG_DIR"] = str(jupyter_dir / "config")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(jupyter_dir / "runtime")

    metadata = _verify_dataset()
    fingerprint = metadata["fingerprint_sha256"]
    for number in (0, 1):
        path = next(NOTEBOOK_DIR.glob(f"{number:02d}_*.ipynb"), None)
        if path is None or not _notebook_is_complete(path):
            raise RuntimeError(f"Notebook {number:02d} is not recorded as successfully executed; audit stopped before research notebooks.")
        print(f"Verified and skipped completed {path.name}.", flush=True)

    archive = metadata.get("archive_validation", {})
    if archive.get("present") and not archive.get("valid"):
        print("Using the independently verified CSV files; the incomplete ZIP is not extracted.", flush=True)

    state = load_state()
    if state.get("dataset_fingerprint") != fingerprint:
        raise RuntimeError("Notebook 01 state does not match the verified dataset fingerprint.")
    if state.get("experiment_mode") != "quick":
        raise RuntimeError("The verified Notebook 00 mode is not Quick; inspect the saved mode before training.")

    notebooks = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))
    for path in notebooks:
        number = int(path.name[:2])
        if number < 2:
            continue
        if _notebook_is_complete(path):
            print(f"Verified and skipped completed {path.name}.", flush=True)
            _verify_state(number, fingerprint)
            continue
        _check_resources(path.name)
        print(f"Running {path.name} serially (Quick mode; one native thread).", flush=True)
        _run_notebook(path)
        _verify_state(number, fingerprint)
        _check_resources(f"the next notebook after {path.name}")
    print("All notebooks 00–20 completed and saved.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = re.sub(r"https?://\S+", "[URL redacted]", str(exc))
        print(f"PIPELINE STOPPED: {type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        raise SystemExit(1)
