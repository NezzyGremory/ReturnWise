from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
MANUAL_DIR = DATA_DIR / "manual"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports" / "generated"
EXPERIMENT_DIR = ROOT / "experiments"

for _path in (KAGGLE_DIR, MANUAL_DIR, ARTIFACT_DIR, REPORT_DIR, EXPERIMENT_DIR):
    _path.mkdir(parents=True, exist_ok=True)
