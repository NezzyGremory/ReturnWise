"""Safe Kaggle CLI integration for the ReturnWise competition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from returnwise.paths import KAGGLE_DIR

COMPETITION = "retail-return-risk-modeling"
EXPECTED_FILES = ("train.csv", "test.csv", "sample_submission.csv")
METADATA_FILE = "dataset_metadata.json"


@dataclass
class KaggleResult:
    ok: bool
    message: str
    command: str | None = None
    details: str | None = None
    files: list[str] | None = None
    metadata: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cli_path() -> str | None:
    return shutil.which("kaggle")


def _redact(text: str) -> str:
    """Remove common token-shaped strings before surfacing CLI diagnostics."""
    text = re.sub(
        r"https?://storage\.googleapis\.com/[^\s\"']+",
        "[redacted signed download URL]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?i)(token|key|password)(\s*[:=]\s*)\S+", r"\1\2[redacted]", text)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    return text[-2500:]


def run_cli(args: list[str], timeout: int = 90) -> KaggleResult:
    executable = _cli_path()
    if not executable:
        return KaggleResult(
            False,
            "Kaggle CLI is not installed in this Python environment.",
            details="Install it with `python -m pip install kaggle`, then restart the app.",
        )
    command = [executable, *args]
    try:
        result = subprocess.run(
            command,
            cwd=KAGGLE_DIR.parent.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return KaggleResult(False, "Kaggle CLI timed out.", " ".join(command), "Check network access and retry.")
    except OSError as exc:
        return KaggleResult(
            False,
            "Kaggle CLI could not start in this environment.",
            " ".join(command),
            _redact(str(exc)),
        )

    stdout = _redact((result.stdout or "").strip())
    stderr = _redact((result.stderr or "").strip())
    if result.returncode == 0:
        return KaggleResult(True, "Kaggle CLI command completed.", " ".join(command), stdout or stderr)
    output = "\n".join(part for part in (stdout, stderr) if part)

    low = output.lower()
    if any(word in low for word in ("401", "unauthorized", "authentication", "credentials", "login")):
        message = "Kaggle authentication is missing or expired."
        detail = "Run `kaggle auth login` in a terminal, then use Check access again. Do not paste credentials into ReturnWise or chat."
    elif any(word in low for word in ("403", "forbidden", "competition rules", "accept the rules", "access denied")):
        message = "Kaggle account does not have competition access yet."
        detail = f"Open https://www.kaggle.com/competitions/{COMPETITION}, sign in, accept its rules if required, and retry."
    elif any(word in low for word in ("winerror 5", "permissionerror", ".kaggle")):
        message = "Kaggle CLI could not access its local configuration directory."
        detail = "Allow the current user to access the configured Kaggle directory, or set KAGGLE_CONFIG_DIR to a writable local directory before starting ReturnWise. Then run `kaggle auth login`."
    elif "404" in low or "not found" in low:
        message = "The competition could not be found for this account."
        detail = f"Verify the competition slug and account access at https://www.kaggle.com/competitions/{COMPETITION}. ReturnWise will not substitute another dataset."
    else:
        message = "Kaggle request failed."
        detail = output or "No diagnostic text was returned by the CLI."
    return KaggleResult(False, message, " ".join(command), detail)


def cli_version() -> KaggleResult:
    result = run_cli(["--version"], timeout=15)
    if result.ok:
        result.message = result.details or "Kaggle CLI is available."
    return result


def list_competition_files() -> KaggleResult:
    """Check CLI auth/access and discover competition file names."""
    result = run_cli(["competitions", "files", COMPETITION, "--format", "json", "--quiet"], timeout=60)
    if not result.ok:
        return result
    names: list[str] = []
    file_bytes: dict[str, int] = {}
    if result.details:
        try:
            payload = json.loads(result.details)
            if isinstance(payload, dict):
                payload = payload.get("files", payload.get("data", []))
            if isinstance(payload, list):
                for item in payload:
                    if not isinstance(item, dict) or not item.get("name"):
                        continue
                    name = str(item["name"]).replace("\\", "/").split("/")[-1]
                    names.append(name)
                    try:
                        if item.get("size") is not None:
                            file_bytes[name] = int(item["size"])
                    except (TypeError, ValueError):
                        continue
        except (TypeError, ValueError, json.JSONDecodeError):
            names = []
    result.files = names
    result.metadata = {"file_bytes": file_bytes} if file_bytes else {}
    result.message = "Competition access verified." if names else "Competition listing returned no recognizable CSV filenames."
    if names:
        missing = sorted(set(EXPECTED_FILES) - set(names))
        if missing:
            result.message = "Competition is accessible, but expected files are missing: " + ", ".join(missing)
            result.ok = False
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe archive path rejected: {member.filename}")
        archive.extractall(destination)


def _inspect_archive(zip_path: Path) -> dict[str, Any]:
    """Check the ZIP central directory, CRCs, and expected CSV members without extracting."""
    marker = Path(str(zip_path) + ".kaggle-partial")
    result: dict[str, Any] = {
        "path": str(zip_path.relative_to(KAGGLE_DIR.parent.parent)) if zip_path.exists() else None,
        "present": zip_path.is_file(),
        "valid": False,
        "expected_members_present": False,
        "partial_marker_present": marker.is_file(),
        "partial_marker_bytes": marker.stat().st_size if marker.is_file() else None,
    }
    if not zip_path.is_file():
        result["error_type"] = "FileNotFoundError"
        return result
    result["size_bytes"] = zip_path.stat().st_size
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            member_names = {Path(member.filename.replace("\\", "/")).name for member in archive.infolist()}
            missing = sorted(set(EXPECTED_FILES) - member_names)
            result.update(
                {
                    "valid": bad_member is None and not missing,
                    "zip_crc_valid": bad_member is None,
                    "expected_members_present": not missing,
                    "missing_members": missing,
                    "bad_member": bad_member,
                    "member_count": len(archive.infolist()),
                }
            )
    except (OSError, zipfile.BadZipFile) as exc:
        result["error_type"] = type(exc).__name__
    return result


def _locate_expected_files(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for filename in EXPECTED_FILES:
        matches = list(root.rglob(filename))
        if matches:
            found[filename] = matches[0]
    return found


def validate_competition_files(files: dict[str, Path]) -> dict[str, Any]:
    missing = sorted(set(EXPECTED_FILES) - set(files))
    if missing:
        raise ValueError("Required competition files are missing: " + ", ".join(missing))
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{name} is empty or unreadable.")
    def scan_csv(path: Path, target_column: str | None = None) -> tuple[int, list[str], set[Any]]:
        rows = 0
        columns: list[str] | None = None
        target_values: set[Any] = set()
        try:
            for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
                if columns is None:
                    columns = list(chunk.columns)
                rows += len(chunk)
                if target_column and target_column in chunk.columns:
                    values = pd.to_numeric(chunk[target_column], errors="coerce").dropna().unique()
                    target_values.update(values.tolist())
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path.name} has no CSV header.") from exc
        if columns is None:
            columns = list(pd.read_csv(path, nrows=0).columns)
        if rows == 0:
            raise ValueError(f"{path.name} contains no data rows.")
        return rows, columns, target_values

    train_rows, train_columns, target_values = scan_csv(files["train.csv"], "returned")
    test_rows, test_columns, _ = scan_csv(files["test.csv"])
    sample_rows, sample_columns, _ = scan_csv(files["sample_submission.csv"])
    if "returned" not in train_columns:
        raise ValueError("train.csv has no `returned` target column.")
    if "returned" in test_columns:
        raise ValueError("test.csv unexpectedly contains `returned`; competition test labels must not be used for evaluation.")
    if not target_values or not target_values.issubset({0, 1}):
        raise ValueError(f"The `returned` target is not binary: {sorted(map(str, target_values))[:10]}")
    schema_match = [c for c in train_columns if c != "returned"] == test_columns
    if not schema_match:
        raise ValueError("train.csv features and test.csv columns do not match in name and order.")
    if len(sample_columns) < 2:
        raise ValueError("sample_submission.csv must have an identifier and a prediction column.")
    return {
        "validation_scope": "full_csv_stream",
        "train_rows_inspected": train_rows,
        "test_rows_inspected": test_rows,
        "sample_rows_inspected": sample_rows,
        "target_column": "returned",
        "train_columns": train_columns,
        "test_columns": test_columns,
        "sample_columns": sample_columns,
        "train_test_schema_match": schema_match,
        "target_values": sorted(map(str, target_values)),
    }


def read_metadata() -> dict[str, Any] | None:
    path = KAGGLE_DIR / METADATA_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sync_competition(force: bool = False) -> KaggleResult:
    """Discover, download, safely extract, validate, fingerprint, and cache data."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    existing = _locate_expected_files(KAGGLE_DIR)
    metadata = read_metadata()
    if not force and len(existing) == len(EXPECTED_FILES) and metadata:
        try:
            checked = datetime.fromisoformat(metadata["last_checked_at"])
            age_days = (datetime.now(timezone.utc) - checked.astimezone(timezone.utc)).days
        except (KeyError, ValueError, TypeError):
            age_days = 8
        if age_days < 7:
            try:
                cache_matches = all(
                    existing[name].stat().st_size == metadata["file_bytes"][name]
                    and sha256_file(existing[name]) == metadata["file_sha256"][name]
                    for name in EXPECTED_FILES
                )
            except (KeyError, OSError, TypeError):
                cache_matches = False
            if cache_matches:
                return KaggleResult(
                    True,
                    "Using the hash-verified Kaggle dataset cache (checked within the last 7 days).",
                    metadata=metadata,
                    files=metadata.get("listing_names"),
                )

    listing = list_competition_files()
    if not listing.ok:
        return listing

    remote_sizes = (listing.metadata or {}).get("file_bytes", {})
    remote_sizes_complete = all(name in remote_sizes for name in EXPECTED_FILES)
    local_sizes_match = (
        len(existing) == len(EXPECTED_FILES)
        and remote_sizes_complete
        and all(existing[name].stat().st_size == remote_sizes[name] for name in EXPECTED_FILES)
    )

    # Adopt a complete local Kaggle cache after checking it against the live
    # listing. This avoids downloading a second copy when only metadata is absent.
    if not force and local_sizes_match:
        try:
            integrity = validate_competition_files(existing)
            file_hashes = {name: sha256_file(existing[name]) for name in EXPECTED_FILES}
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            integrity = None
            cache_error = f"Local CSV cache did not pass validation ({type(exc).__name__})."
        else:
            archive_path = KAGGLE_DIR / f"{COMPETITION}.zip"
            archive_validation = _inspect_archive(archive_path)
            archive_validation["extraction_performed"] = False
            if archive_validation["present"] and not archive_validation["valid"]:
                archive_validation["action"] = "Ignored incomplete archive; CSVs were independently verified."
            fingerprints = hashlib.sha256("".join(file_hashes[n] for n in EXPECTED_FILES).encode()).hexdigest()
            now = _now()
            verified_metadata = {
                "competition": COMPETITION,
                "dataset_version": fingerprints[:16],
                "fingerprint_sha256": fingerprints,
                "file_sha256": file_hashes,
                "file_bytes": {name: existing[name].stat().st_size for name in EXPECTED_FILES},
                "file_mtime_ns": {name: existing[name].stat().st_mtime_ns for name in EXPECTED_FILES},
                "listing_file_bytes": remote_sizes,
                "listing_names": listing.files,
                "validated_existing_cache": True,
                "cache_validated_at": now,
                "last_checked_at": now,
                "integrity": integrity,
                "archive_validation": archive_validation,
                "download_command": None,
            }
            (KAGGLE_DIR / METADATA_FILE).write_text(json.dumps(verified_metadata, indent=2), encoding="utf-8")
            return KaggleResult(
                True,
                "Existing Kaggle CSVs match the official file sizes and passed full-file schema validation; no download was needed.",
                command=listing.command,
                details=(
                    "The existing archive was incomplete and was left untouched. "
                    "The three CSV files were validated independently; no extraction was performed."
                    if archive_validation["present"] and not archive_validation["valid"]
                    else None
                ),
                files=listing.files,
                metadata=verified_metadata,
            )

    download_args = ["competitions", "download", COMPETITION, "-p", str(KAGGLE_DIR)]
    if force:
        download_args.append("-o")
    # Kaggle CLI 2.2.x retries network failures and resumes only when the
    # partial marker matches the current remote object. Leave enough time for
    # that bounded retry policy to finish; do not add a second retry loop here.
    download = run_cli(download_args, timeout=1_900)
    if not download.ok:
        return download
    archives = sorted(KAGGLE_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        return KaggleResult(False, "Kaggle reported success but no ZIP archive was found.", download.command)
    archive_path = archives[0]
    archive_validation = _inspect_archive(archive_path)
    if not archive_validation["valid"] or not archive_validation["expected_members_present"]:
        return KaggleResult(
            False,
            "Downloaded Kaggle archive failed ZIP integrity or required-member validation.",
            download.command,
            json.dumps(archive_validation, ensure_ascii=False),
        )
    archive_validation["sha256"] = sha256_file(archive_path)
    try:
        _safe_extract(archive_path, KAGGLE_DIR / "extracted")
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        return KaggleResult(False, "The Kaggle archive could not be extracted safely.", download.command, str(exc))
    archive_validation["extraction_performed"] = True
    extracted = _locate_expected_files(KAGGLE_DIR / "extracted")
    root_files = _locate_expected_files(KAGGLE_DIR)
    for name in EXPECTED_FILES:
        if name not in extracted and name in root_files:
            extracted[name] = root_files[name]
    missing = [name for name in EXPECTED_FILES if name not in extracted]
    if missing:
        return KaggleResult(False, "Downloaded archive is missing expected competition files.", download.command, ", ".join(missing))

    canonical: dict[str, Path] = {}
    for name, source in extracted.items():
        target = KAGGLE_DIR / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        canonical[name] = target
    try:
        integrity = validate_competition_files(canonical)
        if remote_sizes_complete and any(canonical[name].stat().st_size != remote_sizes[name] for name in EXPECTED_FILES):
            raise ValueError("Extracted CSV byte sizes do not match Kaggle's current file listing.")
        file_hashes = {name: sha256_file(path) for name, path in canonical.items()}
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return KaggleResult(False, "Downloaded data failed integrity validation.", download.command, str(exc))

    fingerprints = hashlib.sha256("".join(file_hashes[n] for n in EXPECTED_FILES).encode()).hexdigest()
    metadata = {
        "competition": COMPETITION,
        "dataset_version": fingerprints[:16],
        "fingerprint_sha256": fingerprints,
        "file_sha256": file_hashes,
        "file_bytes": {name: path.stat().st_size for name, path in canonical.items()},
        "file_mtime_ns": {name: path.stat().st_mtime_ns for name, path in canonical.items()},
        "listing_file_bytes": remote_sizes,
        "listing_names": listing.files,
        "last_synced_at": _now(),
        "last_checked_at": _now(),
        "integrity": integrity,
        "archive_validation": archive_validation,
        "download_command": "kaggle competitions download -c retail-return-risk-modeling -p data/kaggle",
    }
    (KAGGLE_DIR / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return KaggleResult(True, "Competition dataset ZIP and extracted CSV files passed integrity and schema validation.", metadata=metadata, files=listing.files)
