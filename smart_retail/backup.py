"""Versioned JSON backup and validated restore for the local SQLite store."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from smart_retail.db import SCHEMA, atomic, connect
from smart_retail.services import RetailError, STORE_ID


BACKUP_VERSION = 1
MAX_BACKUP_BYTES = 50 * 1024 * 1024
TABLES = (
    "stores", "users", "categories", "suppliers", "customers", "products",
    "transactions", "transaction_items", "payments", "stock_receipts",
    "inventory_movements", "stock_adjustments", "model_registry",
    "sales_forecasts", "prediction_history", "audit_logs", "app_settings",
)
DELETE_ORDER = tuple(reversed(TABLES))


def create_backup() -> bytes:
    with connect() as connection:
        # Keep every table on one SQLite snapshot if a checkout commits during export.
        connection.execute("BEGIN")
        tables = {}
        for table in TABLES:
            rows = connection.execute(f"SELECT * FROM {table}").fetchall()
            tables[table] = [dict(row) for row in rows]
        connection.commit()
    document = {
        "format": "smart-retail-local-backup",
        "schema_version": BACKUP_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables": tables,
    }
    payload = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
    if len(payload) > MAX_BACKUP_BYTES:
        raise RetailError("Backup melebihi batas 50 MB. Gunakan salinan database saat aplikasi berhenti atau minta bantuan administrator.")
    return payload


def parse_backup(payload: bytes | str) -> dict[str, Any]:
    try:
        payload_bytes = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RetailError("File backup bukan JSON UTF-8 yang valid.") from exc
    if len(payload_bytes) > MAX_BACKUP_BYTES:
        raise RetailError("Ukuran backup melebihi batas 50 MB.")
    try:
        document = json.loads(payload_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetailError("File backup bukan JSON UTF-8 yang valid.") from exc
    if not isinstance(document, dict) or document.get("format") != "smart-retail-local-backup":
        raise RetailError("Format file bukan backup Smart Retail.")
    if document.get("schema_version") != BACKUP_VERSION:
        raise RetailError("Versi struktur backup tidak didukung aplikasi ini.")
    tables = document.get("tables")
    if not isinstance(tables, dict) or set(tables) != set(TABLES):
        raise RetailError("Backup tidak berisi seluruh tabel yang diperlukan.")
    for name in TABLES:
        if not isinstance(tables[name], list) or any(not isinstance(row, dict) for row in tables[name]):
            raise RetailError(f"Isi tabel {name} tidak valid.")
    stores = tables["stores"]
    if len(stores) != 1 or stores[0].get("id") != STORE_ID:
        raise RetailError("Backup harus berisi satu toko lokal dengan ID yang valid.")
    _validate_in_memory(tables)
    return document


def _insert_rows(connection: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    for row in rows:
        if not set(row).issubset(columns):
            raise RetailError(f"Backup memiliki kolom yang tidak dikenal pada tabel {table}.")
        names = list(row)
        quoted = ", ".join(f'"{name}"' for name in names)
        placeholders = ", ".join("?" for _ in names)
        connection.execute(
            f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})',
            [row[name] for name in names],
        )


def _validate_in_memory(tables: dict[str, list[dict[str, Any]]]) -> None:
    candidate = sqlite3.connect(":memory:")
    candidate.row_factory = sqlite3.Row
    candidate.execute("PRAGMA foreign_keys=ON")
    try:
        candidate.executescript(SCHEMA)
        for table in DELETE_ORDER:
            candidate.execute(f'DELETE FROM "{table}"')
        for table in TABLES:
            _insert_rows(candidate, table, tables[table])
        broken = candidate.execute("PRAGMA foreign_key_check").fetchone()
        if broken:
            raise RetailError("Backup memiliki relasi yang tidak valid.")
    except sqlite3.Error as exc:
        raise RetailError("Backup melanggar aturan data, misalnya ID duplikat atau relasi yang hilang.") from exc
    except (TypeError, OverflowError, ValueError) as exc:
        raise RetailError("Nilai di dalam backup tidak dapat dipulihkan.") from exc
    finally:
        candidate.close()


def restore_backup(payload: bytes | str) -> dict[str, int]:
    document = parse_backup(payload)
    tables = document["tables"]
    with connect() as connection, atomic(connection):
        for table in DELETE_ORDER:
            connection.execute(f'DELETE FROM "{table}"')
        for table in TABLES:
            _insert_rows(connection, table, tables[table])
        broken = connection.execute("PRAGMA foreign_key_check").fetchone()
        if broken:
            raise RetailError("Pemulihan dibatalkan karena relasi data tidak valid.")
        connection.execute(
            "INSERT INTO audit_logs(store_id, action, entity_type, entity_id, details_json, created_at) VALUES (?, 'restore', 'backup', 'local', ?, ?)",
            (STORE_ID, json.dumps({"backup_created_at": document.get("created_at")}, ensure_ascii=False), datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    counts = {table: len(tables[table]) for table in TABLES}
    counts["audit_logs"] += 1
    return counts

