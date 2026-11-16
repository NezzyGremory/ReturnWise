"""SQLite schema and connection helpers for the local Smart Retail mode."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "smart_retail.sqlite3"
DB_PATH = Path(os.environ.get("SMART_RETAIL_DB", DEFAULT_DB_PATH)).expanduser()

SCHEMA = """
CREATE TABLE IF NOT EXISTS stores (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    full_name TEXT NOT NULL,
    email TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, email)
);
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, name)
);
CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    name TEXT NOT NULL,
    contact TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    lead_time_days INTEGER NOT NULL DEFAULT 7 CHECK(lead_time_days >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, name)
);
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    name TEXT NOT NULL,
    phone TEXT,
    email TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, phone)
);
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    sku TEXT NOT NULL CHECK(length(trim(sku)) > 0),
    barcode TEXT,
    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    description TEXT NOT NULL DEFAULT '',
    cost_price INTEGER NOT NULL CHECK(cost_price >= 0),
    selling_price INTEGER NOT NULL CHECK(selling_price >= 0),
    stock INTEGER NOT NULL DEFAULT 0 CHECK(stock >= 0),
    minimum_stock INTEGER NOT NULL DEFAULT 0 CHECK(minimum_stock >= 0),
    unit TEXT NOT NULL DEFAULT 'pcs',
    supplier_id INTEGER REFERENCES suppliers(id) ON DELETE SET NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, sku),
    UNIQUE(store_id, barcode)
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    transaction_no TEXT NOT NULL UNIQUE,
    idempotency_key TEXT UNIQUE,
    customer_id INTEGER REFERENCES customers(id) ON DELETE SET NULL,
    subtotal INTEGER NOT NULL CHECK(subtotal >= 0),
    discount_amount INTEGER NOT NULL DEFAULT 0 CHECK(discount_amount >= 0),
    tax_rate REAL NOT NULL DEFAULT 0 CHECK(tax_rate >= 0),
    tax_amount INTEGER NOT NULL DEFAULT 0 CHECK(tax_amount >= 0),
    total INTEGER NOT NULL CHECK(total >= 0),
    amount_paid INTEGER NOT NULL CHECK(amount_paid >= 0),
    change_amount INTEGER NOT NULL DEFAULT 0 CHECK(change_amount >= 0),
    payment_method TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed' CHECK(status IN ('completed', 'cancelled', 'refunded')),
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(store_id, transaction_no)
);
CREATE TABLE IF NOT EXISTS transaction_items (
    id INTEGER PRIMARY KEY,
    transaction_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE RESTRICT,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    sku_snapshot TEXT NOT NULL,
    product_name_snapshot TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price INTEGER NOT NULL CHECK(unit_price >= 0),
    unit_cost INTEGER NOT NULL CHECK(unit_cost >= 0),
    line_total INTEGER NOT NULL CHECK(line_total >= 0)
);
CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY,
    transaction_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE RESTRICT,
    method TEXT NOT NULL,
    amount INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'capture' CHECK(kind IN ('capture', 'refund')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stock_receipts (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    supplier_id INTEGER REFERENCES suppliers(id) ON DELETE SET NULL,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    receipt_no TEXT NOT NULL UNIQUE,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_cost INTEGER NOT NULL CHECK(unit_cost >= 0),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inventory_movements (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    transaction_id INTEGER REFERENCES transactions(id) ON DELETE RESTRICT,
    receipt_id INTEGER REFERENCES stock_receipts(id) ON DELETE RESTRICT,
    movement_type TEXT NOT NULL CHECK(movement_type IN ('sale', 'restock', 'adjustment', 'refund', 'cancellation')),
    quantity_delta INTEGER NOT NULL,
    stock_after INTEGER NOT NULL CHECK(stock_after >= 0),
    unit_cost INTEGER NOT NULL DEFAULT 0 CHECK(unit_cost >= 0),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stock_adjustments (
    id INTEGER PRIMARY KEY,
    movement_id INTEGER NOT NULL UNIQUE REFERENCES inventory_movements(id) ON DELETE RESTRICT,
    stock_before INTEGER NOT NULL,
    stock_after INTEGER NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    scope_key TEXT NOT NULL DEFAULT 'store',
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(store_id, model_name, model_version, scope_key)
);
CREATE TABLE IF NOT EXISTS sales_forecasts (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    product_id INTEGER REFERENCES products(id) ON DELETE SET NULL,
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    model_id INTEGER NOT NULL REFERENCES model_registry(id),
    forecast_date TEXT NOT NULL,
    predicted_units REAL NOT NULL CHECK(predicted_units >= 0),
    predicted_revenue REAL NOT NULL CHECK(predicted_revenue >= 0),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    model_id INTEGER NOT NULL REFERENCES model_registry(id),
    product_id INTEGER REFERENCES products(id) ON DELETE SET NULL,
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    prediction_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY,
    store_id INTEGER NOT NULL REFERENCES stores(id),
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_settings (
    store_id INTEGER NOT NULL REFERENCES stores(id),
    setting_key TEXT NOT NULL,
    setting_value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(store_id, setting_key)
);
CREATE INDEX IF NOT EXISTS idx_products_store_active ON products(store_id, is_active, name);
CREATE INDEX IF NOT EXISTS idx_transactions_store_created ON transactions(store_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status, created_at);
CREATE INDEX IF NOT EXISTS idx_items_transaction ON transaction_items(transaction_id);
CREATE INDEX IF NOT EXISTS idx_items_product ON transaction_items(product_id);
CREATE INDEX IF NOT EXISTS idx_movements_product_created ON inventory_movements(product_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_customers_store_name ON customers(store_id, name);
CREATE INDEX IF NOT EXISTS idx_forecasts_store_date ON sales_forecasts(store_id, forecast_date);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO stores(id, name) VALUES (1, 'Toko Utama')"
        )
        connection.execute(
            """INSERT OR IGNORE INTO app_settings(store_id, setting_key, setting_value, updated_at)
               VALUES (1, 'retail_started_at', ?, ?)""",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


@contextmanager
def atomic(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
