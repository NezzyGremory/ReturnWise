"""Transactional retail operations. Presentation code should call these functions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
import secrets
import sqlite3
from typing import Any, Iterable

from smart_retail.db import atomic, connect
from smart_retail.timeutils import utc_date_bounds


STORE_ID = 1
PAYMENT_METHODS = ("Tunai", "Transfer", "QRIS", "Debit/kredit", "Lainnya")


class RetailError(ValueError):
    """A user-correctable retail operation error."""


@contextmanager
def _checkout_atomic(connection: sqlite3.Connection):
    try:
        with atomic(connection):
            yield connection
    except sqlite3.Error as exc:
        raise RetailError("Transaksi gagal disimpan. Stok dan riwayat transaksi tidak berubah.") from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int(value: Any, label: str, minimum: int = 0) -> int:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise RetailError(f"{label} harus berupa angka.") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise RetailError(f"{label} harus berupa bilangan bulat.")
    number = int(parsed)
    if number < minimum:
        raise RetailError(f"{label} minimal {minimum}.")
    if number > 9_223_372_036_854_775_807:
        raise RetailError(f"{label} melebihi batas penyimpanan.")
    return number


def _text(value: Any, label: str, *, required: bool = False) -> str:
    result = "" if value is None else str(value).strip()
    if required and not result:
        raise RetailError(f"{label} wajib diisi.")
    return result


def _audit(
    connection: sqlite3.Connection,
    action: str,
    entity_type: str,
    entity_id: Any,
    details: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """INSERT INTO audit_logs(store_id, action, entity_type, entity_id, details_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (STORE_ID, action, entity_type, str(entity_id), json.dumps(details or {}, ensure_ascii=False), now_iso()),
    )


def get_setting(key: str, default: str = "") -> str:
    with connect() as connection:
        row = connection.execute(
            "SELECT setting_value FROM app_settings WHERE store_id=? AND setting_key=?",
            (STORE_ID, key),
        ).fetchone()
    return row[0] if row else default


def set_setting(key: str, value: str) -> None:
    with connect() as connection, atomic(connection):
        connection.execute(
            """INSERT INTO app_settings(store_id, setting_key, setting_value, updated_at)
               VALUES (?, ?, ?, ?) ON CONFLICT(store_id, setting_key)
               DO UPDATE SET setting_value=excluded.setting_value, updated_at=excluded.updated_at""",
            (STORE_ID, _text(key, "Kunci pengaturan", required=True), str(value), now_iso()),
        )
        _audit(connection, "update", "setting", key, {"value": value})


def list_categories() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, name FROM categories WHERE store_id=? ORDER BY name COLLATE NOCASE",
            (STORE_ID,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_category(name: str) -> int:
    name = _text(name, "Nama kategori", required=True)
    try:
        with connect() as connection, atomic(connection):
            cursor = connection.execute(
                "INSERT INTO categories(store_id, name) VALUES (?, ?)", (STORE_ID, name)
            )
            category_id = int(cursor.lastrowid)
            _audit(connection, "create", "category", category_id, {"name": name})
            return category_id
    except sqlite3.IntegrityError as exc:
        raise RetailError("Nama kategori sudah digunakan.") from exc


def list_suppliers(active_only: bool = False) -> list[dict[str, Any]]:
    where = "WHERE s.store_id=?"
    params: list[Any] = [STORE_ID]
    if active_only:
        where += " AND s.is_active=1"
    query = f"""SELECT s.*,
        (SELECT COUNT(*) FROM products p WHERE p.supplier_id=s.id) AS product_count,
        (SELECT COALESCE(SUM(r.quantity), 0) FROM stock_receipts r WHERE r.supplier_id=s.id) AS units_received
        FROM suppliers s {where} ORDER BY s.name COLLATE NOCASE"""
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def save_supplier(data: dict[str, Any], supplier_id: int | None = None) -> int:
    name = _text(data.get("name"), "Nama supplier", required=True)
    lead = _int(data.get("lead_time_days", 7), "Lead time", 0)
    active = int(bool(data.get("is_active", True)))
    fields = (
        name,
        _text(data.get("contact"), "Kontak"),
        _text(data.get("email"), "Email"),
        _text(data.get("address"), "Alamat"),
        lead,
        active,
    )
    try:
        with connect() as connection, atomic(connection):
            if supplier_id:
                connection.execute(
                    """UPDATE suppliers SET name=?, contact=?, email=?, address=?, lead_time_days=?,
                       is_active=?, updated_at=? WHERE id=? AND store_id=?""",
                    (*fields, now_iso(), supplier_id, STORE_ID),
                )
                if connection.execute("SELECT changes()").fetchone()[0] == 0:
                    raise RetailError("Supplier tidak ditemukan.")
                result_id = supplier_id
                action = "update"
            else:
                cursor = connection.execute(
                    """INSERT INTO suppliers(store_id, name, contact, email, address, lead_time_days, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (STORE_ID, *fields),
                )
                result_id = int(cursor.lastrowid)
                action = "create"
            _audit(connection, action, "supplier", result_id, {"name": name})
            return result_id
    except sqlite3.IntegrityError as exc:
        raise RetailError("Nama supplier sudah digunakan.") from exc


def list_customers(search: str = "") -> list[dict[str, Any]]:
    term = f"%{search.strip()}%"
    with connect() as connection:
        rows = connection.execute(
            """SELECT c.id, c.name, c.phone, c.email, c.created_at,
               COUNT(CASE WHEN t.status='completed' THEN 1 END) AS transaction_count,
               COALESCE(SUM(CASE WHEN t.status='completed' THEN t.subtotal-t.discount_amount ELSE 0 END), 0) AS total_spend,
               MAX(CASE WHEN t.status='completed' THEN t.created_at END) AS last_purchase
               FROM customers c LEFT JOIN transactions t ON t.customer_id=c.id
               WHERE c.store_id=? AND (c.name LIKE ? OR c.phone LIKE ? OR c.email LIKE ?)
               GROUP BY c.id ORDER BY c.name COLLATE NOCASE""",
            (STORE_ID, term, term, term),
        ).fetchall()
    return [dict(row) for row in rows]


def save_customer(data: dict[str, Any], customer_id: int | None = None) -> int:
    name = _text(data.get("name"), "Nama pelanggan", required=True)
    phone = _text(data.get("phone"), "Nomor kontak") or None
    email = _text(data.get("email"), "Email")
    try:
        with connect() as connection, atomic(connection):
            if customer_id:
                connection.execute(
                    "UPDATE customers SET name=?, phone=?, email=?, updated_at=? WHERE id=? AND store_id=?",
                    (name, phone, email, now_iso(), customer_id, STORE_ID),
                )
                if connection.execute("SELECT changes()").fetchone()[0] == 0:
                    raise RetailError("Pelanggan tidak ditemukan.")
                result_id, action = customer_id, "update"
            else:
                cursor = connection.execute(
                    "INSERT INTO customers(store_id, name, phone, email) VALUES (?, ?, ?, ?)",
                    (STORE_ID, name, phone, email),
                )
                result_id, action = int(cursor.lastrowid), "create"
            _audit(connection, action, "customer", result_id, {"name": name})
            return result_id
    except sqlite3.IntegrityError as exc:
        raise RetailError("Nomor kontak pelanggan sudah tercatat.") from exc


def list_products(
    search: str = "", category_id: int | None = None, active_only: bool = True
) -> list[dict[str, Any]]:
    conditions = ["p.store_id=?"]
    params: list[Any] = [STORE_ID]
    if active_only:
        conditions.append("p.is_active=1")
    if category_id is not None:
        conditions.append("p.category_id=?")
        params.append(category_id)
    if search.strip():
        term = f"%{search.strip()}%"
        conditions.append("(p.name LIKE ? OR p.sku LIKE ? OR COALESCE(p.barcode, '') LIKE ?)")
        params.extend([term, term, term])
    query = f"""SELECT p.*, c.name AS category_name, s.name AS supplier_name,
               (p.stock * p.cost_price) AS inventory_value
               FROM products p LEFT JOIN categories c ON c.id=p.category_id
               LEFT JOIN suppliers s ON s.id=p.supplier_id
               WHERE {' AND '.join(conditions)} ORDER BY p.name COLLATE NOCASE"""
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def save_product(data: dict[str, Any], product_id: int | None = None) -> int:
    sku = _text(data.get("sku"), "SKU", required=True)
    name = _text(data.get("name"), "Nama produk", required=True)
    barcode = _text(data.get("barcode"), "Barcode") or None
    cost = _int(data.get("cost_price", 0), "Harga modal")
    price = _int(data.get("selling_price", 0), "Harga jual")
    stock = _int(data.get("stock", 0), "Stok")
    minimum = _int(data.get("minimum_stock", 0), "Stok minimum")
    if price < cost:
        # Below-cost selling may be intentional, so the UI can explain it without blocking.
        pass
    category_id = data.get("category_id") or None
    supplier_id = data.get("supplier_id") or None
    values = (
        sku, barcode, name, category_id,
        _text(data.get("description"), "Deskripsi"), cost, price, stock, minimum,
        _text(data.get("unit") or "pcs", "Satuan", required=True), supplier_id,
        int(bool(data.get("is_active", True))),
    )
    try:
        with connect() as connection, atomic(connection):
            if category_id is not None and not connection.execute(
                "SELECT 1 FROM categories WHERE id=? AND store_id=?", (category_id, STORE_ID)
            ).fetchone():
                raise RetailError("Kategori tidak ditemukan pada toko ini.")
            if supplier_id is not None and not connection.execute(
                "SELECT 1 FROM suppliers WHERE id=? AND store_id=?", (supplier_id, STORE_ID)
            ).fetchone():
                raise RetailError("Supplier tidak ditemukan pada toko ini.")
            if product_id:
                old = connection.execute(
                    "SELECT stock, cost_price FROM products WHERE id=? AND store_id=?",
                    (product_id, STORE_ID),
                ).fetchone()
                if not old:
                    raise RetailError("Produk tidak ditemukan.")
                connection.execute(
                    """UPDATE products SET sku=?, barcode=?, name=?, category_id=?, description=?, cost_price=?,
                       selling_price=?, stock=?, minimum_stock=?, unit=?, supplier_id=?, is_active=?, updated_at=?
                       WHERE id=? AND store_id=?""",
                    (*values, now_iso(), product_id, STORE_ID),
                )
                if old["stock"] != stock:
                    movement = connection.execute(
                        """INSERT INTO inventory_movements(store_id, product_id, movement_type, quantity_delta,
                           stock_after, unit_cost, reason, created_at) VALUES (?, ?, 'adjustment', ?, ?, ?, ?, ?)""",
                        (STORE_ID, product_id, stock - old["stock"], stock, cost, "Perubahan stok melalui edit produk", now_iso()),
                    )
                    connection.execute(
                        "INSERT INTO stock_adjustments(movement_id, stock_before, stock_after, reason) VALUES (?, ?, ?, ?)",
                        (movement.lastrowid, old["stock"], stock, "Perubahan stok melalui edit produk"),
                    )
                result_id, action = product_id, "update"
            else:
                cursor = connection.execute(
                    """INSERT INTO products(store_id, sku, barcode, name, category_id, description, cost_price,
                       selling_price, stock, minimum_stock, unit, supplier_id, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (STORE_ID, *values),
                )
                result_id, action = int(cursor.lastrowid), "create"
                if stock:
                    movement = connection.execute(
                        """INSERT INTO inventory_movements(store_id, product_id, movement_type, quantity_delta,
                           stock_after, unit_cost, reason, created_at) VALUES (?, ?, 'restock', ?, ?, ?, ?, ?)""",
                        (STORE_ID, result_id, stock, stock, cost, "Stok awal produk", now_iso()),
                    )
                    connection.execute(
                        "INSERT INTO stock_adjustments(movement_id, stock_before, stock_after, reason) VALUES (?, 0, ?, ?)",
                        (movement.lastrowid, stock, "Stok awal produk"),
                    )
            _audit(connection, action, "product", result_id, {"sku": sku, "name": name})
            return result_id
    except sqlite3.IntegrityError as exc:
        text = str(exc).lower()
        if "barcode" in text:
            raise RetailError("Barcode sudah digunakan oleh produk lain.") from exc
        if "sku" in text:
            raise RetailError("SKU sudah digunakan oleh produk lain.") from exc
        raise RetailError("Data produk tidak valid atau relasi kategori/supplier tidak ditemukan.") from exc


def archive_product(product_id: int) -> None:
    with connect() as connection, atomic(connection):
        connection.execute(
            "UPDATE products SET is_active=0, updated_at=? WHERE id=? AND store_id=?",
            (now_iso(), product_id, STORE_ID),
        )
        if connection.execute("SELECT changes()").fetchone()[0] == 0:
            raise RetailError("Produk tidak ditemukan.")
        _audit(connection, "archive", "product", product_id)


def _movement(
    connection: sqlite3.Connection,
    product_id: int,
    movement_type: str,
    quantity_delta: int,
    stock_after: int,
    unit_cost: int,
    reason: str,
    *,
    transaction_id: int | None = None,
    receipt_id: int | None = None,
) -> int:
    cursor = connection.execute(
        """INSERT INTO inventory_movements(store_id, product_id, transaction_id, receipt_id, movement_type,
           quantity_delta, stock_after, unit_cost, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (STORE_ID, product_id, transaction_id, receipt_id, movement_type, quantity_delta, stock_after, unit_cost, reason, now_iso()),
    )
    return int(cursor.lastrowid)


def stock_in(
    product_id: int, quantity: Any, reason: str, supplier_id: int | None = None, unit_cost: Any | None = None
) -> str:
    quantity = _int(quantity, "Jumlah stok masuk", 1)
    reason = _text(reason, "Alasan stok masuk", required=True)
    receipt_no = f"RCV-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(3).upper()}"
    with connect() as connection, atomic(connection):
        product = connection.execute(
            "SELECT stock, cost_price FROM products WHERE id=? AND store_id=? AND is_active=1",
            (product_id, STORE_ID),
        ).fetchone()
        if not product:
            raise RetailError("Produk aktif tidak ditemukan.")
        cost = product["cost_price"] if unit_cost in (None, "") else _int(unit_cost, "Harga modal")
        new_stock = product["stock"] + quantity
        cursor = connection.execute(
            """INSERT INTO stock_receipts(store_id, supplier_id, product_id, receipt_no, quantity, unit_cost, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (STORE_ID, supplier_id, product_id, receipt_no, quantity, cost, reason, now_iso()),
        )
        connection.execute(
            "UPDATE products SET stock=?, cost_price=?, updated_at=? WHERE id=?",
            (new_stock, cost, now_iso(), product_id),
        )
        movement_id = _movement(connection, product_id, "restock", quantity, new_stock, cost, reason, receipt_id=cursor.lastrowid)
        _audit(connection, "stock_in", "product", product_id, {"quantity": quantity, "receipt_no": receipt_no, "movement_id": movement_id})
    return receipt_no


def adjust_stock(product_id: int, new_stock: Any, reason: str) -> int:
    new_stock = _int(new_stock, "Stok baru")
    reason = _text(reason, "Alasan penyesuaian", required=True)
    with connect() as connection, atomic(connection):
        product = connection.execute(
            "SELECT stock, cost_price FROM products WHERE id=? AND store_id=?",
            (product_id, STORE_ID),
        ).fetchone()
        if not product:
            raise RetailError("Produk tidak ditemukan.")
        delta = new_stock - product["stock"]
        if delta == 0:
            raise RetailError("Stok baru sama dengan stok saat ini.")
        connection.execute("UPDATE products SET stock=?, updated_at=? WHERE id=?", (new_stock, now_iso(), product_id))
        movement_id = _movement(connection, product_id, "adjustment", delta, new_stock, product["cost_price"], reason)
        connection.execute(
            "INSERT INTO stock_adjustments(movement_id, stock_before, stock_after, reason) VALUES (?, ?, ?, ?)",
            (movement_id, product["stock"], new_stock, reason),
        )
        _audit(connection, "stock_adjustment", "product", product_id, {"before": product["stock"], "after": new_stock, "reason": reason})
    return movement_id


def list_movements(product_id: int | None = None, limit: int = 500) -> list[dict[str, Any]]:
    query = """SELECT m.*, p.name AS product_name, p.sku, t.transaction_no, r.receipt_no
        FROM inventory_movements m JOIN products p ON p.id=m.product_id
        LEFT JOIN transactions t ON t.id=m.transaction_id LEFT JOIN stock_receipts r ON r.id=m.receipt_id
        WHERE m.store_id=?"""
    params: list[Any] = [STORE_ID]
    if product_id is not None:
        query += " AND m.product_id=?"
        params.append(product_id)
    query += " ORDER BY m.created_at DESC, m.id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 2000)))
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def list_receipts(supplier_id: int | None = None, limit: int = 500) -> list[dict[str, Any]]:
    query = """SELECT r.*, p.name AS product_name, p.sku, s.name AS supplier_name
        FROM stock_receipts r JOIN products p ON p.id=r.product_id
        LEFT JOIN suppliers s ON s.id=r.supplier_id WHERE r.store_id=?"""
    params: list[Any] = [STORE_ID]
    if supplier_id is not None:
        query += " AND r.supplier_id=?"
        params.append(supplier_id)
    query += " ORDER BY r.created_at DESC, r.id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 2000)))
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def list_audit_logs(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT action, entity_type, entity_id, details_json, created_at FROM audit_logs WHERE store_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
            (STORE_ID, max(1, min(int(limit), 1000))),
        ).fetchall()
    return [dict(row) for row in rows]


def _transaction_no() -> str:
    return f"TRX-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(4).upper()}"


def create_transaction(
    items: Iterable[dict[str, Any]],
    *,
    payment_method: str,
    amount_paid: Any | None,
    discount_amount: Any = 0,
    tax_rate: Any = 0,
    customer_id: int | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    normalized: dict[int, int] = {}
    for item in items:
        product_id = _int(item.get("product_id"), "Produk", 1)
        quantity = _int(item.get("quantity"), "Kuantitas", 1)
        normalized[product_id] = normalized.get(product_id, 0) + quantity
    if not normalized:
        raise RetailError("Keranjang masih kosong.")
    payment_method = _text(payment_method, "Metode pembayaran", required=True)
    if payment_method not in PAYMENT_METHODS:
        raise RetailError("Metode pembayaran tidak dikenali.")
    try:
        tax_rate_value = float(tax_rate or 0)
    except (TypeError, ValueError) as exc:
        raise RetailError("Tarif pajak harus berupa angka.") from exc
    if not math.isfinite(tax_rate_value) or tax_rate_value < 0 or tax_rate_value > 100:
        raise RetailError("Tarif pajak harus antara 0% dan 100%.")
    discount = _int(discount_amount or 0, "Diskon")

    with connect() as connection, _checkout_atomic(connection):
        if idempotency_key:
            existing = connection.execute(
                "SELECT id, transaction_no, total FROM transactions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return {"id": existing["id"], "transaction_no": existing["transaction_no"], "total": existing["total"], "replayed": True}

        products: list[dict[str, Any]] = []
        subtotal = 0
        for product_id, quantity in normalized.items():
            product = connection.execute(
                "SELECT * FROM products WHERE id=? AND store_id=? AND is_active=1",
                (product_id, STORE_ID),
            ).fetchone()
            if not product:
                raise RetailError("Salah satu produk tidak tersedia atau telah diarsipkan.")
            if quantity > product["stock"]:
                raise RetailError(f"Stok {product['name']} tidak cukup. Tersedia {product['stock']} {product['unit']}.")
            line_total = product["selling_price"] * quantity
            subtotal += line_total
            products.append({"row": product, "quantity": quantity, "line_total": line_total})
        if discount > subtotal:
            raise RetailError("Diskon tidak boleh melebihi subtotal.")
        taxable = subtotal - discount
        tax_amount = int(round(taxable * tax_rate_value / 100))
        total = taxable + tax_amount
        paid = total if amount_paid in (None, "") and payment_method != "Tunai" else _int(amount_paid, "Jumlah dibayar")
        if paid < total:
            raise RetailError("Jumlah dibayar kurang dari total transaksi.")
        if payment_method != "Tunai" and paid != total:
            raise RetailError("Untuk pembayaran non-tunai, jumlah dibayar harus sama dengan total transaksi.")
        change = paid - total if payment_method == "Tunai" else 0
        if customer_id is not None:
            exists = connection.execute(
                "SELECT 1 FROM customers WHERE id=? AND store_id=?", (customer_id, STORE_ID)
            ).fetchone()
            if not exists:
                raise RetailError("Pelanggan yang dipilih tidak ditemukan.")
        created = now_iso()
        transaction_no = _transaction_no()
        cursor = connection.execute(
            """INSERT INTO transactions(store_id, transaction_no, idempotency_key, customer_id, subtotal,
               discount_amount, tax_rate, tax_amount, total, amount_paid, change_amount, payment_method,
               status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?)""",
            (STORE_ID, transaction_no, idempotency_key, customer_id, subtotal, discount, tax_rate_value,
             tax_amount, total, paid, change, payment_method, created, created),
        )
        transaction_id = int(cursor.lastrowid)
        for item in products:
            product = item["row"]
            quantity = item["quantity"]
            connection.execute(
                """INSERT INTO transaction_items(transaction_id, product_id, sku_snapshot, product_name_snapshot,
                   quantity, unit_price, unit_cost, line_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (transaction_id, product["id"], product["sku"], product["name"], quantity,
                 product["selling_price"], product["cost_price"], item["line_total"]),
            )
            stock_after = product["stock"] - quantity
            connection.execute("UPDATE products SET stock=?, updated_at=? WHERE id=?", (stock_after, created, product["id"]))
            _movement(connection, product["id"], "sale", -quantity, stock_after, product["cost_price"], f"Penjualan {transaction_no}", transaction_id=transaction_id)
        connection.execute(
            "INSERT INTO payments(transaction_id, method, amount, kind, created_at) VALUES (?, ?, ?, 'capture', ?)",
            (transaction_id, payment_method, total, created),
        )
        _audit(connection, "checkout", "transaction", transaction_id, {"transaction_no": transaction_no, "total": total, "items": len(products)})
    return {"id": transaction_id, "transaction_no": transaction_no, "total": total, "replayed": False}


def list_transactions(
    search: str = "", start_date: str | None = None, end_date: str | None = None,
    payment_method: str | None = None, status: str | None = None, limit: int = 1000,
    customer_id: int | None = None,
) -> list[dict[str, Any]]:
    where = ["t.store_id=?"]
    params: list[Any] = [STORE_ID]
    if search.strip():
        term = f"%{search.strip()}%"
        where.append("(t.transaction_no LIKE ? OR c.name LIKE ?)")
        params.extend([term, term])
    lower, upper = utc_date_bounds(start_date, end_date)
    if lower:
        where.append("t.created_at>=?")
        params.append(lower)
    if upper:
        where.append("t.created_at<?")
        params.append(upper)
    if payment_method:
        where.append("t.payment_method=?")
        params.append(payment_method)
    if status:
        where.append("t.status=?")
        params.append(status)
    if customer_id is not None:
        where.append("t.customer_id=?")
        params.append(customer_id)
    params.append(max(1, min(int(limit), 5000)))
    query = f"""SELECT t.*, c.name AS customer_name, COUNT(i.id) AS item_count,
        COALESCE(SUM(i.quantity), 0) AS units_sold
        FROM transactions t LEFT JOIN customers c ON c.id=t.customer_id
        LEFT JOIN transaction_items i ON i.transaction_id=t.id
        WHERE {' AND '.join(where)} GROUP BY t.id ORDER BY t.created_at DESC, t.id DESC LIMIT ?"""
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def transaction_detail(transaction_id: int) -> dict[str, Any] | None:
    with connect() as connection:
        transaction = connection.execute(
            """SELECT t.*, c.name AS customer_name, c.phone AS customer_phone
               FROM transactions t LEFT JOIN customers c ON c.id=t.customer_id
               WHERE t.id=? AND t.store_id=?""",
            (transaction_id, STORE_ID),
        ).fetchone()
        if not transaction:
            return None
        items = connection.execute(
            "SELECT * FROM transaction_items WHERE transaction_id=? ORDER BY id", (transaction_id,)
        ).fetchall()
        payments = connection.execute(
            "SELECT * FROM payments WHERE transaction_id=? ORDER BY id", (transaction_id,)
        ).fetchall()
    return {"transaction": dict(transaction), "items": [dict(row) for row in items], "payments": [dict(row) for row in payments]}


def _reverse_transaction(transaction_id: int, reason: str, status: str, *, restock: bool) -> None:
    reason = _text(reason, "Alasan", required=True)
    if status not in {"refunded", "cancelled"}:
        raise ValueError("Invalid reverse status")
    with connect() as connection, atomic(connection):
        transaction = connection.execute(
            "SELECT * FROM transactions WHERE id=? AND store_id=?", (transaction_id, STORE_ID)
        ).fetchone()
        if not transaction:
            raise RetailError("Transaksi tidak ditemukan.")
        if transaction["status"] != "completed":
            raise RetailError("Hanya transaksi selesai yang dapat dibatalkan atau direfund.")
        items = connection.execute(
            "SELECT * FROM transaction_items WHERE transaction_id=?", (transaction_id,)
        ).fetchall()
        created = now_iso()
        if restock:
            for item in items:
                product = connection.execute("SELECT stock FROM products WHERE id=?", (item["product_id"],)).fetchone()
                if not product:
                    raise RetailError("Produk transaksi tidak ditemukan. Perubahan dibatalkan.")
                new_stock = product["stock"] + item["quantity"]
                connection.execute("UPDATE products SET stock=?, updated_at=? WHERE id=?", (new_stock, created, item["product_id"]))
                _movement(
                    connection, item["product_id"], "refund" if status == "refunded" else "cancellation",
                    item["quantity"], new_stock, item["unit_cost"], reason, transaction_id=transaction_id,
                )
        connection.execute(
            "UPDATE transactions SET status=?, reason=?, updated_at=? WHERE id=?",
            (status, reason, created, transaction_id),
        )
        if status == "refunded":
            connection.execute(
                "INSERT INTO payments(transaction_id, method, amount, kind, created_at) VALUES (?, ?, ?, 'refund', ?)",
                (transaction_id, transaction["payment_method"], -transaction["total"], created),
            )
        _audit(connection, status, "transaction", transaction_id, {"reason": reason, "stock_restored": restock})


def refund_transaction(transaction_id: int, reason: str, *, restock: bool = True) -> None:
    _reverse_transaction(transaction_id, reason, "refunded", restock=restock)


def cancel_transaction(transaction_id: int, reason: str, *, restock: bool) -> None:
    """Cancel without deleting history. Restore stock only when the returned goods are resalable."""
    reason = _text(reason, "Alasan pembatalan", required=True)
    if restock:
        _reverse_transaction(transaction_id, reason, "cancelled", restock=True)
        return
    with connect() as connection, atomic(connection):
        transaction = connection.execute(
            "SELECT status FROM transactions WHERE id=? AND store_id=?", (transaction_id, STORE_ID)
        ).fetchone()
        if not transaction:
            raise RetailError("Transaksi tidak ditemukan.")
        if transaction["status"] != "completed":
            raise RetailError("Hanya transaksi selesai yang dapat dibatalkan.")
        connection.execute(
            "UPDATE transactions SET status='cancelled', reason=?, updated_at=? WHERE id=?",
            (reason, now_iso(), transaction_id),
        )
        _audit(connection, "cancelled", "transaction", transaction_id, {"reason": reason, "stock_restored": False})


def import_products(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    if not rows:
        raise RetailError("File CSV tidak berisi baris produk.")
    created = updated = 0
    seen_skus: set[str] = set()
    with connect() as connection, atomic(connection):
        for row_number, row in enumerate(rows, start=2):
            try:
                sku = _text(row.get("sku"), "SKU", required=True)
                if sku.casefold() in seen_skus:
                    raise RetailError("SKU muncul lebih dari sekali pada file.")
                seen_skus.add(sku.casefold())
                name = _text(row.get("name"), "Nama produk", required=True)
                barcode = _text(row.get("barcode"), "Barcode") or None
                category_raw = row.get("category_id")
                supplier_raw = row.get("supplier_id")
                category_id = _int(category_raw, "ID kategori", 1) if str(category_raw or "").strip() else None
                supplier_id = _int(supplier_raw, "ID supplier", 1) if str(supplier_raw or "").strip() else None
                if category_id is not None and not connection.execute(
                    "SELECT 1 FROM categories WHERE id=? AND store_id=?", (category_id, STORE_ID)
                ).fetchone():
                    raise RetailError("ID kategori tidak ditemukan pada toko ini.")
                if supplier_id is not None and not connection.execute(
                    "SELECT 1 FROM suppliers WHERE id=? AND store_id=?", (supplier_id, STORE_ID)
                ).fetchone():
                    raise RetailError("ID supplier tidak ditemukan pada toko ini.")
                cost = _int(row.get("cost_price"), "Harga modal")
                price = _int(row.get("selling_price"), "Harga jual")
                stock = _int(row.get("stock") or 0, "Stok")
                minimum = _int(row.get("minimum_stock") or 0, "Stok minimum")
                unit = _text(row.get("unit") or "pcs", "Satuan", required=True)
                active_raw = row.get("is_active", "1")
                active_text = "1" if active_raw is None or not str(active_raw).strip() else str(active_raw).strip().lower()
                active = int(active_text not in {"0", "false", "no", "tidak"})
                product = connection.execute(
                    "SELECT id, stock FROM products WHERE store_id=? AND sku=?", (STORE_ID, sku)
                ).fetchone()
                fields = (sku, barcode, name, category_id, _text(row.get("description"), "Deskripsi"),
                          cost, price, stock, minimum, unit, supplier_id, active)
                if product:
                    product_id = int(product["id"])
                    connection.execute(
                        """UPDATE products SET sku=?, barcode=?, name=?, category_id=?, description=?, cost_price=?,
                           selling_price=?, stock=?, minimum_stock=?, unit=?, supplier_id=?, is_active=?, updated_at=?
                           WHERE id=? AND store_id=?""",
                        (*fields, now_iso(), product_id, STORE_ID),
                    )
                    if product["stock"] != stock:
                        movement_id = _movement(connection, product_id, "adjustment", stock-product["stock"], stock, cost, "Impor CSV produk")
                        connection.execute(
                            "INSERT INTO stock_adjustments(movement_id, stock_before, stock_after, reason) VALUES (?, ?, ?, ?)",
                            (movement_id, product["stock"], stock, "Impor CSV produk"),
                        )
                    updated += 1
                    action = "update"
                else:
                    cursor = connection.execute(
                        """INSERT INTO products(store_id, sku, barcode, name, category_id, description, cost_price,
                           selling_price, stock, minimum_stock, unit, supplier_id, is_active)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (STORE_ID, *fields),
                    )
                    product_id = int(cursor.lastrowid)
                    if stock:
                        movement_id = _movement(connection, product_id, "restock", stock, stock, cost, "Stok awal dari impor CSV")
                        connection.execute(
                            "INSERT INTO stock_adjustments(movement_id, stock_before, stock_after, reason) VALUES (?, 0, ?, ?)",
                            (movement_id, stock, "Stok awal dari impor CSV"),
                        )
                    created += 1
                    action = "create"
                _audit(connection, action, "product", product_id, {"sku": sku, "name": name, "source": "csv"})
            except (RetailError, sqlite3.IntegrityError) as exc:
                raise RetailError(f"Baris {row_number}: {exc}") from exc
    return {"created": created, "updated": updated}
