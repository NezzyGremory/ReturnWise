from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import pytest

import smart_retail.db as db
import smart_retail.backup as backup_module
from smart_retail.analytics import (
    category_performance, dashboard_data, inventory_analysis, product_performance, rfm_analysis,
)
from smart_retail.backup import create_backup, restore_backup
from smart_retail.intelligence import _daily_frame, inventory_recommendations, run_sales_forecast
from smart_retail.services import (
    RetailError,
    adjust_stock,
    archive_product,
    cancel_transaction,
    create_transaction,
    create_category,
    import_products,
    list_movements,
    list_products,
    list_transactions,
    refund_transaction,
    save_customer,
    save_product,
    stock_in,
    transaction_detail,
)
from smart_retail.ui import receipt_html, safe_csv_frame, timestamp
from smart_retail.timeutils import BUSINESS_TIMEZONE, business_today


@pytest.fixture
def local_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "smart-retail-test.sqlite3")
    db.initialize_database()
    return db.DB_PATH


def make_product(*, stock=5, price=1200, cost=700):
    return save_product({
        "sku": "SKU-001", "name": "Teh Botol", "cost_price": cost,
        "selling_price": price, "stock": stock, "minimum_stock": 2, "unit": "botol",
    })


def test_checkout_is_atomic_updates_stock_and_replays_idempotently(local_db):
    product_id = make_product()
    key = "checkout-test-unique"
    result = create_transaction(
        [{"product_id": product_id, "quantity": 2}],
        payment_method="Tunai", amount_paid=3000, idempotency_key=key,
    )
    replay = create_transaction(
        [{"product_id": product_id, "quantity": 2}],
        payment_method="Tunai", amount_paid=3000, idempotency_key=key,
    )

    assert result["total"] == 2400
    assert result["replayed"] is False
    assert replay["replayed"] is True
    assert replay["transaction_no"] == result["transaction_no"]
    assert list_products()[0]["stock"] == 3
    detail = transaction_detail(result["id"])
    assert detail["items"][0]["quantity"] == 2
    assert detail["transaction"]["change_amount"] == 600
    moves = list_movements(product_id)
    assert [(row["movement_type"], row["quantity_delta"]) for row in moves] == [("sale", -2), ("restock", 5)]


def test_failed_checkout_rolls_back_without_partial_stock_change(local_db):
    product_id = make_product(stock=1)
    with pytest.raises(RetailError, match="tidak cukup"):
        create_transaction(
            [{"product_id": product_id, "quantity": 2}],
            payment_method="Tunai", amount_paid=5000,
        )
    assert list_products()[0]["stock"] == 1
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0


def test_discount_tax_and_non_cash_payment_validation(local_db):
    product_id = make_product(stock=3, price=10_000, cost=6000)
    result = create_transaction(
        [{"product_id": product_id, "quantity": 2}],
        payment_method="Transfer", amount_paid=18_000, discount_amount=2000, tax_rate=0,
    )
    assert result["total"] == 18_000
    assert transaction_detail(result["id"])["transaction"]["discount_amount"] == 2000
    with pytest.raises(RetailError, match="harus sama"):
        create_transaction(
            [{"product_id": product_id, "quantity": 1}],
            payment_method="QRIS", amount_paid=20_000,
        )


def test_tax_discount_payment_and_change_are_saved_consistently(local_db):
    product_id = make_product(stock=3, price=1200)
    result = create_transaction(
        [{"product_id": product_id, "quantity": 2}],
        payment_method="Tunai", amount_paid=2500, discount_amount=200, tax_rate=11,
    )
    detail = transaction_detail(result["id"])
    transaction = detail["transaction"]

    assert transaction["subtotal"] == 2400
    assert transaction["discount_amount"] == 200
    assert transaction["tax_amount"] == 242
    assert transaction["total"] == 2442
    assert transaction["amount_paid"] == 2500
    assert transaction["change_amount"] == 58
    assert [(payment["kind"], payment["amount"]) for payment in detail["payments"]] == [("capture", 2442)]
    assert list_products()[0]["stock"] == 1
    assert [(row["movement_type"], row["quantity_delta"]) for row in list_movements(product_id)[:1]] == [("sale", -2)]
    with pytest.raises(RetailError, match="kurang"):
        create_transaction(
            [{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1000,
        )
    assert list_products()[0]["stock"] == 1


def test_discount_allocation_preserves_net_sales_across_reports(local_db):
    first_category = create_category("Kategori A")
    second_category = create_category("Kategori B")
    first_product = save_product({
        "sku": "ALLOC-1", "name": "Produk A", "category_id": first_category,
        "cost_price": 2, "selling_price": 5, "stock": 2,
    })
    second_product = save_product({
        "sku": "ALLOC-2", "name": "Produk B", "category_id": second_category,
        "cost_price": 2, "selling_price": 5, "stock": 2,
    })
    tx = create_transaction(
        [{"product_id": first_product, "quantity": 1}, {"product_id": second_product, "quantity": 1}],
        payment_method="Tunai", amount_paid=9, discount_amount=1,
    )

    by_product = product_performance()
    by_category = category_performance()
    dashboard = dashboard_data()

    assert tx["total"] == 9
    assert int(by_product["penjualan_bersih"].sum()) == 9
    assert int(by_category["penjualan_bersih"].sum()) == 9
    assert sum(item["net_sales"] for item in dashboard["top_products"]) == 9
    assert by_product.set_index("product_id")["penjualan_bersih"].to_dict() == {first_product: 5, second_product: 4}


def test_daily_reports_use_business_timezone_for_utc_transactions(local_db):
    product_id = make_product()
    tx = create_transaction(
        [{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200,
    )
    local_day = business_today()
    just_after_local_midnight = datetime.combine(local_day, time(0, 30), BUSINESS_TIMEZONE)
    utc_stamp = just_after_local_midnight.astimezone(timezone.utc).isoformat(timespec="seconds")
    with db.connect() as connection:
        connection.execute("UPDATE transactions SET created_at=? WHERE id=?", (utc_stamp, tx["id"]))
        connection.execute(
            "UPDATE inventory_movements SET created_at=? WHERE transaction_id=? AND movement_type='sale'",
            (utc_stamp, tx["id"]),
        )

    dashboard = dashboard_data(local_day, local_day)
    transactions = list_transactions(start_date=local_day.isoformat(), end_date=local_day.isoformat())
    forecast_history = _daily_frame()
    inventory = inventory_analysis()
    recommendation = inventory_recommendations()[0]

    assert dashboard["transactions"] == 1
    assert dashboard["daily"].iloc[0]["tanggal"] == local_day.isoformat()
    assert len(transactions) == 1
    assert forecast_history.iloc[-1]["tanggal"].date() == local_day
    assert inventory["products"][0]["units_30d"] == 1
    assert recommendation["first_sale"] == local_day.isoformat()
    assert recommendation["units_30d"] == 1
    assert timestamp(utc_stamp).replace(",", ".") in receipt_html(transaction_detail(tx["id"]))


def test_concurrent_duplicate_checkout_creates_one_sale(local_db):
    product_id = make_product(stock=3)

    def submit():
        return create_transaction(
            [{"product_id": product_id, "quantity": 2}], payment_method="Tunai",
            amount_paid=3000, idempotency_key="concurrent-double-submit",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))

    assert len({result["id"] for result in results}) == 1
    assert sorted(result["replayed"] for result in results) == [False, True]
    assert list_products()[0]["stock"] == 1
    assert len([row for row in list_movements(product_id) if row["movement_type"] == "sale"]) == 1
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 1


def test_storage_failure_rolls_back_transaction_items_payments_and_stock(local_db):
    product_id = make_product(stock=3)
    with db.connect() as connection:
        connection.execute("""CREATE TRIGGER reject_payment BEFORE INSERT ON payments
            BEGIN SELECT RAISE(ABORT, 'forced storage failure'); END""")

    with pytest.raises(RetailError, match="Transaksi gagal disimpan"):
        create_transaction(
            [{"product_id": product_id, "quantity": 2}], payment_method="Tunai", amount_paid=3000,
        )

    assert list_products()[0]["stock"] == 3
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM transaction_items").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM payments").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM audit_logs WHERE action='checkout'").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM inventory_movements WHERE movement_type='sale'").fetchone()[0] == 0


def test_cancel_and_refund_transactions_are_excluded_from_demand(local_db):
    product_id = make_product(stock=30)
    refunded = create_transaction(
        [{"product_id": product_id, "quantity": 4}], payment_method="Tunai", amount_paid=6000,
    )
    refund_transaction(refunded["id"], "Pengembalian penuh")
    cancelled = create_transaction(
        [{"product_id": product_id, "quantity": 3}], payment_method="Tunai", amount_paid=4000,
    )
    cancel_transaction(cancelled["id"], "Transaksi salah", restock=True)
    lost = create_transaction(
        [{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200,
    )
    stock_before_lost_cancellation = list_products()[0]["stock"]
    cancel_transaction(lost["id"], "Barang tidak kembali", restock=False)
    completed = create_transaction(
        [{"product_id": product_id, "quantity": 2}], payment_method="Tunai", amount_paid=3000,
    )

    assert transaction_detail(cancelled["id"])["transaction"]["status"] == "cancelled"
    assert transaction_detail(lost["id"])["transaction"]["status"] == "cancelled"
    assert list_products()[0]["stock"] == stock_before_lost_cancellation - 2
    assert transaction_detail(completed["id"])["transaction"]["status"] == "completed"
    analysis_row = inventory_analysis()["products"][0]
    recommendation_row = inventory_recommendations()[0]
    assert analysis_row["units_30d"] == 2
    assert recommendation_row["units_30d"] == 2
    assert recommendation_row["first_sale"] == business_today().isoformat()


def test_reopening_database_preserves_transaction_and_inventory(local_db):
    product_id = make_product(stock=5)
    result = create_transaction(
        [{"product_id": product_id, "quantity": 2}], payment_method="Tunai", amount_paid=3000,
    )
    db.initialize_database()

    assert list_products()[0]["stock"] == 3
    assert transaction_detail(result["id"])["transaction"]["transaction_no"] == result["transaction_no"]
    assert [row["movement_type"] for row in list_movements(product_id)] == ["sale", "restock"]

    env = os.environ.copy()
    env["SMART_RETAIL_DB"] = str(local_db)
    code = (
        "import json; from smart_retail.db import initialize_database; "
        "from smart_retail.services import list_products, list_transactions; "
        "initialize_database(); products=list_products(); transactions=list_transactions(); "
        "print(json.dumps({'products': len(products), 'transactions': len(transactions), 'stock': products[0]['stock']}))"
    )
    reopened = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads(reopened.stdout.strip().splitlines()[-1]) == {"products": 1, "transactions": 1, "stock": 3}


def test_integer_validation_does_not_round_large_currency_values(local_db):
    precise = 9_007_199_254_740_993
    product_id = save_product({
        "sku": "PRECISION-1", "name": "Produk presisi", "cost_price": str(precise),
        "selling_price": str(precise), "stock": 1,
    })
    assert next(item for item in list_products() if item["id"] == product_id)["selling_price"] == precise


def test_product_search_treats_sql_fragments_as_literal_text(local_db):
    make_product()
    assert list_products(search="' OR 1=1 --") == []


def test_csv_exports_escape_formula_like_text_without_changing_numbers():
    frame = pd.DataFrame({"text": ["=1+1", "  @SUM(A1:A2)", "ordinary"], "amount": [-12, 2, 3]})
    safe = safe_csv_frame(frame)

    assert safe["text"].tolist() == ["'=1+1", "'  @SUM(A1:A2)", "ordinary"]
    assert safe["amount"].tolist() == [-12, 2, 3]


def test_restock_adjustment_and_refund_are_audited(local_db):
    product_id = make_product(stock=2)
    receipt_no = stock_in(product_id, 4, "Pembelian PO-10")
    assert receipt_no.startswith("RCV-")
    adjust_stock(product_id, 5, "Hasil hitung fisik")
    result = create_transaction(
        [{"product_id": product_id, "quantity": 2}],
        payment_method="Tunai", amount_paid=5000,
    )
    refund_transaction(result["id"], "Barang kembali utuh")
    assert list_products()[0]["stock"] == 5
    detail = transaction_detail(result["id"])
    assert detail["transaction"]["status"] == "refunded"
    assert any(payment["kind"] == "refund" and payment["amount"] < 0 for payment in detail["payments"])
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM stock_adjustments").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM audit_logs WHERE action='refunded'").fetchone()[0] == 1


def test_archive_keeps_product_and_transaction_history(local_db):
    product_id = make_product()
    tx = create_transaction([{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200)
    archive_product(product_id)
    assert list_products() == []
    assert list_products(active_only=False)[0]["is_active"] == 0
    assert transaction_detail(tx["id"])["items"][0]["product_name_snapshot"] == "Teh Botol"


def test_rfm_uses_customer_transactions_without_invented_customers(local_db):
    customer_id = save_customer({"name": "Nadia", "phone": "0812345678"})
    product_id = make_product()
    create_transaction(
        [{"product_id": product_id, "quantity": 1}], payment_method="Tunai",
        amount_paid=1200, customer_id=customer_id,
    )
    frame = rfm_analysis()
    assert len(frame) == 1
    assert frame.iloc[0]["pelanggan"] == "Nadia"
    assert frame.iloc[0]["frekuensi"] == 1
    assert frame.iloc[0]["segmen"] == "Pembelian pertama"


def test_rfm_preserves_the_highest_priority_loyal_segment(local_db):
    customer_id = save_customer({"name": "Toko Loyal", "phone": "081200000001"})
    product_id = make_product(stock=8)
    for index in range(5):
        create_transaction(
            [{"product_id": product_id, "quantity": 1}], payment_method="Tunai",
            amount_paid=1200, customer_id=customer_id, idempotency_key=f"loyal-{index}",
        )

    frame = rfm_analysis()

    assert frame.iloc[0]["frekuensi"] == 5
    assert frame.iloc[0]["segmen"] == "Pelanggan setia bernilai tinggi"


def test_forecast_shows_insufficient_history_instead_of_fabricating(local_db):
    product_id = make_product()
    create_transaction([{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200)
    result = run_sales_forecast(7)
    assert result["ready"] is False
    assert "Belum cukup data" in result["reason"]
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales_forecasts").fetchone()[0] == 0


def test_backup_round_trip_and_rejects_corruption(local_db):
    product_id = make_product()
    tx = create_transaction([{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200)
    backup = create_backup()
    save_product({"sku": "SKU-002", "name": "Kerupuk", "cost_price": 500, "selling_price": 900, "stock": 3})
    counts = restore_backup(backup)
    assert counts["products"] == 1
    assert counts["transactions"] == 1
    assert [row["sku"] for row in list_products()] == ["SKU-001"]
    assert transaction_detail(tx["id"])["transaction"]["total"] == 1200
    invalid = json.loads(backup)
    invalid["tables"]["products"][0]["sku"] = ""
    with pytest.raises(RetailError):
        restore_backup(json.dumps(invalid))
    assert [row["sku"] for row in list_products()] == ["SKU-001"]


def test_backup_uses_one_snapshot_during_concurrent_checkout(local_db, monkeypatch):
    product_id = make_product(stock=5)
    create_transaction([{"product_id": product_id, "quantity": 1}], payment_method="Tunai", amount_paid=1200)
    with db.connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")

    real_connect = backup_module.connect
    injected = False

    class InterleavingConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, query, parameters=()):
            nonlocal injected
            cursor = self.connection.execute(query, parameters)
            if query == "SELECT * FROM products" and not injected:
                injected = True
                create_transaction(
                    [{"product_id": product_id, "quantity": 1}],
                    payment_method="Tunai", amount_paid=1200,
                )
            return cursor

        def __getattr__(self, name):
            return getattr(self.connection, name)

    class ConnectionContext:
        def __enter__(self):
            self.connection_context = real_connect()
            self.connection = self.connection_context.__enter__()
            return InterleavingConnection(self.connection)

        def __exit__(self, exc_type, exc_value, traceback):
            return self.connection_context.__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(backup_module, "connect", ConnectionContext)
    data = backup_module.create_backup()
    restored_counts = backup_module.restore_backup(data)

    assert injected is True
    assert restored_counts["transactions"] == 1
    assert list_products()[0]["stock"] == 4
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM inventory_movements WHERE movement_type='sale'").fetchone()[0] == 1


def test_backup_export_enforces_the_same_size_limit_as_restore(local_db, monkeypatch):
    monkeypatch.setattr(backup_module, "MAX_BACKUP_BYTES", 128)
    with pytest.raises(RetailError, match="Backup melebihi batas 50 MB"):
        backup_module.create_backup()


def test_csv_import_is_all_or_nothing(local_db):
    make_product()
    rows = [
        {"sku": "SKU-002", "name": "Kerupuk", "cost_price": "500", "selling_price": "900", "stock": "2"},
        {"sku": "SKU-003", "name": "Susu", "cost_price": "-1", "selling_price": "2000", "stock": "3"},
    ]
    with pytest.raises(RetailError, match="Baris 3"):
        import_products(rows)
    assert [row["sku"] for row in list_products()] == ["SKU-001"]


def test_forecast_uses_time_ordered_history_for_category(local_db):
    category_id = create_category("Minuman")
    product_id = save_product({
        "sku": "DRINK-1", "name": "Air mineral", "category_id": category_id,
        "cost_price": 500, "selling_price": 1000, "stock": 40,
    })
    today = date.today()
    for offset in range(29, -1, -1):
        result = create_transaction(
            [{"product_id": product_id, "quantity": 1}],
            payment_method="Tunai", amount_paid=1000,
            idempotency_key=f"forecast-history-{offset}",
        )
        with db.connect() as connection:
            connection.execute(
                "UPDATE transactions SET created_at=? WHERE id=?",
                ((today - timedelta(days=offset)).isoformat() + "T12:00:00+00:00", result["id"]),
            )
    result = run_sales_forecast(7, category_id=category_id)
    assert result["ready"] is True
    assert result["history_days"] == 30
    assert len(result["forecast"]) == 7
    assert result["metrics"]["unit"]["MAE"] == 0
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sales_forecasts WHERE category_id=?", (category_id,)).fetchone()[0] == 7
