"""Business summaries derived from completed Smart Retail transactions."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from smart_retail.db import connect
from smart_retail.services import STORE_ID
from smart_retail.timeutils import business_date, business_today, utc_date_bounds


def _dates(start: date | str | None, end: date | str | None) -> tuple[str | None, str | None]:
    return (start.isoformat() if isinstance(start, date) else start,
            end.isoformat() if isinstance(end, date) else end)


def allocated_sales_items(
    start: date | str | None = None, end: date | str | None = None,
) -> list[dict[str, Any]]:
    """Return completed sale lines with integer-rupiah discounts allocated exactly per transaction."""
    start, end = _dates(start, end)
    lower, upper = utc_date_bounds(start, end)
    filters = ["t.store_id=?", "t.status='completed'"]
    params: list[Any] = [STORE_ID]
    if lower:
        filters.append("t.created_at>=?")
        params.append(lower)
    if upper:
        filters.append("t.created_at<?")
        params.append(upper)
    with connect() as connection:
        rows = connection.execute(
            f"""SELECT i.id AS item_id, i.transaction_id, i.product_id,
                i.sku_snapshot AS sku, i.product_name_snapshot AS product,
                i.quantity, i.unit_cost, i.line_total, t.subtotal, t.discount_amount,
                t.created_at, p.category_id,
                COALESCE(c.name, 'Tanpa kategori') AS category
                FROM transaction_items i JOIN transactions t ON t.id=i.transaction_id
                JOIN products p ON p.id=i.product_id LEFT JOIN categories c ON c.id=p.category_id
                WHERE {' AND '.join(filters)} ORDER BY i.transaction_id, i.id""",
            params,
        ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        item["sale_date"] = business_date(item["created_at"]).isoformat()
        grouped.setdefault(int(item["transaction_id"]), []).append(item)
    result: list[dict[str, Any]] = []
    for items in grouped.values():
        subtotal = int(items[0]["subtotal"])
        discount = int(items[0]["discount_amount"])
        net_total = max(0, subtotal - discount)
        basis = sum(int(item["line_total"]) for item in items)
        if basis > 0:
            allocated = [divmod(int(item["line_total"]) * net_total, basis) for item in items]
            remainder = net_total - sum(value for value, _ in allocated)
            order = sorted(range(len(items)), key=lambda index: (-allocated[index][1], int(items[index]["item_id"])))
            for index in order[:remainder]:
                allocated[index] = (allocated[index][0] + 1, allocated[index][1])
            for item, (net_sales, _) in zip(items, allocated):
                item["net_sales"] = net_sales
        else:
            for item in items:
                item["net_sales"] = 0
        for item in items:
            item["cost_total"] = int(item["unit_cost"]) * int(item["quantity"])
        result.extend(items)
    return result


def dashboard_data(start: date | str | None = None, end: date | str | None = None) -> dict[str, Any]:
    start, end = _dates(start, end)
    lower, upper = utc_date_bounds(start, end)
    filters = ["store_id=?", "status='completed'"]
    params: list[Any] = [STORE_ID]
    if lower:
        filters.append("created_at>=?")
        params.append(lower)
    if upper:
        filters.append("created_at<?")
        params.append(upper)
    where = " AND ".join(filters)
    with connect() as connection:
        totals = connection.execute(
            f"""SELECT COALESCE(SUM(subtotal-discount_amount),0) AS net_sales,
                COUNT(*) AS transactions, COALESCE(SUM(tax_amount),0) AS tax_collected,
                COALESCE(SUM(total),0) AS total_paid, COALESCE(AVG(total),0) AS avg_order
                FROM transactions WHERE {where}""", params,
        ).fetchone()
        units = connection.execute(
            f"""SELECT COALESCE(SUM(i.quantity),0) FROM transaction_items i JOIN transactions t ON t.id=i.transaction_id
                WHERE {where.replace('store_id', 't.store_id').replace('status', 't.status').replace('created_at', 't.created_at')}""",
            params,
        ).fetchone()[0]
        stock = connection.execute(
            """SELECT COUNT(*) AS products, COALESCE(SUM(stock * cost_price),0) AS value,
               SUM(CASE WHEN stock=0 THEN 1 ELSE 0 END) AS out_count,
               SUM(CASE WHEN stock>0 AND stock<=minimum_stock THEN 1 ELSE 0 END) AS low_count
               FROM products WHERE store_id=? AND is_active=1""",
            (STORE_ID,),
        ).fetchone()
        recent = connection.execute(
            """SELECT t.id, t.transaction_no, t.total, t.payment_method, t.status, t.created_at,
               c.name AS customer_name FROM transactions t LEFT JOIN customers c ON c.id=t.customer_id
               WHERE t.store_id=? ORDER BY t.created_at DESC LIMIT 8""", (STORE_ID,),
        ).fetchall()
        daily = connection.execute(
            f"""SELECT created_at, subtotal-discount_amount AS penjualan
                FROM transactions WHERE {where} ORDER BY created_at""", params,
        ).fetchall()
    item_frame = pd.DataFrame(allocated_sales_items(start, end))
    if item_frame.empty:
        top_products = []
    else:
        top_products = (
            item_frame.groupby(["product_id", "product"], as_index=False)
            .agg(units=("quantity", "sum"), net_sales=("net_sales", "sum"))
            .sort_values(["units", "product"], ascending=[False, True])
            .head(5).to_dict("records")
        )
    if daily:
        daily_frame = pd.DataFrame([{
            "tanggal": business_date(row["created_at"]).isoformat(), "penjualan": row["penjualan"]
        } for row in daily]).groupby("tanggal", as_index=False)["penjualan"].sum()
    else:
        daily_frame = pd.DataFrame(columns=["tanggal", "penjualan"])
    return {
        "net_sales": int(totals["net_sales"]), "transactions": int(totals["transactions"]),
        "units": int(units or 0), "tax_collected": int(totals["tax_collected"]),
        "total_paid": int(totals["total_paid"]), "avg_order": int(round(totals["avg_order"] or 0)),
        "products": int(stock["products"] or 0), "inventory_value": int(stock["value"] or 0),
        "out_count": int(stock["out_count"] or 0), "low_count": int(stock["low_count"] or 0),
        "recent": [dict(row) for row in recent], "top_products": top_products,
        "daily": daily_frame,
    }


def product_performance(start: date | str | None = None, end: date | str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(allocated_sales_items(start, end))
    columns = ["product_id", "sku", "produk", "unit_terjual", "penjualan_bersih", "harga_pokok", "laba_kotor_estimasi"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.groupby(["product_id", "sku", "product"], as_index=False).agg(
        unit_terjual=("quantity", "sum"),
        penjualan_bersih=("net_sales", "sum"),
        harga_pokok=("cost_total", "sum"),
    ).rename(columns={"product": "produk"})
    result["laba_kotor_estimasi"] = result["penjualan_bersih"] - result["harga_pokok"]
    return result[columns].sort_values(["unit_terjual", "produk"], ascending=[False, True]).reset_index(drop=True)


def category_performance(start: date | str | None = None, end: date | str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(allocated_sales_items(start, end))
    columns = ["kategori", "unit_terjual", "penjualan_bersih", "laba_kotor_estimasi"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["gross_profit"] = frame["net_sales"] - frame["cost_total"]
    result = frame.groupby("category", as_index=False).agg(
        unit_terjual=("quantity", "sum"),
        penjualan_bersih=("net_sales", "sum"),
        laba_kotor_estimasi=("gross_profit", "sum"),
    ).rename(columns={"category": "kategori"})
    return result[columns].sort_values("penjualan_bersih", ascending=False).reset_index(drop=True)


def inventory_analysis() -> dict[str, Any]:
    today = business_today()
    lower, upper = utc_date_bounds(today - timedelta(days=29), today)
    with connect() as connection:
        inventory = connection.execute(
            """SELECT p.id, p.sku, p.name, p.stock, p.minimum_stock, p.unit, p.cost_price,
               COALESCE(SUM(CASE WHEN m.movement_type='sale' AND t.status='completed'
                   AND m.created_at>=? AND m.created_at<? THEN -m.quantity_delta ELSE 0 END),0) AS units_30d,
               SUM(CASE WHEN m.stock_after=0 AND m.movement_type='sale' AND t.status='completed' THEN 1 ELSE 0 END) AS stockout_events
               FROM products p LEFT JOIN inventory_movements m ON m.product_id=p.id
               LEFT JOIN transactions t ON t.id=m.transaction_id
               WHERE p.store_id=? AND p.is_active=1 GROUP BY p.id ORDER BY units_30d DESC""",
            (lower, upper, STORE_ID),
        ).fetchall()
        cogs = connection.execute(
            """SELECT COALESCE(SUM(i.unit_cost*i.quantity),0) FROM transaction_items i
               JOIN transactions t ON t.id=i.transaction_id WHERE t.store_id=? AND t.status='completed'
               AND t.created_at>=? AND t.created_at<?""", (STORE_ID, lower, upper),
        ).fetchone()[0]
    rows = [dict(row) for row in inventory]
    current_value = sum(row["stock"] * row["cost_price"] for row in rows)
    return {
        "products": rows,
        "current_inventory_value": current_value,
        "cogs_30d": int(cogs or 0),
        "turnover_proxy": round(float(cogs or 0) / current_value, 2) if current_value else None,
        "slow_moving": [row for row in rows if row["units_30d"] == 0],
        "fast_moving": sorted(rows, key=lambda row: row["units_30d"], reverse=True)[:10],
        "stockout_events": sum(int(row["stockout_events"] or 0) for row in rows),
        "turnover_note": "Perputaran 30 hari dihitung sebagai HPP penjualan 30 hari dibagi nilai stok saat ini. Nilai stok saat ini menjadi proksi, bukan rata-rata persediaan historis.",
    }


def rfm_analysis() -> pd.DataFrame:
    with connect() as connection:
        rows = connection.execute(
            """SELECT c.id, c.name AS pelanggan, c.phone,
               MAX(t.created_at) AS pembelian_terakhir,
               COUNT(t.id) AS frekuensi,
               SUM(t.subtotal-t.discount_amount) AS nilai_bersih
               FROM customers c JOIN transactions t ON t.customer_id=c.id AND t.status='completed'
               WHERE c.store_id=? GROUP BY c.id ORDER BY pembelian_terakhir DESC""", (STORE_ID,),
        ).fetchall()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    today = business_today()
    frame["hari_sejak_transaksi"] = [
        max(0, (today - business_date(value)).days) for value in frame["pembelian_terakhir"]
    ]
    monetary_q75 = frame["nilai_bersih"].quantile(0.75)
    frame["segmen"] = "Pelanggan reguler"
    frame.loc[(frame["hari_sejak_transaksi"] <= 60) & (frame["frekuensi"] >= 3), "segmen"] = "Pelanggan aktif"
    frame.loc[(frame["hari_sejak_transaksi"] <= 60) & (frame["nilai_bersih"] >= monetary_q75) & (frame["frekuensi"] >= 2), "segmen"] = "Pembelian bernilai tinggi"
    frame.loc[(frame["hari_sejak_transaksi"] <= 30) & (frame["frekuensi"] >= 5) & (frame["nilai_bersih"] >= monetary_q75), "segmen"] = "Pelanggan setia bernilai tinggi"
    frame.loc[frame["frekuensi"] == 1, "segmen"] = "Pembelian pertama"
    frame.loc[(frame["hari_sejak_transaksi"] > 60) & (frame["frekuensi"] >= 2), "segmen"] = "Perlu perhatian"
    frame.loc[frame["hari_sejak_transaksi"] > 90, "segmen"] = "Tidak aktif 90+ hari"
    return frame

