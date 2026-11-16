"""Deterministic retail intelligence with temporal validation and explicit data limits."""

from __future__ import annotations

from datetime import timedelta
import json
import math
from typing import Any

import pandas as pd

from smart_retail.analytics import allocated_sales_items
from smart_retail.db import atomic, connect
from smart_retail.services import STORE_ID, now_iso
from smart_retail.timeutils import business_date, business_today, utc_date_bounds


MODEL_NAME = "Moving average 7 hari"
MODEL_VERSION = "baseline-1"
MIN_HISTORY_DAYS = 28


def _daily_frame(product_id: int | None = None, category_id: int | None = None) -> pd.DataFrame:
    if product_id is not None and category_id is not None:
        raise ValueError("Pilih produk atau kategori, bukan keduanya.")
    rows = allocated_sales_items()
    if product_id is not None:
        rows = [row for row in rows if row["product_id"] == product_id]
    if category_id is not None:
        rows = [row for row in rows if row["category_id"] == category_id]
    if rows:
        frame = pd.DataFrame(rows).groupby("sale_date", as_index=False).agg(unit=("quantity", "sum"), nilai=("net_sales", "sum"))
        frame = frame.rename(columns={"sale_date": "tanggal"})
    else:
        frame = pd.DataFrame(columns=["tanggal", "unit", "nilai"])
    if frame.empty:
        return frame
    frame["tanggal"] = pd.to_datetime(frame["tanggal"])
    frame = frame.set_index("tanggal").sort_index()
    full_index = pd.date_range(frame.index.min(), max(frame.index.max(), pd.Timestamp(business_today())), freq="D")
    return frame.reindex(full_index, fill_value=0).rename_axis("tanggal").reset_index()


def _metric(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    mae = sum(errors) / len(errors) if errors else 0.0
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(errors)) if errors else 0.0
    denom = sum(abs(v) for v in actual)
    wape = (sum(errors) / denom) if denom else None
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "WAPE": round(wape, 4) if wape is not None else None}


def _moving_average(values: list[float], window: int = 7) -> float:
    recent = values[-window:]
    return sum(recent) / len(recent) if recent else 0.0


def run_sales_forecast(
    horizon: int = 7, product_id: int | None = None, category_id: int | None = None
) -> dict[str, Any]:
    if horizon not in (7, 30):
        raise ValueError("Horizon harus 7 atau 30 hari.")
    frame = _daily_frame(product_id, category_id)
    if frame.empty:
        return {"ready": False, "reason": "Belum ada transaksi tersimpan.", "history_days": 0}
    history_days = int(len(frame))
    active_days = int((frame["unit"] > 0).sum())
    if history_days < MIN_HISTORY_DAYS or active_days < 7:
        return {
            "ready": False,
            "reason": "Belum cukup data untuk menghasilkan prediksi yang dapat diandalkan. Diperlukan minimal 28 hari kalender dan penjualan pada sedikitnya 7 hari.",
            "history_days": history_days,
            "active_days": active_days,
        }

    units = frame["unit"].astype(float).tolist()
    revenue = frame["nilai"].fillna(0).astype(float).tolist()
    holdout = min(7, max(1, len(units) // 5))
    unit_actual, unit_pred = [], []
    value_actual, value_pred = [], []
    for i in range(len(units) - holdout, len(units)):
        unit_pred.append(_moving_average(units[:i]))
        unit_actual.append(units[i])
        value_pred.append(_moving_average(revenue[:i]))
        value_actual.append(revenue[i])
    metrics = {"unit": _metric(unit_actual, unit_pred), "revenue": _metric(value_actual, value_pred)}
    last_values_units, last_values_revenue = units[-7:], revenue[-7:]
    next_date = business_today() + timedelta(days=1)
    forecast = []
    for offset in range(horizon):
        predicted_units = max(0.0, _moving_average(last_values_units))
        predicted_revenue = max(0.0, _moving_average(last_values_revenue))
        target = next_date + timedelta(days=offset)
        forecast.append({"tanggal": target.isoformat(), "unit_prediksi": round(predicted_units, 2), "penjualan_prediksi": round(predicted_revenue, 2)})
        last_values_units = [*last_values_units, predicted_units][-7:]
        last_values_revenue = [*last_values_revenue, predicted_revenue][-7:]

    scope_key = f"product:{product_id}" if product_id is not None else f"category:{category_id}" if category_id is not None else "store"
    with connect() as connection, atomic(connection):
        connection.execute(
            """INSERT INTO model_registry(store_id, model_name, model_version, scope_key, metrics_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(store_id, model_name, model_version, scope_key)
               DO UPDATE SET metrics_json=excluded.metrics_json, created_at=excluded.created_at""",
            (STORE_ID, MODEL_NAME, MODEL_VERSION, scope_key, json.dumps(metrics), now_iso()),
        )
        model = connection.execute(
            "SELECT id FROM model_registry WHERE store_id=? AND model_name=? AND model_version=? AND scope_key=?",
            (STORE_ID, MODEL_NAME, MODEL_VERSION, scope_key),
        ).fetchone()
        model_id = model["id"]
        created = now_iso()
        for item in forecast:
            connection.execute(
                """INSERT INTO sales_forecasts(store_id, product_id, category_id, model_id, forecast_date,
                   predicted_units, predicted_revenue, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (STORE_ID, product_id, category_id, model_id, item["tanggal"], item["unit_prediksi"], item["penjualan_prediksi"], created),
            )
        connection.execute(
            "INSERT INTO prediction_history(store_id, model_id, product_id, category_id, prediction_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (STORE_ID, model_id, product_id, category_id,
             json.dumps({"horizon": horizon, "forecast": forecast, "metrics": metrics}, ensure_ascii=False), created),
        )
    actual_frame = frame.tail(30)[["tanggal", "unit", "nilai"]].rename(columns={"unit": "unit_aktual", "nilai": "penjualan_aktual"})
    return {
        "ready": True,
        "history_days": history_days,
        "active_days": active_days,
        "metrics": metrics,
        "forecast": forecast,
        "actual": actual_frame,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at": created,
        "method": "Rata-rata bergerak 7 hari. Prediksi berulang memakai rata-rata 7 nilai terakhir.",
        "limitations": "Evaluasi menggunakan holdout time-based hingga 7 hari terakhir; jumlah hari ditentukan panjang riwayat. Hari tanpa transaksi tercatat dihitung nol; jika toko tidak mencatat transaksi secara rutin, hasil dapat bias. Baseline ini bukan jaminan penjualan.",
    }


def inventory_recommendations() -> list[dict[str, Any]]:
    today = business_today()
    lower, upper = utc_date_bounds(today - timedelta(days=29), today)
    with connect() as connection:
        products = connection.execute(
            """SELECT p.id, p.sku, p.name, p.stock, p.minimum_stock, p.unit, p.cost_price, p.supplier_id,
               COALESCE(s.lead_time_days, 7) AS lead_time_days,
               MIN(CASE WHEN m.movement_type='sale' AND t.status='completed' THEN m.created_at END) AS first_sale_at,
               COALESCE(SUM(CASE WHEN m.movement_type='sale' AND t.status='completed'
                   AND m.created_at>=? AND m.created_at<? THEN -m.quantity_delta ELSE 0 END),0) AS units_30d
               FROM products p LEFT JOIN suppliers s ON s.id=p.supplier_id
               LEFT JOIN inventory_movements m ON m.product_id=p.id
               LEFT JOIN transactions t ON t.id=m.transaction_id
               WHERE p.store_id=? AND p.is_active=1 GROUP BY p.id ORDER BY p.name COLLATE NOCASE""",
            (lower, upper, STORE_ID),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in products:
        product = dict(row)
        first_sale = business_date(product["first_sale_at"]) if product["first_sale_at"] else None
        product["first_sale"] = first_sale.isoformat() if first_sale else None
        product.pop("first_sale_at", None)
        history_days = (today - first_sale).days + 1 if first_sale else 0
        observed_days = min(30, history_days) if history_days else 30
        daily_demand = product["units_30d"] / observed_days
        lead_time = int(product["lead_time_days"] or 0)
        if not first_sale or history_days < 14:
            status = "Perlu pemantauan" if product["stock"] <= product["minimum_stock"] else "Data permintaan terbatas"
            suggested = None
            reorder_point = None
            basis = f"Riwayat penjualan {history_days} hari; minimal 14 hari diperlukan untuk estimasi permintaan."
        else:
            # A one-standard-deviation safety buffer is transparent, bounded, and based on observed 30-day daily demand.
            std_daily = math.sqrt(max(daily_demand, 0.0))
            safety_stock = math.ceil(std_daily * math.sqrt(max(lead_time, 1)))
            reorder_point = math.ceil(daily_demand * lead_time + safety_stock)
            target = math.ceil(daily_demand * (lead_time + 7) + safety_stock)
            suggested = max(0, target - product["stock"])
            if product["stock"] == 0:
                status = "Risiko kehabisan stok"
            elif target > 0 and product["stock"] > target * 2:
                status = "Indikasi stok berlebih"
            elif product["stock"] <= reorder_point or product["stock"] <= product["minimum_stock"]:
                status = "Disarankan restock" if suggested else "Perlu pemantauan"
            else:
                status = "Stok masih aman"
            basis = (f"Rata-rata {daily_demand:.2f} unit/hari dari {product['units_30d']} unit selama {observed_days} hari; "
                     f"lead time {lead_time} hari{' (default)' if not product['supplier_id'] else ''}; "
                     f"safety stock {safety_stock} unit; titik pesan {reorder_point} unit; target {target} unit.")
        result.append({**product, "status": status, "daily_demand": round(daily_demand, 2),
                       "reorder_point": reorder_point, "suggested_quantity": suggested,
                       "history_days": history_days, "basis": basis})
    return result

