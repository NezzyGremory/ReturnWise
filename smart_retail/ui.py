"""Small presentation helpers shared by Smart Retail pages."""

from __future__ import annotations

from datetime import date, datetime
from html import escape
from typing import Any

import pandas as pd
import streamlit as st
from smart_retail.timeutils import BUSINESS_TIMEZONE, business_today


def rupiah(value: Any) -> str:
    try:
        return "Rp " + f"{int(round(float(value or 0))):,}".replace(",", ".")
    except (TypeError, ValueError, OverflowError):
        return "Rp 0"


def page_heading(title: str, description: str) -> None:
    st.title(title)
    st.write(description)
    st.write("Smart Retail · mode lokal · data disimpan pada perangkat ini")


def table(frame: pd.DataFrame | list[dict[str, Any]], alt: str, *, hide_index: bool = True, column_config=None) -> None:
    if isinstance(frame, list):
        frame = pd.DataFrame(frame)
    st.dataframe(frame, hide_index=hide_index, column_config=column_config, alt=alt, width="stretch")


def safe_csv_value(value: Any) -> Any:
    """Keep untrusted text from being interpreted as a spreadsheet formula."""
    if not isinstance(value, str):
        return value
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def safe_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(safe_csv_value)
    return result


def empty_state(message: str, *, title: str = "Belum ada data") -> None:
    st.info(message, title=title)


def date_window(prefix: str = "periode") -> tuple[date, date]:
    period = st.selectbox(
        "Periode",
        ["Hari ini", "7 hari terakhir", "30 hari terakhir", "Rentang tanggal"],
        index=2,
        key=f"{prefix}_period",
    )
    today = business_today()
    if period == "Hari ini":
        return today, today
    if period == "7 hari terakhir":
        from datetime import timedelta
        return today - timedelta(days=6), today
    if period == "Rentang tanggal":
        selected = st.date_input(
            "Pilih rentang tanggal",
            value=(today.replace(day=1), today),
            key=f"{prefix}_custom_range",
        )
        if isinstance(selected, tuple) and len(selected) == 2:
            return selected[0], selected[1]
        return selected, selected
    from datetime import timedelta
    return today - timedelta(days=29), today


def receipt_html(detail: dict[str, Any]) -> str:
    tx = detail["transaction"]
    item_rows = "".join(
        "<tr>"
        f"<td>{escape(item['product_name_snapshot'])}<br><small>{escape(item['sku_snapshot'])}</small></td>"
        f"<td>{item['quantity']}</td><td>Rp {item['unit_price']:,}</td><td>Rp {item['line_total']:,}</td>"
        "</tr>".replace(",", ".")
        for item in detail["items"]
    )
    stamp = escape(timestamp(tx["created_at"]))
    customer = escape(tx.get("customer_name") or "Pelanggan umum")
    return f"""<!doctype html><html lang="id"><meta charset="utf-8"><title>Struk {escape(tx['transaction_no'])}</title>
    <style>body{{font:14px/1.45 system-ui,sans-serif;color:#1b2720;max-width:640px;margin:24px auto;padding:20px}}h1{{font-size:20px;margin:0 0 4px}}p{{margin:4px 0}}table{{width:100%;border-collapse:collapse;margin:18px 0}}th,td{{text-align:left;border-bottom:1px solid #c6d0c6;padding:8px 4px}}th{{font-size:12px}}.total{{font-size:18px;font-weight:700}}@media print{{button{{display:none}}body{{margin:0;max-width:none}}}}</style>
    <h1>Smart Retail</h1><p>Struk transaksi</p><hr><p><b>{escape(tx['transaction_no'])}</b></p><p>{stamp}</p><p>{customer}</p>
    <table><thead><tr><th>Produk</th><th>Qty</th><th>Harga</th><th>Jumlah</th></tr></thead><tbody>{item_rows}</tbody></table>
    <p>Subtotal: Rp {tx['subtotal']:,}</p><p>Diskon: Rp {tx['discount_amount']:,}</p><p>Pajak: Rp {tx['tax_amount']:,}</p>
    <p class="total">Total: Rp {tx['total']:,}</p><p>Metode pembayaran: {escape(tx['payment_method'])}</p>
    <p>Dibayar: Rp {tx['amount_paid']:,}</p><p>Kembalian: Rp {tx['change_amount']:,}</p><hr>
    <p>Terima kasih.</p><button onclick="window.print()">Cetak struk</button></html>""".replace(",", ".")


def timestamp(value: Any) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(BUSINESS_TIMEZONE).strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(value)
