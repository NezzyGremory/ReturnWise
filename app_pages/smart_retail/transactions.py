import io

import pandas as pd
import streamlit as st

from smart_retail.db import initialize_database
from smart_retail.services import (
    PAYMENT_METHODS, RetailError, cancel_transaction, list_transactions,
    refund_transaction, transaction_detail,
)
from smart_retail.ui import date_window, empty_state, page_heading, receipt_html, rupiah, safe_csv_frame, table, timestamp


initialize_database()
page_heading("Transaksi", "Telusuri penjualan tersimpan, periksa rincian, dan catat pembatalan atau refund tanpa menghapus riwayat.")

with st.sidebar:
    st.subheader("Filter transaksi")
    start, end = date_window("transactions")
    method_label = st.selectbox("Metode pembayaran", ["Semua metode", *PAYMENT_METHODS])
    status_label = st.selectbox("Status", ["Semua status", "Selesai", "Refund", "Dibatalkan"])

search = st.text_input("Cari nomor transaksi atau pelanggan", key="transaction_search")
status_map = {"Semua status": None, "Selesai": "completed", "Refund": "refunded", "Dibatalkan": "cancelled"}
transactions = list_transactions(
    search=search,
    start_date=start.isoformat(),
    end_date=end.isoformat(),
    payment_method=None if method_label == "Semua metode" else method_label,
    status=status_map[status_label],
    limit=2000,
)

if transactions:
    visible = pd.DataFrame(transactions)[[
        "transaction_no", "created_at", "customer_name", "units_sold", "subtotal", "discount_amount", "tax_amount", "total", "payment_method", "status"
    ]].rename(columns={
        "transaction_no": "Nomor transaksi", "created_at": "Waktu UTC", "customer_name": "Pelanggan",
        "units_sold": "Unit", "subtotal": "Subtotal (Rp)", "discount_amount": "Diskon (Rp)",
        "tax_amount": "Pajak (Rp)", "total": "Total (Rp)", "payment_method": "Pembayaran", "status": "Status",
    })
    visible["Pelanggan"] = visible["Pelanggan"].fillna("Pelanggan umum")
    visible["Status"] = visible["Status"].map({"completed": "Selesai", "refunded": "Refund", "cancelled": "Dibatalkan"})
    table(visible, "Transaksi pada filter periode, metode pembayaran, dan status terpilih")
    total_net = sum(item["subtotal"] - item["discount_amount"] for item in transactions if item["status"] == "completed")
    st.write(f"Penjualan bersih transaksi selesai pada hasil ini: {rupiah(total_net)}. Refund dan pembatalan tetap tampil di riwayat.")
    buffer = io.StringIO(newline="")
    safe_csv_frame(visible).to_csv(buffer, index=False)
    st.download_button("Ekspor hasil CSV", buffer.getvalue().encode("utf-8-sig"), "transaksi.csv", "text/csv")

    st.divider()
    st.subheader("Rincian transaksi")
    tx_options = {f"{item['transaction_no']} · {rupiah(item['total'])} · {item['status']}": item["id"] for item in transactions}
    chosen_label = st.selectbox("Pilih transaksi", list(tx_options))
    detail = transaction_detail(tx_options[chosen_label])
    if detail:
        tx = detail["transaction"]
        meta_a, meta_b, meta_c = st.columns(3)
        meta_a.metric("Total", rupiah(tx["total"]))
        meta_b.metric("Pembayaran", tx["payment_method"])
        meta_c.metric("Status", {"completed": "Selesai", "refunded": "Refund", "cancelled": "Dibatalkan"}.get(tx["status"], tx["status"]))
        lines = pd.DataFrame([{
            "SKU": item["sku_snapshot"], "Produk": item["product_name_snapshot"],
            "Kuantitas": item["quantity"], "Harga satuan (Rp)": item["unit_price"],
            "Modal satuan (Rp)": item["unit_cost"], "Jumlah (Rp)": item["line_total"],
        } for item in detail["items"]])
        table(lines, "Produk dan harga yang tersimpan pada transaksi terpilih")
        st.write(f"Dibuat {timestamp(tx['created_at'])}. Diskon Rp {tx['discount_amount']:,}; pajak Rp {tx['tax_amount']:,}. Alasan status: {tx['reason'] or 'Tidak ada'}".replace(",", "."))
        st.download_button(
            "Unduh struk untuk dicetak",
            receipt_html(detail),
            f"struk-{tx['transaction_no']}.html",
            "text/html",
        )
        if tx["status"] == "completed":
            with st.expander("Refund atau batalkan transaksi", expanded=False):
                st.warning("Tindakan ini mengubah status transaksi dan tidak menghapus riwayat. Pilih apakah barang kembali layak dijual.", title="Periksa barang sebelum melanjutkan")
                action = st.selectbox("Tindakan", ["Refund penuh", "Batalkan transaksi"], key=f"reverse_action_{tx['id']}")
                restock = st.checkbox("Barang kembali ke stok", value=True, key=f"restock_{tx['id']}")
                reason = st.text_input("Alasan", key=f"reverse_reason_{tx['id']}")
                confirmed = st.checkbox("Saya memahami status transaksi dan stok akan diperbarui.", key=f"reverse_confirm_{tx['id']}")
                if st.button("Simpan perubahan transaksi", disabled=not confirmed or not reason.strip(), key=f"reverse_submit_{tx['id']}"):
                    try:
                        if action == "Refund penuh":
                            refund_transaction(tx["id"], reason, restock=restock)
                        else:
                            cancel_transaction(tx["id"], reason, restock=restock)
                        st.session_state["transaction_action_notice"] = f"Transaksi {tx['transaction_no']} diperbarui."
                        st.rerun()
                    except RetailError as exc:
                        st.error(str(exc))
else:
    empty_state("Tidak ada transaksi yang cocok dengan periode dan filter saat ini.", title="Transaksi tidak ditemukan")

notice = st.session_state.pop("transaction_action_notice", None)
if notice:
    st.success(notice)
