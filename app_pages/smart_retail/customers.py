import pandas as pd
import streamlit as st

from smart_retail.db import initialize_database
from smart_retail.services import RetailError, list_customers, list_transactions, save_customer
from smart_retail.ui import empty_state, page_heading, rupiah, table


initialize_database()
page_heading("Pelanggan", "Simpan kontak seperlunya dan lihat pola pembelian dari transaksi yang tercatat.")

customers = list_customers()
mode = st.segmented_control("Tindakan pelanggan", ["Tambah pelanggan", "Edit pelanggan"], default="Tambah pelanggan")
selected = None
if mode == "Edit pelanggan" and customers:
    customer_id = st.selectbox(
        "Pilih pelanggan",
        [item["id"] for item in customers],
        format_func=lambda cid: next(f"{item['name']} · {item['phone'] or 'tanpa nomor'}" for item in customers if item["id"] == cid),
    )
    selected = next(item for item in customers if item["id"] == customer_id)
elif mode == "Edit pelanggan":
    st.info("Belum ada pelanggan untuk diedit.")

with st.form("customer_form"):
    name = st.text_input("Nama", value=selected["name"] if selected else "")
    phone = st.text_input("Nomor kontak (opsional)", value=selected["phone"] or "" if selected else "")
    email = st.text_input("Email (opsional)", value=selected["email"] or "" if selected else "")
    submit = st.form_submit_button("Simpan pelanggan", type="primary", disabled=(mode == "Edit pelanggan" and selected is None))
if submit:
    try:
        save_customer({"name": name, "phone": phone, "email": email}, selected["id"] if selected else None)
        st.session_state["customer_notice"] = f"Data {name.strip()} tersimpan."
        st.rerun()
    except RetailError as exc:
        st.error(str(exc))

notice = st.session_state.pop("customer_notice", None)
if notice:
    st.success(notice)

st.divider()
st.subheader("Daftar pelanggan")
search = st.text_input("Cari pelanggan", placeholder="Nama atau nomor kontak", key="customer_search")
rows = list_customers(search)
if rows:
    frame = pd.DataFrame(rows)
    frame["nilai_rata_rata"] = frame.apply(
        lambda row: round(row["total_spend"] / row["transaction_count"]) if row["transaction_count"] else 0,
        axis=1,
    )
    frame["kontak"] = frame["phone"].fillna("").map(lambda value: f"••••{value[-4:]}" if value else "Tidak dicatat")
    frame = frame[["name", "kontak", "transaction_count", "total_spend", "nilai_rata_rata", "last_purchase"]].rename(columns={
        "name": "Pelanggan", "kontak": "Kontak", "transaction_count": "Frekuensi transaksi",
        "total_spend": "Pembelian bersih (Rp)", "nilai_rata_rata": "Rata-rata transaksi (Rp)",
        "last_purchase": "Transaksi terakhir",
    })
    table(frame, "Frekuensi dan nilai pembelian pelanggan tersimpan")
    options = {item["name"]: item for item in rows}
    chosen_name = st.selectbox("Lihat riwayat pelanggan", list(options))
    chosen = options[chosen_name]
    st.subheader(f"Riwayat {chosen['name']}")
    history = list_transactions(customer_id=chosen["id"], limit=500)
    if history:
        tx_frame = pd.DataFrame(history)[["transaction_no", "created_at", "total", "payment_method", "status"]].rename(columns={
            "transaction_no": "Nomor transaksi", "created_at": "Waktu UTC", "total": "Total (Rp)",
            "payment_method": "Pembayaran", "status": "Status",
        })
        tx_frame["Status"] = tx_frame["Status"].map({"completed": "Selesai", "refunded": "Refund", "cancelled": "Dibatalkan"})
        table(tx_frame, f"Riwayat transaksi {chosen['name']}")
    else:
        empty_state("Riwayat akan muncul setelah pelanggan dipilih pada transaksi kasir.")
else:
    empty_state("Data pelanggan tercatat saat checkout atau melalui formulir di atas.")
