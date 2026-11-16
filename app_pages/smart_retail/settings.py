import json

import pandas as pd
import streamlit as st

from smart_retail.backup import create_backup, restore_backup
from smart_retail.db import DB_PATH, initialize_database
from smart_retail.services import RetailError, get_setting, list_audit_logs, set_setting
from smart_retail.ui import empty_state, page_heading, table, timestamp


initialize_database()
page_heading("Cadangkan data", "Atur pajak yang tercatat pada transaksi, ekspor cadangan, dan pulihkan salinan yang sudah divalidasi.")

with st.form("tax_settings"):
    current_tax = float(get_setting("tax_rate", "0") or 0)
    tax_rate = st.number_input("Pajak penjualan (%)", min_value=0.0, max_value=100.0, step=0.5, value=current_tax,
                               help="Diterapkan pada transaksi baru setelah diskon. Ini hanya pencatatan, bukan pelaporan pajak.")
    save_tax = st.form_submit_button("Simpan pengaturan pajak", type="primary")
if save_tax:
    try:
        set_setting("tax_rate", str(tax_rate))
        st.success("Tarif pajak tersimpan dan berlaku untuk transaksi berikutnya.")
    except RetailError as exc:
        st.error(str(exc))

st.divider()
st.subheader("Cadangan lokal")
st.write("File cadangan berisi produk, stok, pelanggan, supplier, transaksi, riwayat pergerakan, pengaturan, dan audit log pada database Smart Retail ini.")
try:
    backup_bytes = create_backup()
except RetailError as exc:
    st.warning(str(exc), title="Backup JSON tidak dapat dibuat")
else:
    st.download_button("Unduh backup JSON", data=backup_bytes, file_name="smart-retail-backup.json", mime="application/json", type="primary")
st.write(f"Database aktif: `{DB_PATH}`. Backup ini tidak menyinkronkan data ke perangkat lain.")

st.subheader("Pulihkan backup")
with st.container(border=True):
    st.write("**Perhatian sebelum memulihkan backup**")
    st.write("Pemulihan mengganti seluruh data Smart Retail lokal saat ini. Buat backup terlebih dahulu bila data yang ada masih diperlukan.")
upload = st.file_uploader("Pilih backup JSON", type=["json"])
if upload:
    confirm_restore = st.checkbox("Saya memahami bahwa data lokal akan diganti dengan isi backup ini.")
    if st.button("Validasi dan pulihkan backup", type="primary", disabled=not confirm_restore):
        try:
            counts = restore_backup(upload.getvalue())
            st.session_state["restore_notice"] = {
                "message": "Backup dipulihkan. Jumlah baris yang dimuat tercatat di bawah.",
                "counts": counts,
            }
            st.rerun()
        except RetailError as exc:
            st.error(str(exc))

notice = st.session_state.pop("restore_notice", None)
if notice:
    st.success(notice["message"])
    table(pd.DataFrame([{"Tabel": name, "Baris dipulihkan": count} for name, count in notice["counts"].items()]),
          "Jumlah baris per tabel yang dipulihkan dari backup")

st.divider()
st.subheader("Jejak perubahan terbaru")
logs = list_audit_logs(100)
if logs:
    frame = pd.DataFrame(logs)
    frame["created_at"] = frame["created_at"].map(timestamp)
    frame["details_json"] = frame["details_json"].map(lambda value: json.dumps(json.loads(value), ensure_ascii=False))
    frame = frame.rename(columns={
        "action": "Tindakan", "entity_type": "Jenis data", "entity_id": "ID data",
        "details_json": "Rincian", "created_at": "Waktu",
    })
    table(frame, "Jejak perubahan operasional terbaru")
else:
    empty_state("Perubahan produk, stok, transaksi, dan pengaturan akan tercatat di sini.")

st.info("Mode ini memakai SQLite lokal tanpa akun atau sinkronisasi antarperangkat. Jangan menaruh database pada folder bersama yang terbuka untuk umum.", title="Batasan mode lokal")
