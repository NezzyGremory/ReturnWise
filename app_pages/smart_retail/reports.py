import io
from datetime import timedelta

import pandas as pd
import streamlit as st

from smart_retail.analytics import category_performance, dashboard_data, inventory_analysis, product_performance
from smart_retail.db import initialize_database
from smart_retail.ui import date_window, empty_state, page_heading, rupiah, safe_csv_frame, table


initialize_database()
page_heading("Laporan dan analisis", "Analisis penjualan, produk, dan persediaan dari transaksi yang tersimpan.")
with st.sidebar:
    st.subheader("Periode laporan")
    start, end = date_window("reports")

summary = dashboard_data(start, end)
products = product_performance(start, end)
categories = category_performance(start, end)
period_days = (end - start).days + 1
previous_end = start - timedelta(days=1)
previous_start = previous_end - timedelta(days=period_days - 1)
previous = dashboard_data(previous_start, previous_end)

def delta_percent(current, prior):
    return round((current - prior) / prior * 100, 1) if prior else None

with st.container(horizontal=True):
    sales_delta = delta_percent(summary["net_sales"], previous["net_sales"])
    st.metric("Penjualan bersih", rupiah(summary["net_sales"]), delta=sales_delta,
              border=True, help="Subtotal setelah diskon, sebelum pajak.")
    gross_profit = int(products["laba_kotor_estimasi"].sum()) if not products.empty else 0
    st.metric("Laba kotor estimasi", rupiah(gross_profit), border=True,
              help="Penjualan item setelah alokasi diskon dikurangi modal saat barang dijual.")
    transaction_delta = delta_percent(summary["transactions"], previous["transactions"])
    st.metric("Transaksi selesai", f"{summary['transactions']:,}", delta=transaction_delta,
              border=True)

st.write(
    f"Periode pembanding: {previous_start:%d %b %Y} sampai {previous_end:%d %b %Y}. "
    "Perubahan persentase tidak ditampilkan jika nilai periode pembanding nol."
)

st.subheader("Tren penjualan")
if summary["daily"].empty:
    empty_state("Belum ada penjualan pada periode terpilih.")
else:
    st.line_chart(summary["daily"], x="tanggal", y="penjualan", x_label="Tanggal",
                  y_label="Penjualan bersih (Rp)", alt="Penjualan bersih harian tanpa pajak")

product_col, category_col = st.columns(2, gap="large")
with product_col:
    st.subheader("Kinerja produk")
    if products.empty:
        empty_state("Produk akan dibandingkan setelah transaksi selesai tercatat.")
    else:
        table(products.rename(columns={
            "sku": "SKU", "produk": "Produk", "unit_terjual": "Unit terjual",
            "penjualan_bersih": "Penjualan bersih (Rp)", "harga_pokok": "HPP (Rp)",
            "laba_kotor_estimasi": "Laba kotor estimasi (Rp)",
        }).drop(columns=["product_id"]), "Penjualan bersih dan estimasi laba kotor per produk")
        st.bar_chart(products.head(12), x="produk", y="unit_terjual", x_label="Produk",
                     y_label="Unit terjual", alt="Produk dengan unit terjual terbanyak")
        st.write("**Kontribusi laba kotor tertinggi**")
        profit_leaders = products.nlargest(5, "laba_kotor_estimasi")
        table(profit_leaders[["produk", "unit_terjual", "penjualan_bersih", "laba_kotor_estimasi"]].rename(columns={
            "produk": "Produk", "unit_terjual": "Unit", "penjualan_bersih": "Penjualan bersih (Rp)",
            "laba_kotor_estimasi": "Laba kotor estimasi (Rp)",
        }), "Lima produk dengan kontribusi laba kotor estimasi tertinggi")
        buffer = io.StringIO(newline="")
        safe_csv_frame(products).to_csv(buffer, index=False)
        st.download_button("Ekspor analisis produk", buffer.getvalue().encode("utf-8-sig"), "analisis-produk.csv", "text/csv")

with category_col:
    st.subheader("Kinerja kategori")
    if categories.empty:
        empty_state("Kategori penjualan muncul setelah produk berkategori terjual.")
    else:
        table(categories.rename(columns={
            "kategori": "Kategori", "unit_terjual": "Unit terjual",
            "penjualan_bersih": "Penjualan bersih (Rp)", "laba_kotor_estimasi": "Laba kotor estimasi (Rp)",
        }), "Penjualan bersih dan laba kotor estimasi per kategori")
        st.bar_chart(categories, x="kategori", y="penjualan_bersih", x_label="Kategori",
                     y_label="Penjualan bersih (Rp)", alt="Perbandingan penjualan bersih antar kategori")

st.divider()
st.subheader("Pergerakan persediaan")
inventory = inventory_analysis()
turnover_col, slow_col = st.columns([0.75, 1.25], gap="large")
with turnover_col:
    if inventory["turnover_proxy"] is None:
        st.metric("Proksi perputaran 30 hari", "Belum tersedia")
    else:
        st.metric("Proksi perputaran 30 hari", f"{inventory['turnover_proxy']:.2f}x")
    st.write(inventory["turnover_note"])
    st.write(f"Kejadian stok mencapai nol yang tercatat: {inventory['stockout_events']}.")
with slow_col:
    st.write("**Produk tanpa penjualan tercatat dalam 30 hari**")
    if inventory["slow_moving"]:
        frame = pd.DataFrame(inventory["slow_moving"])[["sku", "name", "stock", "minimum_stock"]].rename(columns={
            "sku": "SKU", "name": "Produk", "stock": "Stok saat ini", "minimum_stock": "Stok minimum",
        })
        table(frame, "Produk tanpa pergerakan penjualan yang tercatat dalam 30 hari")
    else:
        empty_state("Semua produk aktif memiliki penjualan tercatat dalam 30 hari terakhir.")

with st.expander("Cara menghitung dan batasan", expanded=False):
    st.write("**Penjualan bersih** adalah subtotal barang setelah diskon transaksi dan sebelum pajak. Pajak tidak dihitung sebagai pendapatan toko.")
    st.write("**Laba kotor estimasi** menjumlahkan nilai item setelah alokasi diskon proporsional lalu mengurangi harga modal yang tersimpan pada saat penjualan. Hasil ini belum mengurangi biaya operasional, retur di luar aplikasi, atau biaya pembayaran.")
    st.write("**Performa kategori dan produk** hanya memakai transaksi berstatus selesai pada rentang tanggal terpilih. Hari tanpa transaksi tidak ditampilkan sebagai angka nol pada tren.")
    st.write("**Slow moving** berarti tidak ada penjualan tersimpan pada 30 hari kalender terakhir. Riwayat dimulai ketika data dicatat di aplikasi.")
