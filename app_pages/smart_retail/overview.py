import streamlit as st

from smart_retail.analytics import dashboard_data
from smart_retail.db import initialize_database
from smart_retail.ui import date_window, empty_state, page_heading, rupiah, table


initialize_database()
page_heading("Ringkasan toko", "Pantau penjualan, aktivitas kasir, dan kebutuhan stok dari catatan toko.")

with st.sidebar:
    st.subheader("Periode ringkasan")
    start, end = date_window("overview")

data = dashboard_data(start, end)
with st.container(horizontal=True):
    st.metric("Penjualan bersih", rupiah(data["net_sales"]), border=True,
              help="Subtotal transaksi selesai setelah diskon, tanpa pajak.")
    st.metric("Transaksi selesai", f"{data['transactions']:,}", border=True)
    st.metric("Produk terjual", f"{data['units']:,}", border=True)
    st.metric("Rata-rata transaksi", rupiah(data["avg_order"]), border=True)

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("Tren penjualan")
    st.write(f"{start:%d %b %Y} sampai {end:%d %b %Y}. Sumber: transaksi berstatus selesai.")
    if data["daily"].empty:
        empty_state("Belum ada transaksi pada periode ini. Penjualan akan muncul setelah transaksi kasir tersimpan.")
    else:
        st.line_chart(data["daily"], x="tanggal", y="penjualan", x_label="Tanggal",
                      y_label="Penjualan bersih (Rp)", alt="Penjualan bersih harian pada periode terpilih")

with right:
    st.subheader("Kondisi persediaan")
    with st.container(border=True):
        st.metric("Produk aktif", f"{data['products']:,}")
        stock_a, stock_b = st.columns(2)
        stock_a.metric("Stok hampir habis", f"{data['low_count']:,}")
        stock_b.metric("Stok habis", f"{data['out_count']:,}")
        st.write(f"Nilai persediaan berdasarkan harga modal: {rupiah(data['inventory_value'])}.")
        st.page_link("app_pages/smart_retail/inventory.py", label="Buka inventaris", icon=":material/warehouse:")

st.divider()
product_col, activity_col = st.columns([1, 1.25], gap="large")
with product_col:
    st.subheader("Produk terlaris")
    if data["top_products"]:
        import pandas as pd
        frame = pd.DataFrame(data["top_products"]).rename(columns={
            "product": "Produk", "units": "Unit terjual", "net_sales": "Penjualan bersih (Rp)"
        })
        table(frame, "Produk dengan unit penjualan terbanyak pada periode terpilih")
    else:
        empty_state("Belum ada produk terjual pada periode ini.")

with activity_col:
    st.subheader("Transaksi terbaru")
    if data["recent"]:
        import pandas as pd
        frame = pd.DataFrame(data["recent"])[["transaction_no", "created_at", "customer_name", "total", "status"]]
        frame = frame.rename(columns={
            "transaction_no": "Nomor transaksi", "created_at": "Waktu UTC", "customer_name": "Pelanggan",
            "total": "Total (Rp)", "status": "Status",
        })
        frame["Pelanggan"] = frame["Pelanggan"].fillna("Pelanggan umum")
        table(frame, "Delapan transaksi terakhir")
        st.page_link("app_pages/smart_retail/transactions.py", label="Lihat semua transaksi", icon=":material/receipt_long:")
    else:
        empty_state("Transaksi tersimpan akan tampil di sini.")

if data["out_count"] or data["low_count"]:
    st.warning(
        f"Ada {data['out_count']} produk habis dan {data['low_count']} produk pada atau di bawah stok minimum. Periksa daftar inventaris sebelum transaksi berikutnya.",
        title="Perhatian stok",
    )
