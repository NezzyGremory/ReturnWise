import pandas as pd
import streamlit as st

from smart_retail.analytics import inventory_analysis
from smart_retail.db import initialize_database
from smart_retail.services import RetailError, adjust_stock, list_movements, list_products, list_receipts, list_suppliers, stock_in
from smart_retail.ui import empty_state, page_heading, rupiah, table, timestamp


initialize_database()
page_heading("Inventaris", "Catat setiap stok masuk dan penyesuaian dengan alasan yang dapat ditelusuri.")
analysis = inventory_analysis()
with st.container(horizontal=True):
    st.metric("Nilai stok aktif", rupiah(analysis["current_inventory_value"]), border=True,
              help="Jumlah stok saat ini dikalikan harga modal terakhir yang tercatat.")
    st.metric("Stok hampir habis", sum(1 for row in analysis["products"] if 0 < row["stock"] <= row["minimum_stock"]), border=True)
    st.metric("Stok habis", sum(1 for row in analysis["products"] if row["stock"] == 0), border=True)

products = list_products(active_only=True)
suppliers = list_suppliers(active_only=True)
if products:
    product_options = {f"{item['name']} · {item['sku']} · stok {item['stock']} {item['unit']}": item for item in products}
    supplier_options = {"Tanpa supplier": None, **{item["name"]: item["id"] for item in suppliers}}
    stock_col, adjust_col = st.columns(2, gap="large")
    with stock_col:
        st.subheader("Stok masuk")
        with st.form("stock_receipt_form"):
            product_name = st.selectbox("Produk", list(product_options), key="stock_in_product")
            quantity = st.number_input("Jumlah diterima", min_value=1, step=1, value=1)
            supplier_name = st.selectbox("Supplier", list(supplier_options), key="stock_in_supplier")
            unit_cost = st.number_input("Harga modal per unit (Rp)", min_value=0, step=1000, value=0,
                                        help="Isi 0 untuk mempertahankan harga modal produk saat ini.")
            reason = st.text_input("Alasan / nomor dokumen", placeholder="Contoh: pembelian PO-104")
            submit_receipt = st.form_submit_button("Simpan stok masuk", type="primary", width="stretch")
        if submit_receipt:
            try:
                chosen = product_options[product_name]
                actual_cost = unit_cost if unit_cost else None
                receipt_no = stock_in(chosen["id"], quantity, reason, supplier_options[supplier_name], actual_cost)
                st.success(f"Penerimaan {receipt_no} tersimpan.")
                st.rerun()
            except RetailError as exc:
                st.error(str(exc))

    with adjust_col:
        st.subheader("Penyesuaian stok")
        with st.form("stock_adjust_form"):
            adjust_product_name = st.selectbox("Produk", list(product_options), key="adjust_product")
            adjust_to = st.number_input("Stok fisik saat ini", min_value=0, step=1, value=0)
            adjust_reason = st.text_input("Alasan penyesuaian", placeholder="Contoh: hasil hitung fisik")
            submit_adjust = st.form_submit_button("Catat penyesuaian", width="stretch")
        if submit_adjust:
            try:
                chosen = product_options[adjust_product_name]
                movement_id = adjust_stock(chosen["id"], adjust_to, adjust_reason)
                st.success(f"Penyesuaian tersimpan sebagai pergerakan #{movement_id}.")
                st.rerun()
            except RetailError as exc:
                st.error(str(exc))
else:
    st.info("Tambahkan produk sebelum mencatat stok.", title="Inventaris kosong")
    st.page_link("app_pages/smart_retail/products.py", label="Tambah produk", icon=":material/add_box:")

st.divider()
st.subheader("Posisi stok")
if analysis["products"]:
    frame = pd.DataFrame(analysis["products"])[["sku", "name", "stock", "minimum_stock", "unit", "cost_price", "units_30d"]]
    frame = frame.rename(columns={
        "sku": "SKU", "name": "Produk", "stock": "Stok saat ini", "minimum_stock": "Stok minimum",
        "unit": "Satuan", "cost_price": "Harga modal (Rp)", "units_30d": "Terjual 30 hari",
    })
    table(frame, "Stok produk, batas minimum, dan unit terjual dalam 30 hari")
else:
    empty_state("Daftar stok aktif akan muncul setelah produk dibuat.")

movement_rows = list_movements(limit=200)
st.subheader("Riwayat pergerakan stok")
if movement_rows:
    frame = pd.DataFrame(movement_rows)
    frame["created_at"] = frame["created_at"].map(timestamp)
    frame = frame[["created_at", "product_name", "sku", "movement_type", "quantity_delta", "stock_after", "reason", "transaction_no", "receipt_no"]]
    frame = frame.rename(columns={
        "created_at": "Waktu", "product_name": "Produk", "sku": "SKU", "movement_type": "Jenis",
        "quantity_delta": "Perubahan unit", "stock_after": "Stok setelah", "reason": "Alasan",
        "transaction_no": "Transaksi", "receipt_no": "Penerimaan",
    })
    table(frame, "Riwayat pergerakan inventaris terbaru")
else:
    empty_state("Setiap stok masuk, penjualan, retur, dan penyesuaian akan tercatat di sini.")

with st.expander("Riwayat penerimaan supplier", expanded=False):
    receipts = list_receipts(limit=200)
    if receipts:
        frame = pd.DataFrame(receipts)[["receipt_no", "created_at", "supplier_name", "sku", "product_name", "quantity", "unit_cost", "reason"]]
        frame = frame.rename(columns={
            "receipt_no": "Nomor penerimaan", "created_at": "Waktu", "supplier_name": "Supplier", "sku": "SKU",
            "product_name": "Produk", "quantity": "Jumlah", "unit_cost": "Harga modal (Rp)", "reason": "Alasan",
        })
        table(frame, "Riwayat stok masuk berdasarkan supplier")
    else:
        empty_state("Belum ada penerimaan stok yang tersimpan.")
