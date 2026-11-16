import csv
import io

import pandas as pd
import streamlit as st

from smart_retail.db import initialize_database
from smart_retail.services import (
    RetailError, archive_product, create_category, import_products, list_categories,
    list_products, list_suppliers, save_product,
)
from smart_retail.ui import empty_state, page_heading, rupiah, safe_csv_value, table


initialize_database()
page_heading("Produk", "Kelola katalog, harga, pemasok, dan status produk.")

categories = list_categories()
suppliers = list_suppliers(active_only=True)
category_map = {item["name"]: item["id"] for item in categories}
supplier_map = {item["name"]: item["id"] for item in suppliers}

with st.expander("Tambah kategori", expanded=False):
    with st.form("new_category"):
        category_name = st.text_input("Nama kategori")
        category_submit = st.form_submit_button("Simpan kategori", type="primary")
    if category_submit:
        try:
            create_category(category_name)
            st.success("Kategori tersimpan.")
            st.rerun()
        except RetailError as exc:
            st.error(str(exc))

all_products = list_products(active_only=False)
form_mode = st.segmented_control("Tindakan produk", ["Tambah produk", "Edit produk"], default="Tambah produk")
selected_product = None
if form_mode == "Edit produk" and all_products:
    selected_product_id = st.selectbox(
        "Pilih produk untuk diedit",
        [item["id"] for item in all_products],
        format_func=lambda product_id: next(
            f"{item['name']} · {item['sku']}" for item in all_products if item["id"] == product_id
        ),
    )
    selected_product = next(item for item in all_products if item["id"] == selected_product_id)
elif form_mode == "Edit produk":
    st.info("Belum ada produk untuk diedit.")

with st.form("product_form"):
    values = selected_product or {}
    first, second = st.columns(2)
    with first:
        sku = st.text_input("SKU", value=values.get("sku", ""))
        name = st.text_input("Nama produk", value=values.get("name", ""))
        barcode = st.text_input("Barcode", value=values.get("barcode") or "")
        description = st.text_area("Deskripsi", value=values.get("description", ""), height=80)
        category_choices = ["Tanpa kategori", *category_map]
        current_category = values.get("category_name") if values.get("category_name") in category_map else "Tanpa kategori"
        category_name_value = st.selectbox("Kategori", category_choices, index=category_choices.index(current_category))
    with second:
        cost = st.number_input("Harga modal (Rp)", min_value=0, step=1000, value=int(values.get("cost_price", 0)))
        price = st.number_input("Harga jual (Rp)", min_value=0, step=1000, value=int(values.get("selling_price", 0)))
        stock = st.number_input("Stok saat ini", min_value=0, step=1, value=int(values.get("stock", 0)))
        minimum = st.number_input("Stok minimum", min_value=0, step=1, value=int(values.get("minimum_stock", 0)))
        unit = st.text_input("Satuan", value=values.get("unit", "pcs"))
        supplier_choices = ["Tanpa supplier", *supplier_map]
        current_supplier = values.get("supplier_name") if values.get("supplier_name") in supplier_map else "Tanpa supplier"
        supplier_name_value = st.selectbox("Supplier", supplier_choices, index=supplier_choices.index(current_supplier))
        active = st.checkbox("Produk aktif", value=bool(values.get("is_active", 1)))
    submit_product = st.form_submit_button(
        "Simpan perubahan" if selected_product else "Tambah produk",
        type="primary", width="stretch",
        disabled=(form_mode == "Edit produk" and selected_product is None),
    )

if submit_product:
    try:
        product_data = {
            "sku": sku, "name": name, "barcode": barcode, "description": description,
            "category_id": category_map.get(category_name_value),
            "supplier_id": supplier_map.get(supplier_name_value),
            "cost_price": cost, "selling_price": price, "stock": stock,
            "minimum_stock": minimum, "unit": unit, "is_active": active,
        }
        save_product(product_data, selected_product["id"] if selected_product else None)
        st.success("Produk tersimpan. Perubahan stok ikut dicatat pada riwayat inventaris.")
        st.rerun()
    except RetailError as exc:
        st.error(str(exc))

st.divider()
st.subheader("Daftar produk")
filter_col, category_filter_col, sort_col = st.columns([1.2, 0.9, 0.8], vertical_alignment="bottom")
with filter_col:
    search = st.text_input("Cari produk", placeholder="Nama, SKU, atau barcode", key="product_search")
with category_filter_col:
    category_filter_options = {"Semua kategori": None, **category_map}
    category_filter_name = st.selectbox("Filter kategori", list(category_filter_options))
with sort_col:
    sort_by = st.selectbox("Urutkan", ["Nama", "Stok terendah", "Harga tertinggi"])

products = list_products(search=search, category_id=category_filter_options[category_filter_name], active_only=False)
if sort_by == "Stok terendah":
    products.sort(key=lambda item: (item["stock"], item["name"].casefold()))
elif sort_by == "Harga tertinggi":
    products.sort(key=lambda item: (-item["selling_price"], item["name"].casefold()))

if products:
    visible = pd.DataFrame(products)[[
        "sku", "barcode", "name", "category_name", "cost_price", "selling_price", "stock", "minimum_stock", "unit", "supplier_name", "is_active"
    ]].rename(columns={
        "sku": "SKU", "barcode": "Barcode", "name": "Produk", "category_name": "Kategori",
        "cost_price": "Harga modal (Rp)", "selling_price": "Harga jual (Rp)", "stock": "Stok",
        "minimum_stock": "Minimum", "unit": "Satuan", "supplier_name": "Supplier", "is_active": "Aktif",
    })
    visible["Kategori"] = visible["Kategori"].fillna("Tanpa kategori")
    visible["Supplier"] = visible["Supplier"].fillna("Tanpa supplier")
    visible["Aktif"] = visible["Aktif"].map({1: "Aktif", 0: "Diarsipkan"})
    table(visible, "Daftar produk dan harga tersimpan")
    csv_rows = [{
        "sku": row["sku"], "barcode": row["barcode"] or "", "name": row["name"],
        "category_id": row["category_id"] or "", "description": row["description"],
        "cost_price": row["cost_price"], "selling_price": row["selling_price"],
        "stock": row["stock"], "minimum_stock": row["minimum_stock"], "unit": row["unit"],
        "supplier_id": row["supplier_id"] or "", "is_active": row["is_active"],
    } for row in products]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows({key: safe_csv_value(value) for key, value in row.items()} for row in csv_rows)
    st.download_button("Ekspor produk CSV", buffer.getvalue().encode("utf-8-sig"), "produk.csv", "text/csv")

    archive_options = {f"{item['name']} · {item['sku']}": item["id"] for item in products if item["is_active"]}
    if archive_options:
        with st.expander("Arsipkan produk", expanded=False):
            archive_name = st.selectbox("Produk yang akan diarsipkan", list(archive_options))
            confirmed_archive = st.checkbox("Saya memastikan produk ini tidak lagi ditawarkan.")
            if st.button("Arsipkan produk", disabled=not confirmed_archive):
                try:
                    archive_product(archive_options[archive_name])
                    st.success("Produk diarsipkan. Riwayat transaksi tetap tersimpan.")
                    st.rerun()
                except RetailError as exc:
                    st.error(str(exc))
else:
    empty_state("Belum ada produk yang cocok. Tambahkan produk atau ubah kata pencarian.")

st.divider()
st.subheader("Impor produk CSV")
st.write("Kolom wajib: sku, name, cost_price, selling_price. Kolom relasi memakai category_id dan supplier_id dari toko ini.")
upload = st.file_uploader("Pilih file CSV produk", type=["csv"])
if upload is not None:
    if st.button("Validasi dan impor CSV", type="primary"):
        try:
            content = upload.getvalue().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(content))
            if not reader.fieldnames or not {"sku", "name", "cost_price", "selling_price"}.issubset(reader.fieldnames):
                raise RetailError("CSV belum memiliki kolom wajib: sku, name, cost_price, selling_price.")
            result = import_products(list(reader))
            st.success(f"Impor selesai. {result['created']} produk baru, {result['updated']} produk diperbarui.")
            st.rerun()
        except (UnicodeDecodeError, csv.Error, RetailError, ValueError) as exc:
            st.error(str(exc))
