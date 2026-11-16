import pandas as pd
import streamlit as st

from smart_retail.db import initialize_database
from smart_retail.services import RetailError, list_products, list_receipts, list_suppliers, save_supplier
from smart_retail.ui import empty_state, page_heading, table, timestamp


initialize_database()
page_heading("Supplier", "Simpan kontak pemasok, waktu pasok, dan riwayat penerimaan barang.")

suppliers = list_suppliers()
mode = st.segmented_control("Tindakan supplier", ["Tambah supplier", "Edit supplier"], default="Tambah supplier")
selected = None
if mode == "Edit supplier" and suppliers:
    supplier_id = st.selectbox(
        "Pilih supplier", [item["id"] for item in suppliers],
        format_func=lambda sid: next(f"{item['name']} · {'Aktif' if item['is_active'] else 'Nonaktif'}" for item in suppliers if item["id"] == sid),
    )
    selected = next(item for item in suppliers if item["id"] == supplier_id)
elif mode == "Edit supplier":
    st.info("Belum ada supplier untuk diedit.")

with st.form("supplier_form"):
    first, second = st.columns(2)
    with first:
        name = st.text_input("Nama supplier", value=selected["name"] if selected else "")
        contact = st.text_input("Nama kontak / telepon", value=selected["contact"] if selected else "")
        email = st.text_input("Email", value=selected["email"] if selected else "")
    with second:
        address = st.text_area("Alamat", value=selected["address"] if selected else "", height=100)
        lead = st.number_input("Lead time (hari)", min_value=0, step=1, value=int(selected["lead_time_days"]) if selected else 7)
        active = st.checkbox("Supplier aktif", value=bool(selected["is_active"]) if selected else True)
    submit = st.form_submit_button("Simpan supplier", type="primary", disabled=(mode == "Edit supplier" and selected is None))
if submit:
    try:
        save_supplier({"name": name, "contact": contact, "email": email, "address": address,
                       "lead_time_days": lead, "is_active": active}, selected["id"] if selected else None)
        st.session_state["supplier_notice"] = f"Supplier {name.strip()} tersimpan."
        st.rerun()
    except RetailError as exc:
        st.error(str(exc))

notice = st.session_state.pop("supplier_notice", None)
if notice:
    st.success(notice)

st.divider()
st.subheader("Daftar supplier")
if suppliers:
    visible = pd.DataFrame(suppliers)[["name", "contact", "email", "lead_time_days", "product_count", "units_received", "is_active"]].rename(columns={
        "name": "Supplier", "contact": "Kontak", "email": "Email", "lead_time_days": "Lead time (hari)",
        "product_count": "Produk terkait", "units_received": "Unit diterima", "is_active": "Status",
    })
    visible["Status"] = visible["Status"].map({1: "Aktif", 0: "Nonaktif"})
    table(visible, "Daftar supplier, lead time, dan jumlah penerimaan stok")
    supplier_options = {f"{item['name']} · #{item['id']}": item for item in suppliers}
    chosen_name = st.selectbox("Lihat produk dan penerimaan", list(supplier_options))
    chosen = supplier_options[chosen_name]
    products = [item for item in list_products(active_only=False) if item["supplier_id"] == chosen["id"]]
    st.subheader(f"Produk dari {chosen['name']}")
    if products:
        frame = pd.DataFrame(products)[["sku", "name", "stock", "minimum_stock", "unit", "selling_price"]].rename(columns={
            "sku": "SKU", "name": "Produk", "stock": "Stok", "minimum_stock": "Minimum",
            "unit": "Satuan", "selling_price": "Harga jual (Rp)",
        })
        table(frame, f"Produk yang dikaitkan dengan supplier {chosen['name']}")
    else:
        empty_state("Belum ada produk yang dikaitkan dengan supplier ini.")
    receipts = list_receipts(supplier_id=chosen["id"])
    st.subheader("Riwayat stok masuk")
    if receipts:
        frame = pd.DataFrame(receipts)[["receipt_no", "created_at", "product_name", "quantity", "unit_cost", "reason"]].rename(columns={
            "receipt_no": "Nomor penerimaan", "created_at": "Waktu", "product_name": "Produk",
            "quantity": "Jumlah", "unit_cost": "Harga modal (Rp)", "reason": "Alasan",
        })
        frame["Waktu"] = frame["Waktu"].map(timestamp)
        table(frame, f"Penerimaan stok dari supplier {chosen['name']}")
    else:
        empty_state("Belum ada penerimaan stok dari supplier ini.")
else:
    empty_state("Tambahkan supplier untuk menghubungkannya dengan produk dan penerimaan stok.")
