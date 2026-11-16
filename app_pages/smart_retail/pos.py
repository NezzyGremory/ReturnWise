import streamlit as st

from smart_retail.db import initialize_database
from smart_retail.services import (
    PAYMENT_METHODS, RetailError, create_transaction, get_setting, list_categories,
    list_customers, list_products, transaction_detail,
)
from smart_retail.ui import page_heading, receipt_html, rupiah


initialize_database()
page_heading("Kasir", "Cari produk, periksa keranjang, lalu simpan pembayaran dan perubahan stok sebagai satu transaksi.")

notice = st.session_state.pop("pos_notice", None)
if notice:
    st.success(notice)

if "retail_cart" not in st.session_state:
    st.session_state["retail_cart"] = {}
if "retail_checkout_key" not in st.session_state:
    import uuid
    st.session_state["retail_checkout_key"] = str(uuid.uuid4())
if "retail_cart_version" not in st.session_state:
    import uuid
    st.session_state["retail_cart_version"] = str(uuid.uuid4())
cart: dict[int, int] = st.session_state["retail_cart"]

search_col, category_col = st.columns([1.3, 0.8], vertical_alignment="bottom")
with search_col:
    search = st.text_input("Cari produk", placeholder="Nama, SKU, atau barcode", key="pos_search")
with category_col:
    categories = list_categories()
    category_options = {"Semua kategori": None, **{item["name"]: item["id"] for item in categories}}
    category_name = st.selectbox("Kategori", list(category_options), key="pos_category")

products = list_products(search=search, category_id=category_options[category_name])
if not products:
    st.info("Produk aktif belum tersedia. Tambahkan produk sebelum memulai transaksi.", title="Kasir belum siap")
    st.page_link("app_pages/smart_retail/products.py", label="Buka manajemen produk", icon=":material/inventory_2:")
else:
    with st.form("pos_add_item"):
        product_by_id = {item["id"]: item for item in products}
        selected_product = st.selectbox(
            "Pilih produk",
            list(product_by_id),
            format_func=lambda product_id: (
                f"{product_by_id[product_id]['name']} · {product_by_id[product_id]['sku']} · "
                f"{rupiah(product_by_id[product_id]['selling_price'])} · stok {product_by_id[product_id]['stock']}"
            ),
        )
        qty_add = st.number_input("Jumlah", min_value=1, step=1, value=1)
        add_item = st.form_submit_button("Tambah ke keranjang", type="primary", width="stretch")
    if add_item:
        item = product_by_id[selected_product]
        next_qty = cart.get(selected_product, 0) + int(qty_add)
        if next_qty > item["stock"]:
            st.error(f"Stok {item['name']} hanya {item['stock']} {item['unit']}.")
        else:
            cart[selected_product] = next_qty
            st.session_state["retail_cart"] = cart
            import uuid
            st.session_state["retail_cart_version"] = str(uuid.uuid4())
            st.session_state["pos_notice"] = f"{item['name']} ditambahkan ke keranjang."
            st.rerun()

st.divider()
st.subheader("Keranjang")
cart_products = {item["id"]: item for item in list_products(active_only=True)}
subtotal = 0
stock_valid = True
if not cart:
    st.info("Keranjang masih kosong. Cari produk di atas untuk memulai.", title="Belum ada item")
else:
    for product_id in list(cart):
        item = cart_products.get(product_id)
        if item is None:
            cart.pop(product_id, None)
            continue
        row_a, row_b, row_c, row_d = st.columns([2.1, 1, 1.1, 0.65], vertical_alignment="bottom")
        with row_a:
            st.write(f"**{item['name']}**")
            st.write(f"{item['sku']} · {rupiah(item['selling_price'])} · stok {item['stock']} {item['unit']}")
        with row_b:
            quantity = st.number_input(
                f"Kuantitas {item['name']}", min_value=1, max_value=max(1, item["stock"], cart[product_id]),
                value=cart[product_id], step=1, key=f"cart_quantity_{st.session_state['retail_cart_version']}_{product_id}",
            )
            cart[product_id] = int(quantity)
        with row_c:
            line_total = item["selling_price"] * cart[product_id]
            subtotal += line_total
            st.write(f"**{rupiah(line_total)}**")
        with row_d:
            if st.button("Hapus", key=f"remove_{product_id}", help=f"Hapus {item['name']} dari keranjang"):
                cart.pop(product_id, None)
                st.session_state["retail_cart"] = cart
                import uuid
                st.session_state["retail_cart_version"] = str(uuid.uuid4())
                st.rerun()
        if cart[product_id] > item["stock"]:
            stock_valid = False
            st.warning(f"Kuantitas {item['name']} melebihi stok tersedia.")
    st.session_state["retail_cart"] = cart

discount = st.number_input(
    "Diskon transaksi (Rp)", min_value=0, step=1000, value=0,
    key=f"pos_discount_{st.session_state['retail_checkout_key']}",
)
try:
    tax_rate = float(get_setting("tax_rate", "0") or 0)
except ValueError:
    tax_rate = 0.0
taxable = max(0, subtotal - int(discount))
tax_amount = round(taxable * tax_rate / 100)
total = taxable + tax_amount

summary_a, summary_b, summary_c = st.columns(3)
summary_a.metric("Subtotal", rupiah(subtotal))
summary_b.metric(f"Pajak ({tax_rate:g}%)", rupiah(tax_amount))
summary_c.metric("Total pembayaran", rupiah(total))

if subtotal and int(discount) > subtotal:
    st.error("Diskon tidak boleh melebihi subtotal.")

customers = list_customers()
customer_options = {"Pelanggan umum": None, **{f"{item['name']} · {item['phone'] or 'tanpa nomor'}": item["id"] for item in customers}}
with st.form("pos_checkout"):
    payment_method = st.selectbox("Metode pembayaran", PAYMENT_METHODS)
    customer_name = st.selectbox("Pelanggan", list(customer_options))
    amount_paid = st.number_input(
        "Jumlah dibayar (Rp)", min_value=0, step=1000, value=total,
        key=f"pos_amount_{st.session_state['retail_checkout_key']}_{total}", disabled=payment_method != "Tunai",
    )
    st.write(f"Kembalian tunai: {rupiah(max(0, int(amount_paid) - total))}. Pembayaran QRIS dan kartu hanya dicatat sebagai metode, tanpa koneksi gateway.")
    confirmed = st.checkbox(
        "Saya memastikan item dan jumlah pembayaran sudah benar.",
        key=f"pos_confirm_{st.session_state['retail_checkout_key']}",
    )
    checkout = st.form_submit_button(
        "Simpan transaksi",
        type="primary",
        width="stretch",
        disabled=not cart or not stock_valid or (subtotal > 0 and int(discount) > subtotal),
    )

if checkout:
    if not confirmed:
        st.warning("Centang konfirmasi setelah memeriksa keranjang.")
    else:
        try:
            result = create_transaction(
                [{"product_id": product_id, "quantity": quantity} for product_id, quantity in cart.items()],
                payment_method=payment_method,
                amount_paid=int(amount_paid) if payment_method == "Tunai" else total,
                discount_amount=int(discount),
                tax_rate=tax_rate,
                customer_id=customer_options[customer_name],
                idempotency_key=st.session_state["retail_checkout_key"],
            )
            st.session_state["last_retail_transaction_id"] = result["id"]
            st.session_state["retail_cart"] = {}
            import uuid
            st.session_state["retail_checkout_key"] = str(uuid.uuid4())
            st.session_state["retail_cart_version"] = str(uuid.uuid4())
            st.session_state["pos_notice"] = f"Transaksi {result['transaction_no']} tersimpan sebesar {rupiah(result['total'])}."
            st.rerun()
        except RetailError as exc:
            st.error(str(exc))

last_id = st.session_state.get("last_retail_transaction_id")
if last_id:
    detail = transaction_detail(last_id)
    if detail:
        st.divider()
        st.subheader("Struk terakhir")
        st.write(f"Nomor transaksi: **{detail['transaction']['transaction_no']}**")
        st.dataframe(
            [{"Produk": item["product_name_snapshot"], "Kuantitas": item["quantity"],
              "Harga satuan (Rp)": item["unit_price"], "Jumlah (Rp)": item["line_total"]}
             for item in detail["items"]],
            hide_index=True, alt="Rincian item pada struk terakhir", width="stretch",
        )
        st.download_button(
            "Unduh struk untuk dicetak",
            data=receipt_html(detail),
            file_name=f"struk-{detail['transaction']['transaction_no']}.html",
            mime="text/html",
        )
