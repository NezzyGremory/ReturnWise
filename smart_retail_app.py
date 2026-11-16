import streamlit as st


st.set_page_config(
    page_title="Smart Retail Demo",
    page_icon=":material/storefront:",
    layout="wide",
    initial_sidebar_state="expanded",
)

page_groups = {
    "Operasional": [
        ("app_pages/smart_retail/overview.py", "Ringkasan", ":material/dashboard:", "ringkasan", True),
        ("app_pages/smart_retail/pos.py", "Kasir (POS)", ":material/point_of_sale:", "kasir", False),
        ("app_pages/smart_retail/products.py", "Produk", ":material/inventory_2:", "produk", False),
        ("app_pages/smart_retail/inventory.py", "Inventaris", ":material/warehouse:", "inventaris", False),
        ("app_pages/smart_retail/transactions.py", "Transaksi", ":material/receipt_long:", "transaksi", False),
    ],
    "Relasi toko": [
        ("app_pages/smart_retail/customers.py", "Pelanggan", ":material/group:", "pelanggan", False),
        ("app_pages/smart_retail/suppliers.py", "Supplier", ":material/local_shipping:", "supplier", False),
    ],
    "Analitik": [
        ("app_pages/smart_retail/reports.py", "Laporan", ":material/query_stats:", "laporan", False),
        ("app_pages/smart_retail/intelligence.py", "Retail Intelligence", ":material/insights:", "retail-intelligence", False),
    ],
    "Pengaturan data": [
        ("app_pages/smart_retail/settings.py", "Cadangkan data", ":material/backup:", "cadangkan-data", False),
    ],
}

registered_pages = {
    section: [
        st.Page(path, title=title, icon=icon, url_path=url_path, default=default)
        for path, title, icon, url_path, default in entries
    ]
    for section, entries in page_groups.items()
}

page = st.navigation(registered_pages, position="hidden")
with st.sidebar:
    st.info("Demo publik: gunakan data fiktif. Data demo dipakai bersama, dapat berubah, dan bisa terhapus saat layanan restart.")
    for section, entries in page_groups.items():
        st.subheader(section)
        for path, title, icon, _, _ in entries:
            st.page_link(path, label=title, icon=icon, width="stretch")

page.run()
