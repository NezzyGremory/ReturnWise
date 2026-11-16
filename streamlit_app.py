import streamlit as st

st.set_page_config(
    page_title="Smart Retail",
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
    "ReturnWise Research": [
        ("app_pages/01_overview.py", "Overview", ":material/dashboard:", "returnwise-overview", False),
        ("app_pages/02_data.py", "Dataset and quality", ":material/database:", None, False),
        ("app_pages/03_analysis.py", "EDA and statistics", ":material/query_stats:", None, False),
    ],
    "Decision support": [
        ("app_pages/04_risk_queue.py", "Risk queue", ":material/low_priority:", None, False),
        ("app_pages/05_decision_simulator.py", "Decision simulator", ":material/tune:", None, False),
        ("app_pages/06_research.py", "Models and research", ":material/science:", None, False),
    ],
}

registered_pages = {}
for section, entries in page_groups.items():
    registered_pages[section] = [
        st.Page(path, title=title, icon=icon, url_path=url_path, default=default)
        for path, title, icon, url_path, default in entries
    ]

page = st.navigation(registered_pages, position="hidden")
with st.sidebar:
    for section, entries in page_groups.items():
        st.subheader(section)
        for path, title, icon, _, _ in entries:
            st.page_link(path, label=title, icon=icon, width="stretch")

page.run()
