import pandas as pd
import streamlit as st

from smart_retail.analytics import rfm_analysis
from smart_retail.db import initialize_database
from smart_retail.intelligence import inventory_recommendations, run_sales_forecast
from smart_retail.services import RetailError, list_categories, list_products
from smart_retail.ui import empty_state, page_heading, table


initialize_database()
page_heading("Retail Intelligence", "Prediksi baseline dan rekomendasi stok hanya dihitung dari riwayat toko yang tersimpan.")

st.subheader("Sales forecasting dan demand prediction")
st.write("Baseline moving average dipakai sebagai pembanding sederhana. Prediksi tidak dijalankan otomatis dan tidak memakai model ReturnWise.")
products = list_products(active_only=True)
categories = list_categories()
target_options = {"Total toko": ("store", None)}
target_options.update({f"Produk · {item['name']} · {item['sku']}": ("product", item["id"]) for item in products})
target_options.update({f"Kategori · {item['name']}": ("category", item["id"]) for item in categories})
forecast_a, forecast_b, forecast_c = st.columns([1, 0.75, 1.2], vertical_alignment="bottom")
with forecast_a:
    forecast_target = st.selectbox("Cakupan prediksi", list(target_options))
with forecast_b:
    horizon = st.selectbox("Horizon", [7, 30], format_func=lambda days: f"{days} hari")
with forecast_c:
    run_clicked = st.button("Hitung baseline dan evaluasi", type="primary", width="stretch")

if run_clicked:
    try:
        target_type, target_id = target_options[forecast_target]
        result = run_sales_forecast(
            horizon,
            product_id=target_id if target_type == "product" else None,
            category_id=target_id if target_type == "category" else None,
        )
        st.session_state["retail_forecast_result"] = result
        st.session_state["retail_forecast_target"] = forecast_target
        st.session_state["retail_forecast_horizon"] = horizon
    except (RetailError, ValueError) as exc:
        st.error(str(exc))

forecast = st.session_state.get("retail_forecast_result")
if forecast:
    if not forecast.get("ready"):
        st.warning(forecast["reason"], title="Belum cukup data untuk menghasilkan prediksi yang dapat diandalkan")
        st.write(f"Riwayat kalender: {forecast.get('history_days', 0)} hari. Hari dengan penjualan tercatat: {forecast.get('active_days', 0)}.")
    else:
        st.write(f"Cakupan: **{st.session_state.get('retail_forecast_target', 'Total toko')}** · Horizon: **{st.session_state.get('retail_forecast_horizon', 7)} hari** · Versi: **{forecast['model_version']}**")
        st.write(f"Dihitung {forecast['created_at']}. Metode: {forecast['method']}")
        actual = forecast["actual"].copy()
        actual = actual.rename(columns={"tanggal": "tanggal", "penjualan_aktual": "aktual"})
        predicted = pd.DataFrame(forecast["forecast"]).rename(columns={"tanggal": "tanggal", "penjualan_prediksi": "prediksi"})
        actual["tanggal"] = pd.to_datetime(actual["tanggal"])
        predicted["tanggal"] = pd.to_datetime(predicted["tanggal"])
        combined = pd.merge(actual[["tanggal", "aktual"]], predicted[["tanggal", "prediksi"]], on="tanggal", how="outer")
        combined["tanggal"] = pd.to_datetime(combined["tanggal"])
        combined = combined.set_index("tanggal").sort_index()
        st.line_chart(combined, alt="Penjualan aktual tersimpan dan prediksi baseline untuk tanggal mendatang")
        metric_col_a, metric_col_b = st.columns(2, gap="large")
        for col, metric_name in ((metric_col_a, "Penjualan bersih (Rp)"), (metric_col_b, "Unit terjual")):
            with col:
                metric_values = forecast["metrics"]["revenue" if metric_name.startswith("Penjualan") else "unit"]
                st.write(f"**Validasi time-based: {metric_name}**")
                metric_frame = pd.DataFrame([{
                    "MAE": metric_values["MAE"], "RMSE": metric_values["RMSE"],
                    "WAPE": f"{metric_values['WAPE']:.1%}" if metric_values["WAPE"] is not None else "Tidak tersedia",
                }])
                table(metric_frame, f"MAE, RMSE, dan WAPE validasi untuk {metric_name}")
        forecast_table = pd.DataFrame(forecast["forecast"]).rename(columns={
            "tanggal": "Tanggal", "unit_prediksi": "Prediksi unit", "penjualan_prediksi": "Prediksi penjualan (Rp)",
        })
        table(forecast_table, "Nilai forecast baseline per hari")
        st.write(forecast["limitations"])
else:
    st.info("Pilih cakupan prediksi lalu hitung baseline. Sistem akan menolak prediksi jika riwayat belum memenuhi batas minimum.", title="Forecast belum dijalankan")

st.divider()
st.subheader("Smart inventory recommendation")
recommendations = inventory_recommendations()
if recommendations:
    frame = pd.DataFrame(recommendations)[[
        "sku", "name", "stock", "minimum_stock", "units_30d", "daily_demand", "lead_time_days",
        "reorder_point", "suggested_quantity", "status", "basis",
    ]].rename(columns={
        "sku": "SKU", "name": "Produk", "stock": "Stok", "minimum_stock": "Minimum",
        "units_30d": "Terjual 30 hari", "daily_demand": "Rata-rata unit/hari",
        "lead_time_days": "Lead time (hari)", "reorder_point": "Titik pesan",
        "suggested_quantity": "Saran jumlah restock", "status": "Status", "basis": "Dasar hitung",
    })
    frame["Titik pesan"] = frame["Titik pesan"].fillna("Belum tersedia")
    frame["Saran jumlah restock"] = frame["Saran jumlah restock"].fillna("Belum tersedia")
    table(frame, "Rekomendasi inventaris dengan status, titik pesan, dan dasar hitung")
    st.write("Rata-rata permintaan memakai penjualan 30 hari tercatat. Safety stock menggunakan akar rata-rata harian dikalikan akar lead time. Target stok mencakup lead time dan tujuh hari berikutnya. Saran kosong jika riwayat penjualan produk kurang dari 14 hari.")
else:
    empty_state("Tambahkan produk aktif untuk melihat posisi inventaris.")

st.divider()
st.subheader("Segmentasi pelanggan RFM")
st.write("Recency dan frequency memakai transaksi selesai. Monetary adalah total pembelian bersih setelah diskon, sebelum pajak.")
rfm = rfm_analysis()
if rfm.empty:
    empty_state("RFM akan tersedia setelah ada pelanggan dengan transaksi selesai.")
else:
    counts = rfm.groupby("segmen", as_index=False).agg(jumlah_pelanggan=("id", "count"), nilai_beli_median=("nilai_bersih", "median"))
    table(counts.rename(columns={"segmen": "Segmen", "jumlah_pelanggan": "Pelanggan", "nilai_beli_median": "Median pembelian bersih (Rp)"}),
          "Jumlah pelanggan dan median nilai pembelian per segmen RFM")
    st.bar_chart(counts, x="segmen", y="jumlah_pelanggan", x_label="Segmen", y_label="Pelanggan",
                 alt="Jumlah pelanggan di setiap segmen RFM")
    visible_rfm = rfm[["pelanggan", "hari_sejak_transaksi", "frekuensi", "nilai_bersih", "segmen"]].rename(columns={
        "pelanggan": "Pelanggan", "hari_sejak_transaksi": "Hari sejak pembelian", "frekuensi": "Frekuensi transaksi",
        "nilai_bersih": "Pembelian bersih (Rp)", "segmen": "Segmen",
    })
    table(visible_rfm, "Recency, frequency, monetary, dan segmen setiap pelanggan")
    st.write("Aturan segmen memakai umur pembelian terakhir dan frekuensi tercatat. Segmen bukan sifat permanen pelanggan dan berubah saat transaksi baru masuk. Hanya pelanggan dengan nama tersimpan dan transaksi selesai yang dihitung.")
