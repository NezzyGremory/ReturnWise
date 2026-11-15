# Pengembangan Sistem Pendukung Keputusan Berbasis Machine Learning untuk Prioritisasi Risiko Pengembalian Produk pada E-Commerce

## Halaman Awal

**Nama penulis:** [Belum ditemukan dalam repository]  
**Institusi:** [Belum tersedia]  
**Tahun:** 2026  
**Bidang:** Machine learning, analitik e-commerce, dan sistem pendukung keputusan

### Abstrak

Penelitian ini mengaudit ReturnWise, sistem pendukung keputusan yang mengurutkan transaksi berdasarkan estimasi probabilitas pengembalian. Dataset kompetisi berisi 200.000 transaksi berlabel dan 50.000 transaksi uji tanpa label. Data lengkap tanpa baris duplikat, tetapi memiliki anomali nilai numerik dan tidak menyediakan timestamp maupun aturan snapshot fitur. Karena itu, fitur disebut kandidat waktu-pemesanan yang diperluas, bukan fitur pra-pengiriman yang telah terbukti tersedia pada waktu prediksi. Pada validasi silang terstratifikasi tiga fold atas 160.000 baris pengembangan, LightGBM memperoleh ROC-AUC 0,586576, HistGradientBoosting 0,586347, dan Logistic Regression 0,585437. Blend out-of-fold Logistic Regression dan HistGradientBoosting memperoleh ROC-AUC 0,587648, tetapi belum didukung validasi berulang atau holdout independen. Skor holdout historis 0,589481 berasal dari soft-voting lama dan tidak mewakili pipeline terkoreksi. Refit LightGBM terkoreksi dihentikan oleh pemeriksaan sumber daya sebelum pelatihan. Aplikasi telah menyediakan alur analisis lokal, antrean risiko, simulasi kapasitas, dan frontend statis untuk memeringkat probabilitas yang diunggah. Hasil ini mendukung demonstrasi prioritisasi, tetapi belum membuktikan kinerja temporal, dampak operasional, pengurangan retur, atau peningkatan profit.

**Kata kunci:** pengembalian produk, e-commerce, klasifikasi biner, kalibrasi probabilitas, ROC-AUC, prioritisasi risiko, kapasitas pemeriksaan, sistem pendukung keputusan

---

# BAB I — PENDAHULUAN

## 1.1 Latar Belakang

E-commerce mempertemukan konsumen dan produk tanpa selalu memungkinkan konsumen memeriksa produk secara langsung sebelum membeli. Pengembalian produk menjadi bagian dari proses perdagangan dan melibatkan penerimaan, pemeriksaan, pengelompokan, pemulihan nilai, atau pengiriman kembali barang. Tinjauan Karl (2025) menunjukkan bahwa peramalan pengembalian konsumen dikaji dari perspektif sistem informasi, operasi, dan pemasaran dengan tujuan, sumber data, unit analisis, serta metode yang beragam.

Prediksi seluruh transaksi secara manual tidak praktis ketika jumlah pesanan besar. Namun, keluaran model berupa label “retur” atau “tidak retur” saja tidak menjawab transaksi mana yang perlu diperiksa saat kapasitas staf terbatas. Sistem pendukung keputusan perlu menghasilkan skor yang dapat diurutkan, menampilkan batas kapasitas, dan mempertahankan konteks ketidakpastian serta asumsi biaya.

Machine learning terawasi dapat memetakan fitur transaksi ke peluang kelas target. Dalam penelitian ini, target kompetisi returned menandai apakah suatu transaksi dikembalikan. ROC-AUC membandingkan kualitas ranking; Brier score dan log loss membantu menilai probabilitas yang digunakan untuk menghitung kasus harapan. Kalibrasi menjadi relevan ketika besaran probabilitas ditafsirkan, tetapi tidak memperbaiki diskriminasi secara otomatis dan tidak membuktikan bahwa fitur tersedia pada waktu keputusan.

ReturnWise menghubungkan estimasi risiko dengan antrean pemeriksaan berbatas kapasitas. Aplikasi lokal memuat dataset, menampilkan statistik dan risiko kualitas data, menghasilkan antrean menggunakan artifact model historis, dan mensimulasikan kapasitas serta biaya. Frontend statis terpisah memeriksa CSV di browser dan mengurutkan skor probabilitas yang telah disediakan. Tujuan penelitian adalah mendokumentasikan eksperimen aktual, membedakan protokol evaluasi, dan membatasi kesimpulan pada bukti tersimpan.

## 1.2 Identifikasi Masalah

1. Transaksi berisiko perlu diurutkan untuk pemeriksaan dengan kapasitas terbatas.
2. Selisih performa model yang diuji kecil dan belum cukup untuk menyatakan pemenang stabil.
3. Kalibrasi hanya tercatat untuk model historis, bukan kandidat terkoreksi.
4. Beberapa fitur numerik memiliki nilai di luar rentang konvensional tanpa definisi sumber.
5. Dataset tidak berisi timestamp atau aturan snapshot untuk membuktikan ketersediaan fitur pada waktu prediksi.
6. Biaya dan efektivitas review pada simulasi adalah asumsi, bukan pengukuran bisnis.
7. Artifact deployment yang ada merupakan model lama; LightGBM terkoreksi belum di-refit dan dievaluasi pada holdout baru.

## 1.3 Rumusan Masalah

1. Bagaimana struktur, distribusi target, dan kualitas dataset ReturnWise?
2. Bagaimana preprocessing dan feature engineering memengaruhi kesiapan fitur dan hasil CV?
3. Bagaimana performa kandidat model pada validasi silang tiga fold?
4. Apa yang dapat disimpulkan dari evaluasi probabilitas historis dan mengapa hasil itu tidak berlaku bagi pipeline terkoreksi?
5. Bagaimana probabilitas membentuk antrean prioritas dengan kapasitas terbatas?
6. Bagaimana kemampuan dan batasan ReturnWise sebagai DSS?

## 1.4 Tujuan Penelitian

Tujuan penelitian adalah mengaudit dataset dan repository; mendokumentasikan preprocessing, fitur, model, dan protokol; membandingkan hasil yang sebanding; menjelaskan kalibrasi historis; mendokumentasikan ranking dan simulasi kapasitas; serta menjelaskan implementasi aplikasi dan batas klaimnya.

## 1.5 Manfaat Penelitian

**Akademik:** catatan yang menelusuri pertanyaan penelitian ke dataset, konfigurasi, output notebook, laporan, dan keterbatasan.  
**Metodologis:** pemisahan CV, OOF, holdout, kalibrasi, simulasi, dan status deployment.  
**Operasional:** demonstrasi cara menyusun antrean sesuai kapasitas; manfaat operasional masih hipotesis sampai diuji langsung.  
**Pengembangan sistem:** contoh integrasi profiling, estimasi risiko, ranking, dan simulasi kapasitas dalam prototipe.

## 1.6 Batasan Penelitian

- Data berasal dari berkas kompetisi, bukan transaksi operasional ReturnWise.
- Target returned hanya tersedia pada 200.000 baris train; 50.000 baris test tidak berlabel.
- Corrected raw-feature CV mencakup Logistic Regression, HistGradientBoosting, dan LightGBM.
- Metrik utama kompetisi adalah ROC-AUC. Metrik lain hanya dilaporkan untuk protokol yang menyimpannya.
- Split bersifat acak-terstratifikasi; validasi temporal tidak tersedia.
- Timestamp, snapshot fitur, dan cutoff agregat histori tidak tersedia.
- Anomali tidak dikoreksi, di-clip, atau dihapus berdasarkan asumsi yang belum diverifikasi.
- Refit model terkoreksi, kalibrasi kandidat terkoreksi, prediksi baru test, dan evaluasi deployment belum selesai.
- Tidak ada uji kausal untuk menyimpulkan retur dicegah, biaya turun, atau profit meningkat.

## 1.7 Kontribusi Penelitian

| Kontribusi | Status | Batas |
|---|---|---|
| Audit struktur dan kualitas data | Aktual | Berlaku pada berkas lokal |
| Audit risiko waktu fitur | Aktual | Waktu ketersediaan belum terverifikasi |
| CV tiga kandidat raw-feature | Aktual | Satu seed, tiga fold, tanpa interval kepercayaan |
| Interaksi dan seleksi L1 | Aktual | Efek kecil/campuran; tidak menjadi default |
| Blend OOF | Aktual | Belum dikonfirmasi dengan repeated CV/holdout |
| Kalibrasi dan holdout historis | Aktual, historis | Bukan evaluasi pipeline terkoreksi |
| Ranking dan simulasi kapasitas | Implementasi/eksperimen | Biaya dan review adalah asumsi |
| Website ReturnWise | Prototipe | Bukan bukti performa atau dampak |
| Novelty pendekatan | Belum dibuktikan | Perlu tinjauan sistematis dan pembanding sepadan |

---

# BAB II — TINJAUAN PUSTAKA

## 2.1 Pengembalian Produk pada E-Commerce

Pengembalian konsumen menghasilkan aliran balik barang dan pekerjaan reverse logistics. Prediksi dapat berfokus pada keputusan retur per transaksi, volume retur per produk dan periode, atau tingkat retur per item. Unit analisis dan waktu prediksi menentukan fitur yang layak digunakan. Karl (2025) menelaah 25 publikasi dan memetakan tujuan, sumber data, prediktor, serta metode; kajiannya juga menekankan perlunya evaluasi operasional/moneter selain metrik teknis.

Cui, Rajagopalan, dan Ward (2020) memperkirakan volume retur agregat untuk perusahaan aksesori otomotif; Zhu et al. (2018) mengembangkan pendekatan graph lokal untuk propensity retur; Dzyabura et al. (2023) memakai citra dan fitur pra-peluncuran untuk memprediksi tingkat retur item pada keputusan assortment. Unit analisis berbeda sehingga hasilnya bukan angka pembanding langsung bagi klasifikasi per transaksi ReturnWise.

## 2.2 Machine Learning

Penelitian ini menerapkan supervised learning: model mempelajari hubungan fitur X dan label biner Y. Keluaran dapat berupa kelas atau skor/probabilitas untuk kelas positif. ReturnWise memakai peluang returned=1 sebagai dasar ranking, bukan sebagai kepastian hasil.

## 2.3 Algoritma yang Diuji

| Algoritma | Konsep/peran | Bukti di ReturnWise | Keterbatasan |
|---|---|---|---|
| Dummy prior | Baseline prevalensi kelas | Benchmark baseline historis | Tidak memakai pola fitur |
| Logistic Regression | Model linear log-odds; baseline ringkas | Historis dan corrected raw CV; class_weight seimbang, max_iter 2000, fitur numerik diskalakan | Bentuk hubungan linear |
| Random Forest | Agregasi pohon bootstrap untuk pola nonlinier | Benchmark historis quick, 32 pohon | Bukan corrected finalist |
| Extra Trees | Ensemble pohon dengan split lebih acak | Benchmark historis quick, 32 pohon | Skor historis lebih rendah pada protokol tercatat |
| HistGradientBoosting | Boosting pohon histogram | Historis dan corrected raw CV; 48 iterasi, learning_rate 0,08 | Kategori dikodekan ordinal pada hasil terkoreksi |
| LightGBM | Gradient-boosted trees tabular | Corrected CV dan pilot Optuna; konfigurasi quick 48 estimator, learning_rate 0,05, 31 leaves, class_weight seimbang, satu thread, seed 42 | Belum refit/deployment atau holdout baru |
| Soft voting/mean blend | Merata-ratakan probabilitas beberapa model | Soft vote historis HGB+LR; blend OOF corrected | Gain OOF kecil; eksperimen historis dan terkoreksi berbeda |

CatBoost, XGBoost, SVM, decision tree, dan TabPFN muncul dalam dokumentasi/registry, tetapi keberadaan paket atau rencana bukan bukti eksperimen selesai. Tidak ada perbandingan akhir yang melaporkan model-model itu sebagai hasil teruji.

## 2.4 Data Preprocessing

Validasi memeriksa skema, target, jumlah baris, identifier, integritas berkas, dan konsistensi train/test. Raw-feature pipeline menggunakan imputasi median numerik dan modus kategorikal, lalu one-hot encoding; parameter dipelajari di dalam fold training. Logistic Regression menggunakan scaling numerik. HistGradientBoosting memakai ordinal encoding. Anomali domain hanya ditandai; tidak disamakan dengan outlier statistik dan tidak dihapus.

## 2.5 Feature Engineering dan Selection

Tiga interaksi diuji: product_price × discount_percent; num_product_views ÷ session_length_minutes; dan product_price × product_rating. Rasio tak hingga diubah menjadi missing untuk imputer. Selector L1 memakai SelectFromModel dengan Logistic Regression (C=0,1; saga; class_weight seimbang; threshold median), dipasang di training fold saja. Efek feature selection memburuk sehingga selector tidak dipakai sebagai default.

## 2.6 Evaluasi Model

Untuk probabilitas p_i dan label y_i:

- **ROC-AUC:** luas kurva TPR terhadap FPR; mengukur ranking relatif. 0,5 adalah acuan ranking acak, bukan akurasi kelas.
- **Precision:** TP/(TP+FP), ketepatan prediksi positif pada threshold tertentu.
- **Recall:** TP/(TP+FN), bagian kasus positif yang ditemukan.
- **F1:** 2 × precision × recall/(precision+recall).
- **Brier score:** (1/n) Σ(p_i−y_i)²; lebih kecil lebih baik.
- **Log loss:** −(1/n) Σ[y_i log(p_i)+(1−y_i)log(1−p_i)]; menghukum probabilitas yakin yang salah.
- **PR-AUC/average precision:** ringkasan precision-recall; dibaca bersama prevalence.

Cross-validation membagi data pengembangan secara terstratifikasi. OOF menyimpan prediksi validasi untuk setiap baris yang tidak ikut melatih model tersebut. Rata-rata metrik antar-fold dan metrik gabungan OOF merupakan ringkasan berbeda. Tidak ada repeated CV, bootstrap interval, atau paired confidence interval pada hasil terkini.

## 2.7 Probability Calibration

Kalibrasi menilai kesesuaian probabilitas prediksi dengan frekuensi kejadian pada kelompok skor serupa. Reliability diagram membandingkan mean prediction dengan observed frequency per bin. Brier dan log loss menilai kualitas probabilitas keseluruhan namun tidak menggantikan validasi eksternal. Notebook lama membandingkan uncalibrated, sigmoid, isotonic, dan beta; beta dipilih untuk model historis menggunakan CV training-side lalu dipasang pada partisi kalibrasi terpisah. Kull, Silva Filho, dan Flach (2017) mengembangkan beta calibration. Kandidat pipeline terkoreksi belum dikalibrasi.

## 2.8 Sistem Pendukung Keputusan

DSS menyajikan data, analisis, dan alternatif kepada pengambil keputusan. Dalam ReturnWise, komponen prediksi memberi skor risiko, sedangkan ranking dan simulator membantu memilih transaksi sesuai kapasitas. Keputusan akhir tetap pada pengguna. Tidak ada bukti bahwa rekomendasi membatalkan order atau mencegah retur.

## 2.9 Risk Ranking dan Kapasitas

Untuk N transaksi dan kapasitas k, ranking memilih k probabilitas tertinggi. Ini antrean relatif, bukan kepastian hasil. Implementasi cost-sensitive menggunakan skor p·C_FN − [C_review+(1−p)·C_FP]. Turunannya terhadap p adalah C_FN+C_FP, sehingga pada biaya tetap nonnegatif ranking tersebut monotonic dan urutannya sama dengan probabilitas. Jadi, mengganti nama skor biaya tetap tidak otomatis memperbaiki ranking.

Biaya false positive, false negative, dan review adalah input skenario. Simulasi mengasumsikan review dapat mengidentifikasi retur pada kasus terpilih; asumsi ini belum diuji dengan intervensi nyata.

## 2.10 Penelitian Terdahulu

| Penulis/tahun | Fokus/metode/data | Temuan yang diverifikasi | Batas perbandingan |
|---|---|---|---|
| Karl (2025) | Tinjauan sistematis 25 publikasi forecasting retur e-commerce | Memetakan tujuan, sumber data, prediktor, metode, dan agenda penelitian | Bukan benchmark model ReturnWise |
| Zhu et al. (2018) | Weighted hybrid graph dan random-walk lokal pada beberapa dataset e-commerce | Melaporkan peningkatan terhadap metode pembanding pada dataset artikel | Membutuhkan graph/histori; data dan protokol berbeda |
| Cui et al. (2020) | Prediksi volume agregat retur menurut retailer, kategori, dan periode | LASSO/Elastic Net dengan interaksi memberi prediksi agregat yang kuat | Target bukan retur individual per order |
| Mishra & Dutta (2024) | Logistic Regression, Random Forest, boosting pada data industri e-commerce; ROC/AUC | Menguji prediksi retur dan variabel terkait | Data dan evaluasi tidak langsung sebanding |
| Ma & Wang (2024) | COP-HGNN customer-order-product pada platform fashion dua tahun | Lebih baik pada repeat customers; prediksi customer baru terbatas | Data graph tidak tersedia pada CSV ReturnWise |
| Duong et al. (2025) | Review Amazon, topic modeling, model interpretable pada lima kategori | Mengkaji prediktor intrinsik/ekstrinsik perilaku retur | Target/label teks berbeda dari kolom transaksi kompetisi |
| Dzyabura et al. (2023) | Citra dan fitur pra-peluncuran untuk return rate item serta assortment apparel | Melaporkan ilustrasi peningkatan profit 8,3% pada retailer focal | Bukan bukti dampak ReturnWise; unit item pra-peluncuran |

## 2.11 Research Gap

Literatur telah mencakup volume/tingkat retur agregat, propensity pelanggan/transaksi, graph learning, model interpretable, dan keputusan assortment. Penelitian ini tidak menyatakan prediksi retur atau prioritisasi kapasitas belum pernah diteliti. Tinjauan di sini bukan systematic review dan tidak mencakup seluruh database maupun solusi peserta Kaggle. Posisi ReturnWise yang didukung bukti adalah audit dataset spesifik, evaluasi kandidat raw-feature, audit ketersediaan waktu, dan demonstrasi ranking berbatas kapasitas. Klaim novelty gabungan calibration-aware cost-sensitive prioritization belum dibuktikan dan membutuhkan tinjauan sistematis serta evaluasi empiris yang setara.

---

# BAB III — METODOLOGI PENELITIAN

## 3.1 Jenis dan Pendekatan

Penelitian bersifat kuantitatif-eksperimental dan rekayasa prototipe DSS. Tahap eksperimen memeriksa kualitas data dan membandingkan model pada split terstratifikasi. Tahap rekayasa mendokumentasikan integrasi artifact, antrean risiko, dan simulasi kapasitas. Tidak dilakukan uji intervensi operasional. Pemeriksaan dependensi environment yang tersimpan ditampilkan pada **Gambar 1, “Pemeriksaan Dependensi Environment ReturnWise”**; gambar itu mendokumentasikan pemeriksaan paket, bukan reproduksi seluruh eksperimen.

## 3.2 Objek Penelitian

Objeknya adalah transaksi e-commerce kompetisi Kaggle E-Commerce Returns: Who Will Send It Back? dan ReturnWise, prototipe yang membantu memprioritaskan antrean pemeriksaan. Fokus model adalah returned; fokus sistem adalah ranking, bukan penolakan transaksi otomatis.

## 3.3 Sumber Data

Data berasal dari [kompetisi Retail Return Risk Modeling di Kaggle](https://www.kaggle.com/competitions/retail-return-risk-modeling), disimpan di data/kaggle/. Fingerprint tercatat: 2ae5e87906198d2e3688c12bd807b33194115aea579aacccd070a97732539f56. ZIP lokal tidak lengkap dan tidak diekstrak; CSV tervalidasi digunakan.

| Berkas | Baris | Kolom | Peran |
|---|---:|---:|---|
| train.csv | 200.000 | 16 | Fitur, label, identifier |
| test.csv | 50.000 | 15 | Fitur tanpa label |
| sample_submission.csv | 50.000 | 2 | Template ID dan returned |
| returnwise_submission.csv | 50.000 | 2 | Prediksi export historis lama, bukan corrected refit |

## 3.4 Deskripsi Dataset

| Kolom | Tipe/distinct | Ringkasan train | Makna/peran | Batas |
|---|---|---|---|---|
| customer_age | numerik/78 | 10–87; median 58 | Usia pelanggan | Snapshot tidak tercatat |
| product_price | numerik/199.800 | −183,29; median 24,46; 855,08 | Harga transaksi | Nilai negatif perlu definisi |
| discount_percent | numerik/199.907 | −7,54; median 42,95; 78,79 | Diskon | Sebagian di luar 0–100 |
| product_rating | numerik/199.936 | 0,445–5,453; median 2,819 | Rating produk | Dapat mencakup rating setelah transaksi |
| past_purchase_count | integer/26 | 1–26; median 9 | Hitungan pembelian lalu | Cutoff tidak diketahui |
| past_return_rate | numerik/199.934 | −0,061–0,866; median 0,191 | Tingkat retur historis | Denominator/cutoff tidak diketahui |
| delivery_delay_days | numerik/199.947 | −9,05–7,961; median 0,519 | Keterlambatan aktual | Post-shipment; dikeluarkan |
| session_length_minutes | numerik/199.921 | −11,01–137,65; median 81,56 | Durasi sesi | Batas waktu sesi tidak tercatat |
| num_product_views | integer/50 | −5–44; median 13 | Tampilan produk | Nilai negatif; asal tidak diketahui |
| device_type | kategorikal/3 | mobile, desktop, tablet | Perangkat | Asosiasi bukan sebab |
| product_category | kategorikal/6 | clothing, toys, beauty, sports, home, electronics | Kategori produk | Asosiasi bukan sebab |
| shipping_method | kategorikal/3 | express, standard, same day | Metode pengiriman | Tersedia setelah pilihan checkout |
| payment_method | kategorikal/4 | Apple Pay, credit card, PayPal, debit card | Metode pembayaran | Snapshot tidak tercatat |
| used_coupon | integer biner/2 | 0/1 | Kupon dipakai | Timestamp tidak tersedia |
| returned | integer biner/2 | 0/1 | Target | Tidak menjadi prediktor |
| order_id | integer/200.000 | 0–249.999; median 124.891 | Identifier | Bukan timestamp atau fitur |

## 3.5 Data Understanding

Target terdiri dari 94.919 retur (47,4595%) dan 105.081 tidak retur (52,5405%). Skema train/test sama setelah target dihapus; order_id tidak overlap. ID tidak berurutan menurut posisi baris dan korelasinya terhadap target sekitar −0,002, bukan bukti urutan waktu.

Empat fitur kategorikal memiliki cardinality rendah. Selisih train/test deskriptif maksimum sekitar 0,0065 SMD numerik dan 0,0045 proporsi kategori; ini bukan uji drift. Distribusi target ada pada **Gambar 3**; boxplot/proporsi kategorikal pada **Gambar 4–11**; asosiasi dan korelasi ada pada **Gambar 12–14**.

## 3.6 Data Quality

Tidak ditemukan missing cell, duplikat persis, target di luar 0/1, atau mismatch skema.

| Fitur | Screening | Baris | Persentase |
|---|---|---:|---:|
| product_price | <0 | 27.285 | 13,64% |
| num_product_views | <0 | 21.443 | 10,72% |
| session_length_minutes | <0 | 1.487 | 0,74% |
| discount_percent | di luar 0–100 | 1.686 | 0,84% |
| product_rating | di luar 0–5 umum | 3.831 | 1,92% |
| past_return_rate | di luar 0–1 | 2.220 | 1,11% |

Scan tidak menemukan sentinel ekstrem −999, −99, 999, 9999. Nilai −1, 0, dan 1 pada num_product_views muncul dalam rentang −5 sampai 44, sehingga −1 saja tidak terbukti sentinel. Tidak ada nilai atau baris yang dikoreksi.

Return rate mentah: clothing 53,05%, toys 50,31%, beauty 49,72%, electronics 41,73%; desktop 51,91%, mobile 45,12%; express 53,22%, same day 44,29%; Apple Pay 52,17%, debit card 44,18%. Proporsi tersebut tidak mengendalikan confounding.

## 3.7 Leakage dan Ketersediaan Fitur

returned adalah label. order_id dikeluarkan sebagai identifier. delivery_delay_days dikeluarkan karena membutuhkan hasil pengiriman. past_purchase_count dan past_return_rate dikeluarkan karena cutoff agregat tidak diketahui. Tidak ada timestamp transaksi/snapshot.

Ada inkonsistensi internal mengenai product_rating. Tabel feature availability FINAL_DATA_QUALITY.md menyarankan dikecualikan sampai aturan as-of tersedia. FINAL_RETURNWISE_RESULTS.md dan FINAL_EXPERIMENT_LOG.md memasukkannya ke kandidat 11 fitur dengan caveat snapshot. Karena itu, tabel CV berikut merujuk ke kandidat 11 kolom yang menyertakan rating. Hasil untuk kandidat ketat 10 fitur tanpa product_rating belum tersedia. Keputusan penulis/data owner diperlukan sebelum naskah difinalkan.

Kandidat 11 fitur eksperimen: customer_age, product_price, discount_percent, product_rating, session_length_minutes, num_product_views, device_type, product_category, shipping_method, payment_method, used_coupon. Ini bukan bukti semua fitur tersedia sebelum pengiriman.

## 3.8 Preprocessing

Pipeline memvalidasi skema, memisahkan target, mengeluarkan identifier/post-event/agregat yang tidak terverifikasi, mempertahankan raw values, serta memasang imputasi median numerik dan modus kategorikal di dalam training fold. LR dan LightGBM memakai one-hot untuk kategori; HGB memakai ordinal coding. Tiga interaksi diuji terpisah; nilai rasio tak hingga menjadi missing sebelum imputasi. Data saat ini tidak memiliki missing cell, tetapi imputer merupakan bagian pipeline. Pipeline tidak memvalidasi makna nilai negatif.

## 3.9 EDA

Notebook 04 memeriksa distribusi target, empat fitur numerik menurut target, dan empat fitur kategori. Notebook 05 menghitung asosiasi fitur-target (rank-biserial atau Cramér’s V) dan Spearman antar-prediktor. Notebook 06 menampilkan p-value Benjamini–Hochberg. Asosiasi/p-value bukan sebab-akibat dan dipengaruhi ukuran sampel.

## 3.10 Feature Engineering

| Model | Fitur | ROC-AUC ± SD | PR-AUC ± SD |
|---|---|---:|---:|
| Logistic Regression | Raw | 0,585437 ± 0,003905 | 0,551053 ± 0,002281 |
| Logistic Regression | Interaksi | 0,585517 ± 0,003789 | 0,551199 ± 0,002165 |
| HistGradientBoosting | Raw | 0,586347 ± 0,003246 | 0,552064 ± 0,002376 |
| HistGradientBoosting | Interaksi | 0,585704 ± 0,003985 | 0,551442 ± 0,002219 |
| LightGBM | Raw | 0,586576 ± 0,003231 | 0,553064 ± 0,002638 |
| LightGBM | Interaksi | 0,586535 ± 0,003175 | 0,553219 ± 0,002359 |

Efek kecil dan campuran; raw features menjadi default. Histogram interaksi ada pada **Gambar 16–18**.

## 3.11 Feature Selection

L1 fold-local menghasilkan ROC-AUC 0,583493 (SD 0,005381) dan PR-AUC 0,548963 (SD 0,003794), dibandingkan tanpa seleksi 0,585437 dan 0,551053. Waktu fit rata-rata naik 1,49 ke 2,59 detik per fold. Dua belas encoded columns dipilih per fold, sepuluh konsisten. Selector tidak dipakai. RFE, Boruta, mutual information, dan seleksi SHAP tidak dijalankan.

## 3.12 Pembagian Data dan Validasi

Split awal memisahkan 160.000 development dan 40.000 internal holdout, stratified seed 42. Corrected CV memakai tiga fold pada development, sekitar 53.333 baris validasi per fold. Evaluasi historis membagi development menjadi 120.000 fit dan 40.000 calibration lalu menilai 40.000 holdout. Decision-study menggunakan subset tersendiri. Split acak tidak menggantikan validasi temporal. Holdout lama telah dipakai model historis dan bukan holdout baru kandidat corrected.

## 3.13 Model Development

| Model | Konfigurasi utama | Mean ROC-AUC | SD fold |
|---|---|---:|---:|
| Logistic Regression | class_weight balanced; max_iter 2000; numeric scaling | 0,585437 | 0,003905 |
| HistGradientBoosting | 48 iterasi; learning_rate 0,08; kategori ordinal | 0,586347 | 0,003246 |
| LightGBM | 48 estimator; learning_rate 0,05; num_leaves 31; balanced; n_jobs 1; seed 42 | **0,586576** | 0,003231 |

Selisih LightGBM-HGB 0,000229 tanpa repeated CV atau interval berpasangan; hanya mean tertinggi pada desain ini yang didukung.

## 3.14 Hyperparameter Tuning

Pilot Optuna TPE/SQLite menjalankan dua trial LightGBM pada tiga fold: 0,586433 dan 0,587598. Parameter terbaik antara lain n_estimators=132, learning_rate=0,10198, num_leaves=16, max_depth=10, min_child_samples=30, subsample=0,7550, colsample_bytree=0,8574. Dua trial adalah smoke test, bukan optimum, dan bukan konfigurasi kandidat refit default.

## 3.15 Ensemble

OOF single model: LR 0,585378 ROC-AUC, HGB 0,586270, LightGBM 0,586492. Blend LR+HGB memiliki ROC-AUC 0,587648 dan PR-AUC 0,553586; LR+LightGBM 0,587528 dan 0,553849; HGB+LightGBM 0,587210 dan 0,553152. Gain kecil; prediksi berkorelasi tinggi; tidak ada repeated CV atau holdout independen.

## 3.16 Kalibrasi

Sigmoid, isotonic, dan beta diuji pada soft vote historis melalui training-side CV. Beta dipakai pada holdout historis: Brier 0,243089 dan log loss 0,679141; tanpa kalibrasi 0,2433 dan 0,6796. Perbaikan kecil, ranking nyaris sama. Sigmoid memiliki ECE CV sedikit lebih rendah. Kandidat LightGBM terkoreksi belum dikalibrasi.

## 3.17 Evaluasi Model dan Protokol

| Protokol | Model | Hasil | Interpretasi |
|---|---|---|---|
| Corrected raw 3-fold CV | LightGBM | ROC-AUC 0,586576; SD 0,003231 | Mean tertinggi single model pada satu desain |
| Corrected raw 3-fold CV | HistGradientBoosting | 0,586347; SD 0,003246 | Dekat dengan LightGBM |
| Corrected raw 3-fold CV | Logistic Regression | 0,585437; SD 0,003905 | Baseline sederhana |
| Corrected OOF | LR+HGB blend | ROC-AUC 0,587648; PR-AUC 0,553586 | Eksploratif, bukan holdout |
| Historical holdout 40.000 | Soft vote HGB+LR, beta | ROC-AUC 0,589481; AP 0,556060; Brier 0,243089; log loss 0,679141 | Pipeline lama, bukan corrected raw |

Threshold 0,50 pada holdout historis: precision 0,554684, recall 0,435735, F1 0,488067; TN=14.375, FP=6.641, FN=10.712, TP=8.272. Baseline prevalensi konstan memiliki Brier 0,249355 dan log loss 0,691856. Nilai ini tidak dialihkan ke kandidat terkoreksi.

## 3.18 Prioritisasi Risiko

Artifact lokal historis menghasilkan raw score dan probabilitas beta-calibrated. Antrean mengurutkan probabilitas menurun dan memberi priority rank; pengguna menentukan jumlah yang masuk kapasitas. Frontend statis hanya menerima probability column valid dan tidak membuat prediksi.

## 3.19 Simulasi Kapasitas

Kapasitas historis: 5%, 10%, 20%, 30%. Skenario medium memakai biaya asumsi FP=1, FN=5, review=0,25. Satu ablation pada 10%: ranking probabilitas dan cost-sensitive masing-masing berbiaya 68.140, precision@K 0,6144, recall@K 0,1295. Calibration+cost+uncertainty: 70.010, precision 0,5169, recall 0,1089.

Analisis kapasitas lain melaporkan probability ranking 10%: biaya 85.090, precision 0,6178, recall 0,1302, FP 1.529, FN 16.513; random review biaya 88.480 dan precision 0,4768. Subset/protokol berbeda dari ablation 68.140; tidak digabungkan. Biaya simulasi dan asumsi review sempurna bukan dampak terukur.

## 3.20 Arsitektur ReturnWise

Alur lokal: CSV/Kaggle connector → profiling/feature audit → preprocessing dan model tersimpan → scoring → ranking antrean → kapasitas dan ekspor CSV. Package returnwise mencakup acquisition/profiling, feature transform, model, calibration, decision logic, explanation, export, storage, dan report. Streamlit memiliki enam halaman. Artifact joblib historis dan metadata disimpan lokal.

Frontend site/ adalah HTML/CSS/JavaScript statis. CSV dibaca di browser; tidak ada scoring API/model joblib pada situs tersebut. Ranking hanya mungkin bila CSV tanpa label sudah berisi probability yang valid. Tidak ditemukan database atau role management.

## 3.21 Tools dan Teknologi

| Komponen | Teknologi |
|---|---|
| Bahasa/notebook | Python, Jupyter |
| Analisis | pandas, NumPy, SciPy, scikit-learn |
| Visualisasi | Matplotlib, pandas plotting |
| Model corrected | LightGBM |
| Pilot tuning | Optuna, SQLite |
| Aplikasi lokal | Streamlit |
| Static frontend | HTML, CSS, JavaScript, konfigurasi Netlify |
| Artifact/state | joblib, JSON |

Audit mencatat Python 3.12.10, Streamlit 1.66.1, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0, Optuna 4.9.0. Notebook 00 adalah snapshot awal ketika beberapa paket opsional belum tersedia.

---

# BAB IV — HASIL DAN PEMBAHASAN

## 4.1 Hasil Data Understanding

Dataset memiliki 200.000 train dan 50.000 test. Prevalensi positif 47,4595%. **Gambar 3** menunjukkan distribusi target; **Gambar 2** membandingkan ukuran berkas, bukan jumlah observasi.

## 4.2 Hasil Data Quality

Missingness dan duplikasi bukan masalah utama. Risiko utama adalah semantik nilai dan provenance waktu. **Gambar 5** memperlihatkan outlier/nilai negatif harga; **Gambar 6–7** distribusi diskon/rating; **Gambar 14** p-value terkoreksi BH. Nilai anomali dipertahankan.

## 4.3 Hasil EDA

Boxplot **Gambar 4–7** membandingkan distribusi fitur numerik menurut returned. Tumpang tindih distribusi tidak menentukan kegunaan fitur gabungan. **Gambar 8–11** menunjukkan variasi proporsi retur menurut kategori; perbedaan dapat mencerminkan komposisi data/confounding, bukan sebab-akibat. **Gambar 12** mencampur ukuran efek berbeda sehingga sumbunya harus dibaca hati-hati. **Gambar 13** adalah korelasi antar-prediktor, bukan fitur-target. **Gambar 14** tidak menggantikan ukuran efek atau CV.

## 4.4 Hasil Preprocessing

**Gambar 15** menunjukkan split 160.000/40.000; test kompetisi tidak dipakai untuk evaluasi berlabel. Split acak tidak menguji periode mendatang.

## 4.5 Hasil Feature Engineering

Histogram **Gambar 16–18** menunjukkan distribusi interaksi miring dengan ekor ekstrem, khususnya rasio tampilan/menit. Perubahan tiga-fold kecil dan campuran; raw features tetap default.

## 4.6 Hasil Feature Selection

Seleksi L1 menurunkan ROC-AUC sekitar 0,00194 dan PR-AUC sekitar 0,00209 serta menaikkan waktu fit. Tidak ada dukungan untuk mengaktifkan selector.

## 4.7 Hasil Model Comparison

**Gambar 19–21** adalah output notebook tersimpan dari waktu eksekusi masing-masing. Gambar 19 memuat baseline awal, Gambar 20 benchmark historis yang belum mencakup kandidat LightGBM corrected, dan Gambar 21 blend notebook tersimpan. Grafik tersebut tidak menggantikan tabel CV terkoreksi karena protokol/pipeline berbeda.

## 4.8 Hasil Cross-Validation

LightGBM mean ROC-AUC tertinggi, tetapi selisih 0,000229 dari HGB lebih kecil daripada variasi fold tercatat. Tidak ada paired test, repeated CV, atau interval. Hasil mendukung kandidat untuk evaluasi lanjutan, bukan deployment winner.

## 4.9 Hasil Ensemble

Blend LR+HGB memiliki OOF ROC-AUC 0,587648; LR+LightGBM memiliki OOF PR-AUC 0,553849. Satu desain OOF tidak membuktikan stabilitas/temporal generalization. Gambar 21 dibaca sesuai output notebooknya.

## 4.10 Hasil Kalibrasi

Beta memperbaiki Brier/log loss model historis hanya sedikit. **Gambar 22** menampilkan Brier model lama; **Gambar 28** reliability diagram holdout lama. Titik dekat diagonal pada sepuluh bin tidak membuktikan kalibrasi di populasi lain. Tidak ada calibration plot LightGBM corrected.

## 4.11 Hasil Risk Prioritization

Ranking mengikuti probabilitas. Kalibrasi monotonic mempertahankan urutan; manfaatnya untuk interpretasi peluang/kasus harapan. **Gambar 24** menunjukkan uncertainty flag; **Gambar 25–27** menampilkan simulasi dan ablation.

## 4.12 Hasil Capacity Simulation

Pada ablation 10%, cost-sensitive sama dengan probability ranking akibat fungsi biaya monotonic; uncertainty-aware lebih buruk. Tabel kapasitas lain membandingkan probability dan random review tetapi berasal dari protokol berbeda. **Gambar 26** menunjukkan recall@K meningkat saat kapasitas naik; ini akibat lebih banyak baris direview, bukan bukti intervensi mencegah retur. **Gambar 29** meletakkan metrik berbeda pada satu sumbu; tinggi batang tidak membandingkan arah/kualitas metrik secara langsung.

## 4.13 Pembahasan

Data lengkap secara struktur masih memiliki masalah nilai dan waktu fitur. Corrected raw-feature models memiliki ranking di atas acuan acak pada mean CV, tetapi ROC-AUC sekitar 0,586 menunjukkan sinyal terbatas. Selisih kecil memerlukan replikasi. Holdout historis 0,589481 berasal dari model/pipeline berbeda dan bukan bukti kandidat corrected lebih baik.

Kalibrasi dan simulasi menunjukkan cara probabilitas dapat dihubungkan ke keputusan, namun biaya dan efektivitas review bukan data empiris. **Gambar 23, “Permutation Importance pada Validation”**, merupakan interpretasi lokal dari validasi pada notebook explainability dan tidak membuktikan sebab-akibat atau kepentingan fitur pada populasi lain. ReturnWise mendemonstrasikan antrean; belum ada bukti perubahan perilaku pengguna atau penurunan retur.

## 4.14 Keterbatasan Hasil

- Skor hanya sedikit di atas ranking acak; tidak ada interval kepercayaan.
- Satu seed/tiga fold tidak membuktikan robustness.
- Tidak ada validasi temporal, eksternal, subgroup calibration, drift monitoring, atau label-noise audit.
- Semantik anomali belum dikonfirmasi.
- Timestamp/snapshot fitur tidak tersedia; status product_rating juga bertentangan antar dokumen.
- Holdout historis bukan evaluasi kandidat terkoreksi.
- LightGBM terkoreksi belum dikalibrasi dan belum mendapat holdout baru.
- Biaya dan efektivitas review adalah asumsi; tidak ada evaluasi kausal/pilot operasional.
- Artifact lama bukan deployment refit kandidat terkoreksi.

## 4.15 Gambaran Umum ReturnWise

ReturnWise adalah workspace penelitian untuk estimasi risiko retur dan prioritisasi antrean pemeriksaan. Calon pengguna berdasarkan fungsi: analis data, peneliti, pengelola operasi, dan tim pemeriksa. Repository tidak menunjukkan autentikasi/role management.

## 4.16 Tujuan Website

Website menyatukan pemeriksaan dataset, informasi model, antrean risiko, dan skenario kapasitas. UI membantu memeriksa keluaran tetapi tidak mengesahkan model.

## 4.17 Pengguna

Pengguna memasukkan dataset dan meninjau hasil; tidak ada akses berbasis peran yang tercatat. Keputusan tindak lanjut berada pada pengguna, bukan model.

## 4.18 Fitur Website

| Fitur | Input/proses | Output/manfaat | Batasan |
|---|---|---|---|
| Overview | Status dataset/artifact | Ringkasan alur dan status eksperimen | Metrik historis dilabeli historis |
| Dataset and quality | Kaggle cache/CLI atau CSV lokal; validasi skema, missing, duplikat, ID, flag nilai | Preview, dictionary, quality flags, target balance | Tidak memperbaiki otomatis; Kaggle perlu akses lokal |
| EDA and statistics | Dataset aktif | Ringkasan numerik/kategori, target, asosiasi dan p-value | Asosiasi bukan kausal |
| Risk queue | CSV tanpa target dan artifact historis | Skor raw/calibrated, ranking, kapasitas, pencarian, ekspor | Model eksperimental; target returned harus dihapus |
| On-demand explanation | Baris pilihan dan background sample | SHAP jika perhitungan/dependensi tersedia | SHAP dapat gagal/tidak tersedia; kontribusi bukan sebab |
| Decision simulator | Antrean, kapasitas, biaya, threshold, confidence ceiling | Kasus ekspektasian, antrean, perbandingan strategi | Asumsi skenario, bukan biaya aktual |
| Models and research | Metadata, eksperimen, artifact | Status protokol, metrik, report download | Tidak ada corrected holdout final |

### Frontend statis

site/ memprofil CSV di browser, menunjukkan baris/kolom/missing/duplikat dan mengurutkan baris bila file tanpa label berisi calibrated_probability atau return_probability valid. Pengguna memilih kapasitas, mencari, dan mengunduh antrean. Jika target hadir atau skor tidak valid/tidak ada, ranking dinonaktifkan. Demo rows bersifat ilustratif. Netlify static site tidak menjalankan Python, memuat joblib, atau menyediakan scoring API.

## 4.19 Alur Penggunaan

1. Pilih dataset kompetisi tervalidasi atau impor CSV lokal.
2. Periksa skema, kualitas, identifier, dan risiko timing fitur.
3. Tinjau pola pada halaman EDA.
4. Untuk scoring Streamlit, unggah transaksi tanpa target; artifact lama memberi skor jika fingerprint sesuai.
5. Urutkan probabilitas dan pilih jumlah review.
6. Jalankan simulator dengan asumsi kapasitas/biaya.
7. Tinjau dan ekspor antrean/laporan dengan memperhatikan status eksperimen.

Frontend statis menerima skor probabilitas yang sudah ada; ia tidak melakukan scoring.

## 4.20 Arsitektur Teknis

streamlit_app.py mendaftarkan enam halaman. Package returnwise menangani data, model, kalibrasi, keputusan, penjelasan, export, storage, dan report. Dataset/state/artifact disimpan lokal; notebook berurutan memakai joblib state. Frontend site/ berjalan di browser sebagai static HTML/CSS/JavaScript. Tidak ditemukan API inferensi atau database. Status deployment Netlify tidak bisa dipastikan dari kode.

## 4.21 Perbedaan Model dan Sistem

Model mengubah fitur menjadi skor. Sistem mencakup input, validasi, penerapan model, ranking, kontrol kapasitas, interpretasi, dan ekspor. UI tidak menggantikan validasi statistik/operasional.

## 4.22 Manfaat dan Batasan Website

Manfaat aktual: profiling lebih terstruktur, protokol hasil dapat dibaca terpisah, dan prioritisasi kapasitas dapat didemonstrasikan. Batasannya: artifact historis, model corrected belum refit, timing fitur belum terbukti, simulasi memakai asumsi, dan static site tidak memprediksi. Tidak ada klaim retur dicegah, biaya turun, pelanggan lebih puas, atau profit naik.

---

# BAB V — KESIMPULAN DAN SARAN

## 5.1 Kesimpulan

1. Dataset lokal berisi 200.000 train dan 50.000 test tanpa label. Prevalensi retur 47,4595%; missing cell/duplikat persis tidak ditemukan, tetapi sejumlah nilai di luar rentang belum dijelaskan.
2. Corrected pipeline mengevaluasi kandidat raw 11 kolom; feature interaction kecil/campuran dan seleksi L1 menurunkan metrik.
3. Mean ROC-AUC tiga fold adalah 0,586576 LightGBM, 0,586347 HGB, 0,585437 LR. Selisih terlalu kecil untuk menyatakan pemenang robust. Blend LR+HGB OOF 0,587648 belum punya holdout independen.
4. Holdout 0,589481 dan metrik kalibrasi merupakan hasil soft vote historis. Kandidat corrected belum dikalibrasi atau dievaluasi pada holdout baru.
5. ReturnWise membuat ranking dan simulasi kapasitas dari probabilitas tersimpan; belum ada bukti biaya bisnis atau efektivitas review. Ranking fixed-cost sama dengan probability ranking secara matematis.
6. Refit LightGBM terkoreksi dihentikan pada preflight RAM sebelum training. Artifact dan CSV prediksi yang tersedia adalah export historis.

## 5.2 Saran

- Minta definisi nilai anomali kepada pemilik data.
- Tetapkan timestamp keputusan, snapshot, dan cutoff setiap fitur.
- Putuskan status product_rating dan ulangi eksperimen untuk feature set final.
- Jika sumber daya memadai, jalankan repeated CV, paired uncertainty intervals, dan holdout corrected yang tidak dipakai untuk seleksi.
- Kalibrasikan kandidat corrected pada partisi terpisah; laporkan Brier/log loss/reliability dan ukuran tambahan.
- Gunakan validasi temporal setelah timestamp tersedia.
- Ukur biaya review dan uji apakah intervensi mengubah hasil.
- Selesaikan refit, metadata, parity check, validasi ID/schema, dan submission sebelum menyatakan siap deploy.

## 5.3 Penelitian Lanjutan

Uji threshold dan kapasitas dengan biaya terukur; nilai ranking per kelompok dan periode; bandingkan model graph/history ketika data point-in-time tersedia; dan jalankan pilot terkontrol. Klaim novelty perlu dibandingkan secara sistematis dengan riset return prediction, calibration, cost-sensitive learning, dan capacity-constrained review.

---

# Daftar Pustaka

## Referensi akademik

- Cui, H., Rajagopalan, S., & Ward, A. R. (2020). Predicting product return volume using machine learning methods. *European Journal of Operational Research, 281*(3), 612–627. [DOI](https://doi.org/10.1016/j.ejor.2019.05.046)
- Dzyabura, D., El Kihal, S., Hauser, J. R., & Ibragimov, M. (2023). Leveraging the power of images in managing product return rates. *Marketing Science, 42*(6), 1125–1142. [DOI](https://doi.org/10.1287/mksc.2023.1451)
- Duong, Q. H., Zhou, L., Nguyen, V. T., & Meng, M. (2025). Understanding and predicting online product return behavior: An interpretable machine learning approach. *International Journal of Production Economics, 280*, 109499. [DOI](https://doi.org/10.1016/j.ijpe.2024.109499)
- Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of IJCAI* (pp. 973–977). [IJCAI](https://ijcai.org/proceedings/2001/133)
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. [DOI](https://doi.org/10.1016/j.patrec.2005.10.010)
- Karl, D. (2025). Forecasting e-commerce consumer returns: A systematic literature review. *Management Review Quarterly, 75*, 1–56. [DOI](https://doi.org/10.1007/s11301-024-00436-x)
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- Kull, M., Silva Filho, T. M., & Flach, P. (2017). Beta calibration: A well-founded and easily implemented improvement on logistic calibration for binary classifiers. In *Proceedings of AISTATS* (PMLR 54, pp. 623–631). [PMLR](https://proceedings.mlr.press/v54/kull17a.html)
- Ma, S., & Wang, W. (2024). Proactive return prediction in online fashion retail using heterogeneous graph neural networks. *Electronics, 13*(7), 1398. [DOI](https://doi.org/10.3390/electronics13071398)
- Mishra, A., & Dutta, P. (2024). Return management in e-commerce firms: A machine learning approach to predict product returns and examine variables influencing returns. *Journal of Cleaner Production, 477*, 143802. [DOI](https://doi.org/10.1016/j.jclepro.2024.143802)
- Zhu, Y., Li, J., He, J., Quanz, B. L., & Deshpande, A. A. (2018). A local algorithm for product return prediction in e-commerce. In *Proceedings of IJCAI-18* (pp. 3718–3724). [DOI](https://doi.org/10.24963/ijcai.2018/517)

## Dataset dan dokumentasi teknis

- Kaggle. (2026). *E-Commerce Returns: Who Will Send It Back?* [Kompetisi](https://www.kaggle.com/competitions/retail-return-risk-modeling)
- ReturnWise repository (2026). Laporan internal: FINAL_DATA_QUALITY.md, FINAL_MODEL_EVALUATION.md, FINAL_EXPERIMENT_LOG.md, FINAL_DEPLOYMENT_DATA_STATUS.md, FINAL_RETURNWISE_RESULTS.md, dan deep_research_audit.md.
- ReturnWise repository (2026). Source code pada returnwise/, app_pages/, site/, dan notebooks/00–20.
- scikit-learn developers (2026). Dokumentasi scikit-learn. [Situs resmi](https://scikit-learn.org/)
- Streamlit (2026). Dokumentasi Streamlit. [Situs resmi](https://docs.streamlit.io/)

---

# Lampiran

## A. Notebook yang diperiksa

Seluruh 21 notebook diperiksa: 00 environment setup; 01 Kaggle acquisition; 02 dataset understanding; 03 data quality; 04 EDA; 05 statistical analysis; 06 problem identification; 07 preprocessing; 08 feature engineering; 09 baseline ML; 10 advanced benchmarking; 11 hyperparameter optimization; 12 ensemble; 13 calibration; 14 explainable AI; 15 uncertainty; 16 cost-sensitive decision; 17 adaptive priority; 18 ablation; 19 final historical evaluation; 20 historical export.

## B. Laporan repository yang ditelaah

FINAL_RETURNWISE_RESULTS.md, FINAL_DATA_QUALITY.md, FINAL_MODEL_EVALUATION.md, FINAL_EXPERIMENT_LOG.md, FINAL_DEPLOYMENT_DATA_STATUS.md, returnwise_research_report.md, final_research_findings.md, deep_research_audit.md, dataset_deep_analysis.md, model_comparison.md, feature_engineering_analysis.md, feature_selection_analysis.md, calibration_analysis.md, decision_optimization.md, hyperparameter_optimization.md, advanced_experiment_results.md, ablation_robustness.md, dan explainability_analysis.md.

## C. Artifact dan implementasi

- Dataset: data/kaggle/train.csv, test.csv, sample_submission.csv, metadata.
- Artifact lama: artifacts/returnwise_model.joblib dan artifacts/metadata.json.
- Prediksi lama: data/kaggle/returnwise_submission.csv; timestamp sejalan export Notebook 20 pada 2026-11-15, bukan prediksi LightGBM terkoreksi.
- State/riwayat eksperimen: experiments/research_state.joblib dan rekaman lokal.
- Source aplikasi: streamlit_app.py, app_pages/, returnwise/, site/.
- Script: scripts/run_research_pipeline.py, scripts/build_notebooks.py, scripts/finalize_model_submission.py.
- Repository mencatat suite integrasi sebelumnya 6 passed; suite tidak dijalankan ulang selama penyusunan dokumentasi ini.

## D. Grafik

29 PNG diekstrak dari output notebook dan didokumentasikan satu per satu—termasuk judul, notebook sumber, uraian, interpretasi, dan batas pembacaan—pada [GRAFIK_HASIL.md](GRAFIK_HASIL.md). Penomoran di sana digunakan pada laporan ini.

## E. Informasi yang belum tersedia/perlu keputusan

1. Nama penulis, institusi, dan data administratif.
2. Definisi sumber nilai negatif/out-of-range.
3. Timestamp transaksi, snapshot fitur, dan cutoff histori.
4. Keputusan final atas product_rating.
5. Holdout baru, repeated CV, interval, dan kalibrasi kandidat terkoreksi.
6. Refit model, metadata final, dan submission dari pipeline terkoreksi.
7. Biaya review aktual, efektivitas pemeriksaan, dan evaluasi lapangan.
8. Tinjauan literatur sistematis penuh dan pemeriksaan solusi peserta Kaggle.
