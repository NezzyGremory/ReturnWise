# Audit Teknis Pasca Implementasi — Smart Retail

**Tanggal audit:** 16 November 2026

**Status:** Layak untuk penggunaan lokal terbatas setelah backup operasional. **Belum siap untuk deployment produksi atau akses bersama.**

## Ringkasan

Smart Retail adalah aplikasi Streamlit multi-halaman dengan layanan transaksi Python dan database SQLite satu toko. Audit menemukan dan memperbaiki beberapa cacat perhitungan dan keamanan: pergerakan dari transaksi refund/batal tidak lagi dihitung sebagai permintaan aktif; alokasi diskon laporan sekarang mempertahankan nilai rupiah total; rentang laporan mengikuti zona waktu bisnis; prioritas segmen pelanggan loyal tidak tertimpa; backup dibuat dari satu snapshot dan tidak bisa melewati batas ukuran restore; kegagalan penyimpanan checkout sekarang dibatalkan seluruhnya dan ditampilkan sebagai kesalahan aplikasi; ekspor CSV meloloskan teks yang menyerupai formula spreadsheet; dan server lokal default hanya bind ke loopback.

Suite akhir lulus **31/31 tes**: 18 tes yang sudah ada dan 13 regression test tambahan untuk audit. Seluruh 10 halaman Smart Retail dirender melalui Streamlit AppTest, checkout UI diuji, dan server Streamlit lokal memberikan HTTP 200 pada halaman serta health endpoint.

Belum ada autentikasi, skema migrasi database, deployment Python terkonfigurasi, disk persisten untuk hosting, atau backup terjadwal. Karena itu aplikasi tidak dinyatakan production-ready.

## Arsitektur aktual

- `streamlit_app.py` mendaftarkan halaman Smart Retail dan ReturnWise dalam satu aplikasi Streamlit.
- Smart Retail memiliki 10 halaman operasional. Logika bisnis dipisah ke [services.py](../smart_retail/services.py), ringkasan dan laporan ke [analytics.py](../smart_retail/analytics.py), prediksi dan rekomendasi ke [intelligence.py](../smart_retail/intelligence.py), serta ekspor/pemulihan ke [backup.py](../smart_retail/backup.py).
- Persistensi adalah SQLite file tunggal, satu toko dengan `STORE_ID = 1`. Tidak ada REST API, layanan backend terpisah, sinkronisasi antarperangkat, atau payment gateway.
- `site/` adalah frontend ReturnWise statis tersendiri. Konfigurasi Netlify hanya memublikasikan direktori `site/` melalui `publish = "site"` di `netlify.toml`.
- Workspace `returnwise/` tidak diubah dan tidak ada retraining model ReturnWise dalam audit ini.

## Database dan persistensi

| Pemeriksaan | Hasil |
|---|---|
| Engine | SQLite melalui modul Python `sqlite3`; foreign keys aktif dan koneksi memakai `busy_timeout` 30 detik. |
| Lokasi default | `data/smart_retail.sqlite3`; dapat diganti dengan `SMART_RETAIL_DB` sebelum proses aplikasi dimulai. |
| Database yang diperiksa | File default ada, 172.032 byte, `PRAGMA integrity_check` menghasilkan `ok`, journal mode `delete`, tanpa file WAL/SHM. |
| Isi database saat audit | Satu baris `stores`, satu `app_settings`; tabel pengguna, produk, pelanggan, transaksi, pembayaran, gerakan stok, forecast, dan audit log masih kosong. Tidak ada transaksi pengguna yang ditemukan atau dibaca. |
| Restart/persistensi | Tes membuat produk dan transaksi pada file sementara, lalu proses Python baru membuka file yang sama. Transaksi dan stok tetap ada. Data operasi ditulis dengan transaksi SQLite dan commit setelah seluruh operasi berhasil. |
| Database sementara | Fixture tes menggunakan `tmp_path`; uji server juga memakai file sementara. Runtime normal tidak memilih database sementara: default-nya file tetap di `data/`. |
| Backup/pemulihan | Backup JSON memuat tabel aplikasi dari satu snapshot SQLite. Restore memeriksa format, versi, ukuran maksimum 50 MiB, batasan skema dan foreign keys, lalu mengganti data dalam satu transaksi dan mencatat aksi restore. Round-trip, penolakan file rusak, serta snapshot saat checkout bersamaan telah diuji. |

Backup saat ini hanya manual. JSON tidak dienkripsi atau ditandatangani dan memuat data pelanggan serta transaksi; simpan file dengan ACL perangkat yang sesuai. Validator belum mencocokkan seluruh invariant bisnis lintas tabel, seperti jumlah item terhadap subtotal atau saldo stok terhadap seluruh ledger. Database memakai `CREATE TABLE IF NOT EXISTS` tanpa mekanisme migrasi/versi skema; perubahan skema setelah data operasional masuk perlu prosedur migrasi sebelum upgrade.

## Integritas POS dan inventaris

| Skenario | Hasil verifikasi |
|---|---|
| Checkout normal | Berhasil; item, pembayaran, pengurangan stok, movement, dan audit log tersimpan. |
| Stok tidak cukup | Ditolak sebelum transaksi atau pembayaran dibuat; stok tidak berubah. |
| Double submission bersamaan | Dua checkout dengan kunci idempotensi sama menghasilkan satu transaksi, satu pembayaran, dan satu movement penjualan. |
| Diskon, pajak, pembayaran tunai, kembalian | Diuji: subtotal Rp2.400 − diskon Rp200 + pajak 11% Rp242 = Rp2.442; pembayaran Rp2.500 menghasilkan kembalian Rp58. Pembayaran kurang ditolak. |
| Non-tunai | Jumlah yang dicatat harus sama dengan total transaksi. QRIS/kartu hanya metode yang dicatat, tidak terhubung ke gateway. |
| Refund | Refund penuh membuat catatan pembayaran negatif; stok dikembalikan bila opsi restock dipilih. Refund parsial tidak tersedia. |
| Pembatalan | Status dan alasan dipertahankan. Stok hanya kembali bila restock dipilih; tanpa restock, pengurangan fisik tetap ada. |
| Kegagalan penyimpanan | Trigger yang sengaja menggagalkan insert pembayaran membuktikan rollback: tidak ada transaksi, item, pembayaran, audit checkout, atau movement penjualan parsial; stok tetap sama. |

Checkout menghitung `subtotal = Σ(harga × kuantitas)`, mengurangi diskon, menghitung pajak pada nilai setelah diskon, lalu menambahkan pajak. Tarif dibatasi 0–100%. Pembulatan pajak memakai `round()` Python, yaitu pembulatan ties-to-even; kebijakan pembulatan rupiah untuk invoice/pelaporan pajak perlu disetujui pemilik proses sebelum pemakaian akuntansi formal. Kunci idempotensi saat ini tidak mengikat payload asal; penggunaan ulang kunci yang sama untuk keranjang berbeda dapat mengembalikan transaksi lama. UI menghasilkan kunci baru per checkout, namun layanan sebaiknya menyimpan atau membandingkan fingerprint permintaan.

Inventaris memakai batas stok nonnegatif pada aplikasi dan skema. Gerakan penjualan/refund/pembatalan dicatat di transaksi yang sama. Metrik permintaan sekarang hanya menjumlah penjualan dari transaksi berstatus selesai, sehingga refund dan pembatalan tidak membesarkan rekomendasi permintaan.

## Retail Intelligence dan analitik

Tidak ditemukan fitur placeholder pada forecasting, rekomendasi stok, RFM, atau laporan penjualan. Implementasinya deterministik dan menggunakan data transaksi lokal; ini bukan model ML terlatih dan tidak menggunakan model ReturnWise.

| Fitur | Implementasi dan data | Data minimum/validasi | Batasan |
|---|---|---|---|
| Sales forecasting | Rata-rata bergerak 7 hari atas unit atau penjualan bersih per hari. Prediksi 7/30 hari dibuat secara rekursif dari tujuh nilai terakhir. Forecast, versi baseline, dan metrik disimpan. | Minimal 28 hari kalender dan penjualan pada 7 hari. Validasi time-ordered pada holdout akhir (`min(7, max(1, panjang/5))`) dengan MAE, RMSE, dan WAPE; setiap prediksi holdout hanya memakai hari sebelumnya. | Bukan model terlatih; tidak menangkap tren, musim, promo atau hari libur; tidak menyediakan interval ketidakpastian. Hari tanpa transaksi dianggap nol sehingga pencatatan yang tidak rutin membiasakan hasil. Tidak ada baseline pembanding atau validasi lintas toko. |
| Demand prediction / saran stok | Unit permintaan selesai selama 30 hari dibagi jumlah hari observasi. Lead time supplier digunakan, atau default 7 hari. Titik pesan = permintaan harian × lead time + safety stock. Target menambah 7 hari permintaan. | Minimal 14 hari sejak penjualan selesai pertama sebelum jumlah saran diberikan; jika belum cukup, jumlah saran kosong. | Safety stock adalah heuristik `ceil(sqrt(mean harian) × sqrt(lead time))`, asumsi varians mirip Poisson; bukan estimasi empiris yang diuji. Kehilangan penjualan saat stok habis tidak dimodelkan, sehingga permintaan dapat terpotong. Tidak ada backtest rekomendasi. |
| RFM | Recency = hari sejak transaksi selesai terbaru; Frequency = jumlah transaksi selesai; Monetary = subtotal setelah diskon, sebelum pajak. Segmen menggunakan aturan umur 30/60/90 hari, frekuensi, dan kuartil ke-75 Monetary. | Tidak ada jumlah pelanggan minimum atau validasi eksternal. Tes tambahan memastikan segmen loyal prioritas tinggi tidak tertimpa aturan segmen umum. | Segmentasi aturan, bukan RFM quantile/score baku. Hasil berubah menurut komposisi pelanggan toko dan bukan karakter permanen. |
| Sales analytics | Penjualan bersih, pajak, total transaksi, produk/kategori, HPP dari snapshot biaya saat transaksi, dan turnover proxy dari HPP 30 hari ÷ nilai stok saat ini. | Data berasal dari transaksi berstatus selesai dan rentang waktu bisnis; tes memastikan alokasi diskon antar-item mempertahankan total rupiah. | Rata-rata pesanan memakai `total` termasuk pajak, sedangkan penjualan bersih tidak. Kategori historis mengikuti kategori produk saat ini karena item transaksi belum menyimpan snapshot kategori. Mengubah kategori produk dapat mengubah laporan/forecast kategori masa lalu. Turnover memakai nilai stok sekarang, bukan rata-rata stok historis, sehingga hanya proksi. |

Tanggal transaksi disimpan UTC. Sebelum audit, filter harian memakai tanggal UTC dan dapat salah mengelompokkan transaksi setelah tengah malam Jakarta. Periode laporan, RFM, forecast, rekomendasi stok, dan waktu struk sekarang mengikuti zona bisnis `Asia/Jakarta`, dapat diubah dengan `SMART_RETAIL_TIMEZONE`. Tes menempatkan transaksi pukul 00:30 waktu toko untuk memverifikasi filter harian.

## Keamanan

- **Validasi input:** teks wajib dipangkas dan divalidasi; kuantitas/harga/diskon menggunakan validasi integer presisi berbasis `Decimal` dengan batas signed 64-bit; batasan stok, status, foreign key dan nilai nonnegatif juga ditegakkan SQLite. Email, nomor telepon, dan panjang field belum memiliki validasi format/batas eksplisit.
- **SQL injection:** nilai pencarian dan data layanan dikirim sebagai parameter SQL. Tes memasukkan fragmen SQL sebagai pencarian dan mendapat hasil literal kosong. Identifier pada backup dibatasi oleh daftar tabel dan kolom skema.
- **Password/auth:** tidak ada layar login atau otorisasi; tabel `users` belum dipakai. Tidak ada password yang disimpan, tetapi semua halaman dan pemulihan data terbuka bagi siapa pun yang dapat mengakses server.
- **Binding lokal:** sebelum audit, proses Streamlit mengiklankan IP LAN sementara aplikasi tanpa login. Konfigurasi sekarang bind ke `127.0.0.1`; startup lokal hanya menampilkan URL loopback. Jangan mengubahnya ke alamat publik sebelum menambahkan autentikasi dan kontrol akses. Opsi `server.address` memang mengatur alamat listen Streamlit ([dokumentasi konfigurasi Streamlit](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)).
- **Upload:** CSV dibatasi oleh filter ekstensi di UI, decoding UTF-8-SIG, header wajib dan validasi tiap baris; seluruh impor dilakukan atomik. Backup harus JSON UTF-8 dengan struktur, versi, tabel, constraints dan foreign keys yang sesuai. Batas restore/ekspor backup sama-sama 50 MiB. `server.maxUploadSize` global saat ini 512 MiB dan impor CSV belum punya batas byte/baris/ukuran field khusus, sehingga file besar dapat menghabiskan memori.
- **Spreadsheet/HTML:** teks berbahaya untuk formula spreadsheet diberi prefiks aman di ekspor produk/transaksi/laporan; data struk HTML di-escape.
- **Backup dan audit log:** backup tidak dienkripsi. Audit log mencatat perubahan produk, stok, transaksi, pengaturan dan restore, tetapi tidak memiliki actor karena tidak ada autentikasi, tidak mencatat akses/login, dan tidak tahan manipulasi lokal.

## Kesiapan lokal dan deployment

Pengujian lokal dilakukan dengan Python **3.12.10** dan Streamlit **1.66.1**. Aplikasi berjalan melalui `python -m streamlit run streamlit_app.py`; smoke test menggunakan SQLite sementara mendapat `200 OK` untuk `/` dan `/_stcore/health`. Bind lokal default hanya loopback.

Pemakaian satu operator di satu mesin dapat dilakukan dengan membatasi akses OS, menetapkan lokasi `SMART_RETAIL_DB` yang tersimpan, memilih `SMART_RETAIL_TIMEZONE`, dan membuat backup rutin. SQLite cocok untuk pola lokal satu file/instance ini, tetapi tidak disiapkan sebagai database bersama untuk beberapa replica atau filesystem jaringan.

**Deployment belum siap.** Tidak ada Dockerfile, konfigurasi process manager, disk persisten, migrasi skema, pemisahan dependency, atau pengaturan akses produksi. `requirements.txt` mencampur Streamlit dengan notebook, Kaggle dan pustaka riset/model opsional yang berat; deployment Smart Retail sebaiknya memakai dependency set yang dipisah.

Netlify yang ada hanya memublikasikan direktori statis `site/`, bukan aplikasi Streamlit. Netlify Functions menjalankan kode berdasarkan event dalam runtime ephemeral; Netlify juga mendokumentasikan produk Netlify Server untuk server berumur panjang, tetapi repo ini tidak mengonfigurasi adapter/server atau database persisten untuk Smart Retail ([Functions overview](https://docs.netlify.com/build/functions/overview/), [Netlify Server](https://docs.netlify.com/build/server/)). Streamlit memerlukan Python server pada mesin hosting; panduan resminya menunjukkan container dengan proses Streamlit, port dan health check ([panduan Docker Streamlit](https://docs.streamlit.io/deploy/tutorials/docker), [dependency deployment](https://docs.streamlit.io/deploy/concepts/dependencies)). Jadi konfigurasi Netlify saat ini tidak dapat menjalankan backend ini. Pilih host Python/container satu replica dengan volume persisten dan autentikasi, atau rancang ulang backend serta penyimpanan untuk platform serverless.

## Temuan dan risiko tersisa

| Prioritas | Temuan | Status/risiko |
|---|---|---|
| P1 | Tidak ada autentikasi atau role access. | Risiko tinggi bila server diekspos ke LAN publik/hosting. Binding loopback memperkecil risiko penggunaan lokal; belum menyelesaikan skenario host bersama. |
| P1 | Tidak ada backup terjadwal atau recovery off-site. JSON manual plaintext, maksimum 50 MiB. | Risiko kehilangan data perangkat; backup rutin, terenkripsi dan uji pemulihan perlu sebelum operasi penting. |
| P1 | Tidak ada deployment recipe atau volume persisten. | Container/hosting dengan filesystem sementara dapat kehilangan transaksi saat restart/redeploy. Belum layak produksi. |
| P2 | Tidak ada migrasi skema database. | Upgrade yang mengubah tabel dapat gagal pada database lama. Buat migrasi berurutan dan uji restore sebelum rilis data-bearing. |
| P2 | Item transaksi tidak menyimpan kategori pada saat penjualan. | Perubahan kategori produk dapat mereklasifikasi data masa lalu. Simpan `category_id_snapshot` setelah migrasi terencana. |
| P2 | Kunci idempotensi tidak terikat ke fingerprint payload. | Pemakaian ulang kunci oleh pemanggil berbeda dapat mengembalikan transaksi sebelumnya tanpa membandingkan keranjang. UI normal membuat kunci baru, tetapi kontrak layanan perlu diperketat. |
| P2 | Validasi backup memeriksa struktur dan constraints, bukan rekonsiliasi aritmetika/ledger. | Backup hasil aplikasi lolos round-trip, tetapi backup yang diedit dapat berisi relasi yang valid dengan nilai bisnis yang tidak cocok. Tambahkan rekonsiliasi sebelum restore. |
| P2 | Impor CSV tidak membatasi ukuran/jumlah baris/teks. | File sampai batas global 512 MiB diproses di memori; batasi khusus fitur impor sebelum dibuka untuk banyak pengguna. |
| P2 | Aturan pembulatan pajak menggunakan ties-to-even. | Selisih dapat terjadi satu rupiah pada nilai tepat setengah; tetapkan kebijakan pembulatan bisnis/pajak dan tambah tes batas. |
| P3 | Forecast dan rekomendasi stok adalah heuristik sederhana. | Jangan tafsirkan sebagai jaminan penjualan atau jumlah restock optimal. Validasi pada histori toko yang cukup sebelum keputusan otomatis. |

## Perbaikan yang dilakukan selama audit

1. Filter demand/stockout mengecualikan sale dari transaksi refund atau pembatalan.
2. Alokasi diskon item menggunakan largest remainder integer agar total produk, kategori dan forecast sama dengan penjualan bersih.
3. Tanggal filter, laporan, rekomendasi, RFM, forecast dan struk memakai timezone bisnis; default `Asia/Jakarta`.
4. Prioritas segmen RFM loyal bernilai tinggi tidak lagi tertimpa label pembelian bernilai tinggi.
5. Backup membaca satu snapshot, dan batas 50 MiB berlaku pada ekspor serta restore. UI memberi pesan jika backup tidak dapat dibuat.
6. Kegagalan SQLite pada checkout menjadi `RetailError` setelah rollback atomik.
7. Validasi angka menghindari kehilangan presisi float untuk nilai uang besar; ekspor CSV meloloskan formula berbahaya.
8. Server default bind ke loopback, bukan alamat jaringan.

## Hasil regression test

Perintah akhir: `.venv\Scripts\python.exe -m pytest -q`

Hasil: **31 passed in 35.37s**.

18 tes yang sudah ada tetap dijalankan. 13 tes tambahan mencakup perhitungan pajak/diskon, alokasi diskon, timezone harian, konkurensi idempotensi, rollback kegagalan penyimpanan, refund/pembatalan dan permintaan stok, persistensi lintas proses, presisi angka, SQL injection, formula CSV, prioritas RFM, konsistensi snapshot backup, dan batas ukuran backup. AppTest merender seluruh 10 halaman Smart Retail serta menyimpan checkout POS terkonfirmasi. Tes ReturnWise yang sudah ada juga lulus; tidak ada berkas di `returnwise/` yang diubah.

Interaksi kasir diuji melalui AppTest. Operasi katalog, inventaris, laporan, transaksi, dan backup diuji di tingkat layanan serta halaman terkait diuji dapat dirender; seluruh alur form tersebut belum diuji end-to-end melalui browser.

Browser automation agent tidak berhasil memulai Chrome/CDP (`CDP response channel closed`), jadi audit tidak mengklaim hasil interaksi browser penuh. Pengganti yang dijalankan adalah AppTest Streamlit untuk halaman/checkout dan smoke test HTTP lokal untuk halaman utama serta health endpoint.

