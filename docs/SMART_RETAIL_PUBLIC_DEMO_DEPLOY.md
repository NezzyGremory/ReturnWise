# Deploy Smart Retail sebagai demo publik

Smart Retail berjalan sebagai aplikasi Streamlit dan menyimpan transaksi pada SQLite. `netlify.toml` di repo ini hanya menerbitkan situs statis ReturnWise dari folder `site/`. Untuk menjalankan Smart Retail, gunakan host Python.

## Render

Repo ini menyediakan `render.yaml` untuk membuat layanan web Render paket Free dengan proses Streamlit dan health check. Database demo ditulis ke `/tmp/smart_retail_demo.sqlite3` pada filesystem sementara.

1. Pastikan perubahan ini sudah ada di repo Git yang dapat dihubungkan ke Render.
2. Di Render, pilih **New > Blueprint**, lalu hubungkan repo tersebut.
3. Tinjau blueprint `smart-retail-demo`, lalu buat layanan. Blueprint mengatur Python 3.12.10, dependency khusus Smart Retail, alamat listen `0.0.0.0`, port dari `$PORT`, health check, dan zona waktu `Asia/Jakarta`.
4. Tunggu deploy selesai, lalu buka URL `onrender.com` dari halaman layanan. Health check menggunakan `/_stcore/health`.
5. Isi demo secara manual dengan produk dan transaksi fiktif. Jangan impor backup, data pelanggan, atau transaksi toko nyata.

Entrypoint khusus `smart_retail_app.py` hanya menampilkan halaman Smart Retail. `requirements-smart-retail.txt` menghindari pemasangan pustaka riset ReturnWise yang tidak diperlukan oleh demo.

## Batas demo

Demo ini terbuka tanpa login. Pengunjung dapat melihat dan mengubah database demo bersama, termasuk mencoba checkout, mengubah inventaris, serta mengelola produk. Gunakan hanya data fiktif dan hindari memasukkan informasi pribadi atau pembayaran nyata. File SQLite pada layanan Free tidak persisten dan dapat hilang saat restart, redeploy, atau siklus layanan. Jangan gunakan URL demo sebagai POS toko.

Jangan menambah replica atau menganggap file SQLite sementara sebagai database multi-kasir. Untuk POS operasional, perlu autentikasi dan database persisten dengan backup terjadwal.

## Netlify

Netlify tetap dapat menyajikan situs statis dari folder `site/`, tetapi tidak menjalankan entrypoint Streamlit ini. Jika ingin pengunjung membuka demo lewat situs Netlify, tambahkan tautan menuju URL Render setelah URL layanan tersedia. Runtime Smart Retail tetap berada di Render.
