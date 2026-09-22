# TASK 01 - Audit Dataset SIBI

Dataset tersedia di:
data/raw/Mono_Background

Lakukan audit dataset tanpa mengubah file asli.

Periksa:
1. Jumlah folder/kelas.
2. Nama seluruh kelas.
3. Identifikasi huruf A-Z yang tidak tersedia.
4. Hitung jumlah gambar setiap kelas.
5. Hitung total gambar.
6. Periksa format file.
7. Periksa resolusi gambar.
8. Periksa file corrupt.
9. Periksa imbalance antar kelas.
10. Buat laporan tabel lengkap.

Gunakan Python untuk memproses dataset.
Jangan memasukkan seluruh gambar ke context LLM.
Jangan melakukan training.

Simpan hasil:
reports/dataset_summary.csv
reports/DATASET_REPORT.md

Jika ditemukan bahwa J dan Z tidak tersedia, laporkan sebagai missing/dynamic classes.
Jangan membuat data sintetis sebagai pengganti J dan Z.