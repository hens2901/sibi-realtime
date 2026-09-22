# TASK 02 - Ekstraksi Hand Landmark Dataset SIBI

Dataset sumber:
data/raw/Mono_Background

Dataset memiliki 24 kelas:
A-I dan K-Y.

Tujuan:
Menguji apakah MediaPipe Hands cocok digunakan untuk dataset ini
dan mengekstrak landmark tangan sebagai fitur untuk model ANN.

Kerjakan:

1. Gunakan MediaPipe Hands untuk membaca seluruh gambar dataset.
2. Jangan mengubah atau menghapus dataset asli.
3. Deteksi maksimal satu tangan pada setiap gambar.
4. Ambil 21 landmark tangan.
5. Gunakan koordinat x, y, z sehingga setiap gambar menghasilkan 63 fitur.
6. Normalisasi landmark agar tidak bergantung pada posisi tangan di gambar:
   - gunakan wrist sebagai titik referensi;
   - translasi landmark relatif terhadap wrist;
   - lakukan normalisasi skala berdasarkan ukuran tangan.
7. Simpan label kelas bersama fitur landmark.

Buat output:
data/processed/sibi_landmarks.csv

Format:
f1,f2,...,f63,label

Lakukan evaluasi proses ekstraksi:
- jumlah gambar total;
- jumlah gambar berhasil dideteksi;
- jumlah gambar gagal dideteksi;
- detection rate keseluruhan;
- detection rate setiap kelas.

Simpan file gambar yang gagal hanya sebagai daftar path, jangan copy file:
reports/landmark_failures.csv

Simpan statistik:
reports/landmark_detection_summary.csv
reports/LANDMARK_REPORT.md

Tambahkan pemeriksaan:
- missing value;
- NaN;
- jumlah fitur harus tepat 63;
- distribusi kelas setelah ekstraksi.

Jangan melakukan training model pada tahap ini.
Jangan membuat data sintetis.
Jangan melakukan augmentasi.
Jangan menggunakan CNN.

Jika MediaPipe belum terinstall, install dependency yang diperlukan.

Gunakan Python untuk memproses dataset.
Jangan memasukkan seluruh gambar ke context LLM.

Setelah selesai, laporkan apakah pendekatan MediaPipe landmark layak
digunakan berdasarkan detection rate.