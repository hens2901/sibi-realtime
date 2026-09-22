# TASK 04 - Real-Time SIBI Scanner

Gunakan model:
- models/sibi_mlp.joblib
- models/scaler.joblib
- models/label_encoder.joblib
- models/hand_landmarker.task

Tujuan:
Membuat aplikasi webcam real-time untuk mengenali gesture SIBI statis.

Model hanya mendukung 24 kelas:
A-I dan K-Y.
J dan Z belum didukung.

Buat file realtime.py.

Fungsi:
1. Buka webcam menggunakan OpenCV.
2. Deteksi tangan menggunakan MediaPipe.
3. Ambil 21 landmark tangan.
4. Gunakan preprocessing yang sama dengan saat training:
   - landmark relatif terhadap wrist;
   - normalisasi skala;
   - urutan fitur harus sama.
5. Hasilkan tepat 63 fitur.
6. Terapkan scaler.
7. Jalankan model ANN.
8. Decode kelas dengan label encoder.

Tampilkan:
- landmark tangan;
- bounding box;
- huruf prediksi;
- confidence;
- FPS.

Gunakan confidence threshold 0.70.
Jika di bawah threshold tampilkan "Tidak dikenali".

Gunakan temporal smoothing dari beberapa frame terakhir agar prediksi stabil.

Kontrol:
- Q = keluar
- R = reset teks
- SPACE = tambahkan huruf stabil ke hasil ejaan
- BACKSPACE = hapus karakter terakhir

Tampilkan hasil ejaan di bagian atas frame.

Jangan training ulang model.
Jangan mengubah file model.
Jangan membuat J/Z sintetis.

Setelah implementasi:
- validasi model/scaler/encoder dapat dimuat;
- validasi input selalu 63 fitur;
- lakukan smoke test;
- perbaiki error.

Buat laporan:
reports/REALTIME_REPORT.md