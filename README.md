# SIBI Real-Time Recognition

Aplikasi prototype penelitian untuk **pengenalan abjad jari SIBI (Sistem Isyarat
Bahasa Indonesia)** secara realtime melalui webcam, berbasis Streamlit +
streamlit-webrtc + MediaPipe HandLandmarker + MLPClassifier.

## Ringkasan

- **24 gesture statis** didukung: A–I dan K–Y.
- **J dan Z belum didukung** — keduanya gesture dinamis dan masih dalam
  pengembangan (model GRU terpisah, belum diintegrasikan ke aplikasi ini).
- **Model statis default: V2**; **Current** tersedia sebagai baseline/fallback
  (dapat dipilih di sidebar "Model statis").
- Aplikasi ini **prototype penelitian**, bukan pengganti penerjemah bahasa
  isyarat profesional.

## Teknologi

- Streamlit + streamlit-webrtc (webcam realtime via WebRTC)
- MediaPipe HandLandmarker (21 landmark tangan → 63 fitur)
- scikit-learn MLPClassifier
- OpenCV (headless), NumPy, pandas, joblib

Preprocessing: MediaPipe → 21 landmark (x, y, z) → wrist-relative normalization
→ scale normalization → 63 fitur → scaler + MLP (model terpilih).

Rejection: prediksi diterima hanya jika `confidence ≥ 0.85` **dan**
`margin (top1 − top2) ≥ 0.20`; selain itu "Tidak dikenali". Ada temporal
smoothing dan perlindungan stale prediction.

## Menjalankan lokal

```bash
pip install -r requirements.txt
python -m streamlit run app_streamlit.py
```

Kemudian buka **http://localhost:8501**.

### Kebutuhan kamera & izin browser

- Diperlukan webcam; pada browser **izinkan akses kamera** saat diminta.
- Klik **START** pada komponen kamera, pilih perangkat dengan **SELECT DEVICE**
  bila perlu.
- Mirror **ON** secara default (tampilan seperti cermin).

## Menggunakan aplikasi

1. Izinkan akses kamera.
2. Arahkan satu tangan ke kamera (seluruh jari terlihat).
3. Bentuk salah satu abjad SIBI.
4. Tahan posisi sampai gesture dikenali.
5. Tekan **Tambahkan huruf** untuk menyusun kata.

Kontrol lain: **Spasi**, **Hapus**, **Reset** (dengan konfirmasi).

## Deployment (Streamlit Community Cloud)

- Entry point: `app_streamlit.py`.
- `requirements.txt` = dependency **runtime** saja (tanpa TensorFlow/Keras).
- `packages.txt` = paket apt Linux (libgl1, libglib2.0-0) untuk MediaPipe/OpenCV.
- Aset model yang diperlukan ada di `models/` (joblib + `hand_landmarker.task`).
- WebRTC memakai STUN publik; untuk jaringan ketat mungkin perlu TURN
  (belum dikonfigurasi, tanpa secret di repo).

## Struktur penting

```
app_streamlit.py          # entrypoint Streamlit
realtime.py               # primitif preprocessing/inference (dipakai ulang)
models/                   # model statis (V2 + Current) + hand_landmarker.task
.streamlit/config.toml    # tema
requirements.txt          # runtime (deployment)
requirements-dev.txt      # training/eksperimen (lokal)
```

## Catatan penelitian

- Akurasi dari pengujian dataset dapat berbeda dari performa webcam nyata
  (pencahayaan, jarak, latar, orientasi tangan).
- Sebagian kelas statis sulit (mis. K, N, R, U, V, X) masih dievaluasi; V2 tidak
  diklaim lebih baik untuk semua huruf.
- Aplikasi tidak menyimpan video pengguna.
