# STREAMLIT CLOUD DEPLOYMENT CHECKLIST - SIBI Real-Time

Persiapan deploy ke Streamlit Community Cloud. **Tidak ada push/deploy** pada
task ini — hanya prepare & verify. Model tidak di-retrain; classifier dan
preprocessing tidak diubah; GRU J/Z tidak diintegrasikan.

## Checklist

- [x] Runtime paths relative (`Path(__file__).resolve().parent`; tidak ada `D:\`/`C:\` di `app_streamlit.py`/`realtime.py`)
- [x] Linux case-safe (nama file/modul konsisten huruf kecil)
- [x] Requirements minimal (runtime saja, dipin)
- [x] TensorFlow dikecualikan (tidak dipakai app; import TF di mediapipe bersifat *optional*)
- [x] OpenCV headless (`opencv-python-headless`) + `packages.txt` (libgl1, libglib2.0-0)
- [x] Model V2 tersedia (`sibi_mlp_augmented_v2` + scaler + encoder)
- [ ] ~~Baseline/Current tersedia~~ → **Current DIHAPUS dari runtime cloud** (V2-only);
  tetap disimpan di project penelitian utama `D:\AI-Agent\sibi-realtime`.
- [x] MediaPipe task tersedia (`models/hand_landmarker.task`)
- [x] STUN dikonfigurasi (publik, tanpa secret)
- [x] Tidak ada credential/secret di repo
- [x] Tidak ada file dataset/raw yang dibutuhkan runtime
- [x] Smoke test cloud lulus
- [x] Local Streamlit boot lulus
- [x] Siap untuk GitHub (struktur & .gitignore)
- [x] Siap untuk Streamlit Community Cloud

## 1. File yang perlu masuk GitHub

Wajib (runtime):
- `app_streamlit.py` (entrypoint)
- `realtime.py` (primitif preprocessing/inference, di-import app)
- `models/sibi_mlp_augmented.joblib`
- `models/scaler_augmented.joblib`
- `models/label_encoder_augmented.joblib`
- `models/sibi_mlp_augmented_v2.joblib`
- `models/scaler_augmented_v2.joblib`
- `models/label_encoder_augmented_v2.joblib`
- `models/hand_landmarker.task`
- `.streamlit/config.toml`
- `requirements.txt`
- `packages.txt`
- `README.md`
- `.gitignore`
- `scripts/smoke_test_cloud.py` (opsional, untuk verifikasi)

Tidak dibutuhkan runtime (boleh tidak di-commit / di-gitignore):
- `scripts/*` (training/eksperimen/audit), `hybrid_router.py`,
  `hybrid_realtime_test.py`, `dynamic_jz_inference.py`, `app_*.py` lain
- `requirements-dev.txt` (opsional)
- `reports/*` (kecuali checklist ini bila ingin disertakan)

## 2. File/folder yang TIDAK boleh masuk

- `data/raw/` (Mono_Background, J, Z — dataset & video)
- `data/processed/` (landmark, cache, dynamic_sequences)
- `data/training/`, `data/validation/`
- `reports/realtime_dynamic_debug/`, `reports/dynamic_previews/`,
  `reports/static_reference/` (debug/gambar besar)
- `models/dynamic_jz_*` (GRU & artefak dinamis — belum dipakai app)
- `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.ipynb_checkpoints/`
- `.env*`, `.streamlit/secrets.toml`
- file `*.log`/temporary

## 3. Final requirements (`requirements.txt`)

```
streamlit==1.64.0
streamlit-webrtc==0.78.1
av==17.1.0
mediapipe==1.0.1
numpy==2.4.6
pandas==3.0.6
scikit-learn==1.9.1
joblib==1.6.0
opencv-python-headless>=4.8,<6
```

`packages.txt`: `libgl1`, `libglib2.0-0`.
`requirements-dev.txt`: tambahan training (`tensorflow==2.21.0`, matplotlib, Pillow, scipy) — **tidak** untuk Cloud.

## 4. Runtime model size

| File | MB |
|---|--:|
| `models/sibi_mlp_augmented.joblib` | 0.421 |
| `models/scaler_augmented.joblib` | 0.003 |
| `models/label_encoder_augmented.joblib` | ~0.000 |
| `models/sibi_mlp_augmented_v2.joblib` | 0.422 |
| `models/scaler_augmented_v2.joblib` | 0.002 |
| `models/label_encoder_augmented_v2.joblib` | ~0.000 |
| `models/hand_landmarker.task` | 7.457 |
| **Total runtime** | **~8.3 MB** |

Kecil/wajar → commit biasa, **tanpa Git LFS**.
(Artefak non-runtime: `sibi_mlp.joblib` 0.421 MB, `dynamic_jz_gru.keras` dll —
tidak diperlukan app.)

## 5. STUN configuration

`realtime` app WebRTC memakai `rtc_configuration`:
`{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}` (publik, tanpa
secret). Cocok untuk deployment HTTPS remote umum.
Jika jaringan ketat membutuhkan TURN, tambahkan kredensial via
`st.secrets`/environment (JANGAN hardcode). Belum dikonfigurasi.

## 6. Hasil smoke test

- `scripts/smoke_test_cloud.py`: **lulus** (import app, path relative, aset ada,
  V2 & Current dimuat, scaler/encoder sesuai model, MediaPipe task, 63 fitur,
  probabilitas valid, runtime code tidak impor TF, requirements tanpa TF).
- `pip install --dry-run -r requirements.txt`: **resolusi sukses**.
- Streamlit boot lokal: health `ok`, halaman HTTP 200.
- `scripts/smoke_test_streamlit.py`: 40/40 (termasuk AppTest) — dari tahap
  sebelumnya (tidak berubah).

## 7. Issue yang ditemukan & diperbaiki

1. `requirements.txt` lama memuat dependency training (`matplotlib`, `Pillow`,
   `opencv-python`) dan tanpa `av`, tanpa pin → diganti runtime-only + pin;
   training dipindah ke `requirements-dev.txt`.
2. Belum ada `packages.txt` (libGL/libglib untuk MediaPipe/OpenCV di Linux) →
   ditambahkan.
3. Belum ada `.gitignore` (dataset/debug berisiko ter-commit) → ditambahkan.
4. Belum ada `README.md` → dibuat.
5. Pesan error model kurang cloud-friendly → "Model aplikasi tidak dapat
   dimuat." + startup validation aset (`missing_runtime_assets`) tanpa
   menampilkan traceback mentah.
6. TensorFlow: awalnya tampak diperlukan karena `import mediapipe` menarik TF di
   env lokal. Diverifikasi bahwa import TF di mediapipe bersifat **optional**
   (try/except) → **TF tidak dimasukkan** ke requirements.
7. Artefak dinamis (`models/dynamic_jz_*`) tidak diperlukan app → di-gitignore.

## 8. Catatan OpenCV

Aplikasi tidak memakai GUI (`cv2.imshow`), sehingga `opencv-python-headless`
dipin. Namun **mediapipe 1.0.1 wajib** menarik `opencv-contrib-python` (tidak
bisa dihindari). Keduanya menyediakan paket `cv2` pada versi yang sama
(5.0.0.93); aplikasi hanya memakai modul standar (tanpa GUI) sehingga tetap
aman. Alternatif tanpa duplikasi adalah menghapus `opencv-python-headless` dan
mengandalkan `opencv-contrib-python` dari mediapipe.

## 9. Linux compatibility audit

- `pathlib` dipakai; tidak ada path absolut Windows di runtime code.
- Nama file/modul konsisten huruf kecil; `import realtime` resolve dari root.
- OpenCV headless + `packages.txt` untuk libGL/libglib.
- MediaPipe task dimuat relatif (`models/hand_landmarker.task`).
- joblib/pickle: `scikit-learn==1.9.1` dipin agar cocok dengan artefak.
- Tidak ada dependency dev (TF/matplotlib eksplisit) yang bocor ke runtime.

## 10. Keterbatasan verifikasi

- **Clean-venv install penuh** belum dijalankan (hanya `pip --dry-run` yang
  sukses) — environment lokal sudah berisi semua dependency. Disarankan uji
  fresh venv/Streamlit Cloud sebelum publikasi.
- Webcam browser nyata tetap perlu diuji manual di Cloud (HTTPS + izin kamera).

## 11. Perintah deploy (untuk nanti, bukan dijalankan sekarang)

```bash
pip install -r requirements.txt
python -m streamlit run app_streamlit.py
# Streamlit Cloud: entrypoint app_streamlit.py, requirements.txt, packages.txt
```
