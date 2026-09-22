# TASK 03 - Training ANN Klasifikasi SIBI

Dataset fitur:
data/processed/sibi_landmarks.csv

Dataset berisi 63 fitur landmark tangan dan satu kolom label.
Jumlah kelas adalah 24:
A-I dan K-Y.
Huruf J dan Z tidak tersedia dan jangan dibuat secara sintetis.

Tujuan:
Membangun Artificial Neural Network yang ringan untuk
pengenalan gesture SIBI secara real-time.

Gunakan:
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn JANGAN digunakan
- joblib

Gunakan MLPClassifier dari scikit-learn sebagai ANN.

Tahapan:

1. Baca sibi_landmarks.csv.
2. Validasi:
   - tidak ada NaN;
   - jumlah fitur = 63;
   - label valid;
   - tampilkan jumlah sampel setiap kelas.

3. Pisahkan data menggunakan stratified train/test split.
   Gunakan random_state tetap agar eksperimen reproducible.

4. Gunakan StandardScaler.
   Penting:
   scaler hanya boleh fit pada training data untuk mencegah data leakage.

5. Encode label jika diperlukan.

6. Buat ANN awal menggunakan MLPClassifier.

Gunakan arsitektur yang wajar untuk dataset kecil ini, misalnya:
63 input
-> hidden layer 128
-> hidden layer 64
-> output 24 kelas.

Gunakan:
- activation ReLU
- Adam
- early stopping
- random_state tetap.

7. Jangan membuat model terlalu besar karena dataset hanya sekitar 1.424 sampel.

8. Evaluasi menggunakan:
- accuracy
- precision macro
- recall macro
- F1-score macro
- classification report
- confusion matrix

9. Tambahkan 5-fold Stratified Cross Validation untuk mendapatkan
estimasi performa yang lebih stabil.

10. Laporkan:
- train accuracy
- test accuracy
- macro precision
- macro recall
- macro F1
- mean dan standard deviation cross-validation
- kelas dengan recall terendah
- kelas yang paling sering tertukar.

11. Simpan:
models/sibi_mlp.joblib
models/scaler.joblib
models/label_encoder.joblib jika digunakan.

12. Simpan laporan:
reports/classification_report.csv
reports/confusion_matrix.png
reports/TRAINING_REPORT.md

13. Periksa kemungkinan overfitting dengan membandingkan
akurasi training dan testing.

14. Jangan mengubah dataset raw.
15. Jangan membuat data sintetis.
16. Jangan memasukkan seluruh dataset ke context LLM.
Gunakan Python untuk training dan hanya baca ringkasan hasil.

17. Setelah training selesai, lakukan test loading ulang terhadap
model dan scaler untuk memastikan file model dapat digunakan kembali.

PENTING:
Model ini baru untuk 24 gesture statis.
Jangan mengklaim model mendukung J atau Z.