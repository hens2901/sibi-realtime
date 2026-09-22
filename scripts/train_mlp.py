"""Training ANN (MLPClassifier) untuk klasifikasi SIBI statis 24 kelas.

- Input: data/processed/sibi_landmarks.csv (63 fitur + label)
- Split stratified, StandardScaler fit hanya pada train (anti leakage)
- Evaluasi: accuracy, macro precision/recall/F1, classification report, confusion matrix
- 5-fold Stratified Cross Validation
- Simpan model, scaler, label encoder + laporan

Output:
- models/sibi_mlp.joblib
- models/scaler.joblib
- models/label_encoder.joblib
- reports/classification_report.csv
- reports/confusion_matrix.png
- reports/TRAINING_REPORT.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "sibi_landmarks.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MLP_PATH = MODELS_DIR / "sibi_mlp.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
REPORT_CSV = REPORTS_DIR / "classification_report.csv"
CM_PNG = REPORTS_DIR / "confusion_matrix.png"
REPORT_MD = REPORTS_DIR / "TRAINING_REPORT.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
FEATURE_COLS_PREFIX = "f"


def load_and_validate() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c != "label"]

    assert len(feature_cols) == 63, f"Jumlah fitur harus 63, ditemukan {len(feature_cols)}"
    assert "label" in df.columns, "Kolom label tidak ditemukan"
    assert df[feature_cols].isna().sum().sum() == 0, "Ditemukan NaN pada fitur"
    assert df["label"].isna().sum() == 0, "Ditemukan NaN pada label"
    assert df["label"].nunique() == 24, f"Jumlah kelas harus 24, ditemukan {df['label'].nunique()}"

    X = df[feature_cols].astype(np.float64)
    y = df["label"].astype(str)
    return X, y


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def save_confusion_matrix(cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - SIBI MLP (24 kelas)",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cm[i, j] > thresh else "black",
                )
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_validate()
    class_counts = y.value_counts().sort_index()

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    labels = list(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    macro_prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, test_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)

    report_dict = classification_report(
        y_test, test_pred, labels=np.arange(len(labels)), target_names=labels,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_CSV, encoding="utf-8")

    cm = confusion_matrix(y_test, test_pred, labels=np.arange(len(labels)))
    save_confusion_matrix(cm, labels)

    cv_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", build_model()),
        ]
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(cv_pipeline, X, y_enc, cv=cv, scoring="accuracy", n_jobs=1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    per_class = report_df.loc[labels]
    lowest_recall_class = per_class["recall"].idxmin()
    lowest_recall_value = float(per_class.loc[lowest_recall_class, "recall"])

    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    if cm_off.sum() > 0:
        flat = int(np.argmax(cm_off))
        true_idx, pred_idx = np.unravel_index(flat, cm_off.shape)
        confused_true = labels[true_idx]
        confused_pred = labels[pred_idx]
        confused_count = int(cm_off[true_idx, pred_idx])
    else:
        confused_true = confused_pred = "-"
        confused_count = 0

    joblib.dump(model, MLP_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    reload_model = joblib.load(MLP_PATH)
    reload_scaler = joblib.load(SCALER_PATH)
    reload_encoder = joblib.load(ENCODER_PATH)
    reload_sample = reload_scaler.transform(X_test.iloc[:5])
    reload_pred = reload_encoder.inverse_transform(reload_model.predict(reload_sample))
    reload_ok = bool(
        len(reload_pred) == 5
        and all(p in labels for p in reload_pred)
        and reload_model.predict(reload_scaler.transform(X_test)).shape[0] == len(X_test)
    )

    gap = train_acc - test_acc
    if gap > 0.10:
        overfit_status = "Indikasi overfitting (gap > 10%)"
    elif gap > 0.05:
        overfit_status = "Sedikit indikasi overfitting (gap 5-10%)"
    else:
        overfit_status = "Tidak ada indikasi overfitting signifikan (gap <= 5%)"

    lines: list[str] = []
    a = lines.append
    a("# TRAINING REPORT - ANN SIBI (MLPClassifier)")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Data: `{FEATURES_CSV.relative_to(ROOT).as_posix()}`")
    a(f"- Total sampel: **{len(X)}**, fitur: **63**, kelas: **{len(labels)}**")
    a("- Arsitektur: 63 -> 128 -> 64 -> 24 (ReLU, Adam, early stopping)")
    a(f"- Random state: {RANDOM_STATE}, test size: {TEST_SIZE}, CV: {N_SPLITS}-fold stratified")
    a("")
    a("## 1. Validasi Data")
    a("")
    a(f"- NaN pada fitur: **0**")
    a(f"- Jumlah fitur: **63**")
    a(f"- Jumlah kelas valid: **{len(labels)}**")
    a(f"- Train/test split: **{len(X_train)}** / **{len(X_test)}**")
    a("")
    a("### Jumlah Sampel per Kelas")
    a("")
    a("| Kelas | Jumlah |")
    a("|:-----:|-------:|")
    for name, cnt in class_counts.items():
        a(f"| {name} | {cnt} |")
    a(f"| **TOTAL** | **{int(class_counts.sum())}** |")
    a("")
    a("## 2. Metrik Utama")
    a("")
    a("| Metrik | Nilai |")
    a("|:-------|------:|")
    a(f"| Train accuracy | {train_acc:.4f} |")
    a(f"| Test accuracy | {test_acc:.4f} |")
    a(f"| Macro precision | {macro_prec:.4f} |")
    a(f"| Macro recall | {macro_rec:.4f} |")
    a(f"| Macro F1 | {macro_f1:.4f} |")
    a(f"| CV accuracy mean | {cv_mean:.4f} |")
    a(f"| CV accuracy std | {cv_std:.4f} |")
    a("")
    a(f"- Skor CV per fold: {', '.join(f'{s:.4f}' for s in cv_scores)}")
    a("")
    a("## 3. Analisis Kesalahan")
    a("")
    a(f"- Kelas dengan recall terendah: **{lowest_recall_class}** (recall {lowest_recall_value:.4f})")
    a(f"- Pasangan paling sering tertukar: **{confused_true} -> {confused_pred}** ({confused_count} kali)")
    a("")
    a("## 4. Pemeriksaan Overfitting")
    a("")
    a(f"- Train accuracy: {train_acc:.4f}")
    a(f"- Test accuracy: {test_acc:.4f}")
    a(f"- Gap (train - test): {gap:.4f}")
    a(f"- Status: **{overfit_status}**")
    a("")
    a("## 5. Uji Loading Ulang Model")
    a("")
    a(f"- Model/scaler/encoder berhasil dimuat ulang: **{'YA' if reload_ok else 'TIDAK'}**")
    a(f"- Contoh prediksi (5 sampel test): {', '.join(map(str, reload_pred))}")
    a("")
    a("## 6. File Output")
    a("")
    a(f"- `{MLP_PATH.relative_to(ROOT).as_posix()}`")
    a(f"- `{SCALER_PATH.relative_to(ROOT).as_posix()}`")
    a(f"- `{ENCODER_PATH.relative_to(ROOT).as_posix()}`")
    a(f"- `{REPORT_CSV.relative_to(ROOT).as_posix()}`")
    a(f"- `{CM_PNG.relative_to(ROOT).as_posix()}`")
    a("")
    a("> Model ini hanya untuk **24 gesture statis** (A-I, K-Y). "
      "Model TIDAK mendukung J dan Z (kelas dinamis, tidak tersedia di dataset).")
    a("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Train accuracy   : {train_acc:.4f}")
    print(f"Test accuracy    : {test_acc:.4f}")
    print(f"Macro precision  : {macro_prec:.4f}")
    print(f"Macro recall     : {macro_rec:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"CV mean +/- std  : {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"Lowest recall    : {lowest_recall_class} ({lowest_recall_value:.4f})")
    print(f"Most confused    : {confused_true} -> {confused_pred} ({confused_count})")
    print(f"Overfit gap      : {gap:.4f} ({overfit_status})")
    print(f"Reload OK        : {reload_ok}")
    print(f"Model            : {MLP_PATH.relative_to(ROOT)}")
    print(f"Report           : {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
