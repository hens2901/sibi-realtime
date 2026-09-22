"""Eksperimen mirror augmentation (image-level horizontal flip) untuk SIBI.

Dasar: reports/HANDEDNESS_REPORT.md (dataset 96% satu chirality; model tidak
invarian terhadap handedness).

Prinsip anti-leakage:
- Split stratified train/test direproduksi persis dari eksperimen baseline
  (random_state=42, test_size=0.2, stratify) via data/processed/sibi_landmarks.csv,
  lalu dipetakan ke path gambar.
- Augmentasi (flip citra + re-deteksi MediaPipe) HANYA pada TRAINING SET.
- TEST SET original tidak disentuh; mirrored test hanya untuk evaluasi robustness.

Output:
- models/sibi_mlp_augmented.joblib
- models/scaler_augmented.joblib
- models/label_encoder_augmented.joblib
- reports/HANDEDNESS_AUGMENTATION_REPORT.md
- reports/augmentation_split.csv
- reports/aug_train_files.txt / reports/aug_test_files.txt
- reports/aug_metrics.csv / reports/aug_per_class.csv
- reports/aug_cm_*.png
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402

DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

FEATURES_CSV = PROCESSED_DIR / "sibi_landmarks.csv"
CACHE_NPZ = PROCESSED_DIR / "handedness_cache.npz"

MLP_AUG = MODELS_DIR / "sibi_mlp_augmented.joblib"
SCALER_AUG = MODELS_DIR / "scaler_augmented.joblib"
ENCODER_AUG = MODELS_DIR / "label_encoder_augmented.joblib"

SPLIT_CSV = REPORTS_DIR / "augmentation_split.csv"
TRAIN_LIST = REPORTS_DIR / "aug_train_files.txt"
TEST_LIST = REPORTS_DIR / "aug_test_files.txt"
METRICS_CSV = REPORTS_DIR / "aug_metrics.csv"
PER_CLASS_CSV = REPORTS_DIR / "aug_per_class.csv"
REPORT_MD = REPORTS_DIR / "HANDEDNESS_AUGMENTATION_REPORT.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = rt.CONFIDENCE_THRESHOLD
MARGIN = rt.MARGIN_THRESHOLD


# --------------------------------------------------------------------------- #
# Dataset & split
# --------------------------------------------------------------------------- #

def class_dirs() -> list[Path]:
    return sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)


def images_of(cdir: Path) -> list[Path]:
    return sorted([f for f in cdir.iterdir() if f.is_file()])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def detect_norm(landmarker, rgb: np.ndarray):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_image)
    if not res.hand_landmarks:
        return None, None
    hand = res.hand_landmarks[0]
    hd = "Unknown"
    if res.handedness and res.handedness[0]:
        hd = str(res.handedness[0][0].category_name)
    return rt.normalize_landmarks(hand), hd


def build_original_map(landmarker) -> tuple[list[str], list[str], np.ndarray]:
    """Path, class, handedness untuk semua gambar terdeteksi (urutan = CSV)."""
    paths, classes, handed = [], [], []
    for cdir in class_dirs():
        for img in images_of(cdir):
            _, hd = detect_norm(landmarker, load_rgb(img))
            if hd is None:
                continue
            paths.append(img.relative_to(ROOT).as_posix())
            classes.append(cdir.name)
            handed.append(hd)
    return paths, classes, handed


def get_paths_classes(landmarker) -> tuple[list[str], list[str], list[str]]:
    """Gunakan cache audit bila cocok dengan CSV; jika tidak, deteksi ulang."""
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c != "label"]
    if CACHE_NPZ.exists():
        d = np.load(CACHE_NPZ, allow_pickle=True)
        paths = [str(p) for p in d["paths"]]
        classes = [str(c) for c in d["classes"]]
        handed = [str(h) for h in d["handed"]]
        feats = d["features"]
        if len(paths) == len(df) and np.allclose(feats, df[feature_cols].to_numpy(), atol=1e-6):
            print(f"[cache] memakai {CACHE_NPZ.name} ({len(paths)} baris, cocok dengan CSV)")
            return paths, classes, handed
        print("[cache] cache tidak cocok dengan CSV -> deteksi ulang")
    paths, classes, handed = build_original_map(landmarker)
    return paths, classes, handed


def build_model() -> MLPClassifier:
    """Arsitektur & hyperparameter IDENTIK dengan baseline (scripts/train_mlp.py)."""
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


# --------------------------------------------------------------------------- #
# Evaluasi (dengan rejection mechanism)
# --------------------------------------------------------------------------- #

def evaluate(model, scaler, encoder, X_df: pd.DataFrame, y_true: list[str],
             feature_cols: list[str]) -> dict:
    X_df = X_df[feature_cols]
    probs = model.predict_proba(scaler.transform(X_df))
    preds = encoder.inverse_transform(np.argmax(probs, axis=1))

    accepted = np.zeros(len(y_true), dtype=bool)
    margins = np.zeros(len(y_true))
    for i in range(len(y_true)):
        t = rt.top2_from_probs(probs[i], list(encoder.classes_))
        margins[i] = t.margin
        accepted[i] = rt.passes_rejection(t.top1_prob, t.margin, THRESHOLD, MARGIN)

    y_true_arr = np.array(y_true)
    acc = accuracy_score(y_true_arr, preds)
    result = {
        "n": len(y_true),
        "accuracy": float(acc),
        "macro_precision": float(precision_score(y_true_arr, preds, average="macro",
                                                 zero_division=0, labels=list(encoder.classes_))),
        "macro_recall": float(recall_score(y_true_arr, preds, average="macro",
                                           zero_division=0, labels=list(encoder.classes_))),
        "macro_f1": float(f1_score(y_true_arr, preds, average="macro",
                                   zero_division=0, labels=list(encoder.classes_))),
        "rejection_rate": float(1.0 - accepted.mean()) if len(y_true) else 0.0,
        "coverage": float(accepted.mean()) if len(y_true) else 0.0,
        "accepted_accuracy": float((preds[accepted] == y_true_arr[accepted]).mean())
        if accepted.any() else 0.0,
        "preds": preds,
        "accepted": accepted,
        "margins": margins,
    }
    report = classification_report(
        y_true_arr, preds, labels=list(encoder.classes_),
        target_names=list(encoder.classes_), output_dict=True, zero_division=0,
    )
    result["per_class"] = pd.DataFrame(report).transpose()
    return result


def save_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, ylabel="True", xlabel="Predicted",
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    thresh = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # ---- 1-2. Split stratified (reproduksi baseline) ----
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c != "label"]
    y_all = df["label"].astype(str).to_numpy()
    y_enc_all = LabelEncoder().fit_transform(y_all)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y_enc_all, random_state=RANDOM_STATE
    )
    print(f"split: train={len(train_idx)} test={len(test_idx)} (baseline random_state=42)")

    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    try:
        paths, classes, handed = get_paths_classes(landmarker)
        assert len(paths) == len(df), (
            f"Jumlah gambar terdeteksi ({len(paths)}) != baris CSV ({len(df)}). "
            "Hapus data/processed/handedness_cache.npz lalu jalankan ulang."
        )
        train_paths = [paths[i] for i in train_idx]
        test_paths = [paths[i] for i in test_idx]
        train_labels = [classes[i] for i in train_idx]
        test_labels = [classes[i] for i in test_idx]
        train_handed = [handed[i] for i in train_idx]
        test_handed = [handed[i] for i in test_idx]
        train_orig = df.iloc[train_idx][feature_cols].to_numpy(dtype=np.float64)
        test_orig = df.iloc[test_idx][feature_cols].to_numpy(dtype=np.float64)

        # ---- 4c-d. Flip citra + re-deteksi MediaPipe (TRAIN ONLY untuk training) ----
        print("flip + re-detect: training set ...")
        train_flip, train_flip_labels, train_flip_fail = [], [], []
        for p, lab in zip(train_paths, train_labels):
            rgb = load_rgb(ROOT / p)
            f, _ = detect_norm(landmarker, rgb[:, ::-1])
            if f is None:
                train_flip_fail.append(p)
            else:
                train_flip.append(f)
                train_flip_labels.append(lab)
        print(f"  train flip ok={len(train_flip)} gagal={len(train_flip_fail)}")

        # ---- 8. Mirrored test (evaluasi robustness, BUKAN training) ----
        print("flip + re-detect: mirrored test set ...")
        test_mir, test_mir_handed, test_mir_labels, test_mir_fail = [], [], [], []
        for p, lab in zip(test_paths, test_labels):
            rgb = load_rgb(ROOT / p)
            f, hd = detect_norm(landmarker, rgb[:, ::-1])
            if f is None:
                test_mir_fail.append(p)
            else:
                test_mir.append(f)
                test_mir_handed.append(hd)
                test_mir_labels.append(lab)
    finally:
        landmarker.close()

    # ---- 5. Gabungkan original + flip (training saja) ----
    X_train_aug = np.vstack([train_orig,
                             np.array(train_flip, dtype=np.float64)])
    y_train_aug = np.array(train_labels + train_flip_labels)
    assert len(X_train_aug) == len(train_orig) + len(train_flip)

    # ---- 6-7. Scaler fit hanya pada training augmented ----
    df_train_aug = pd.DataFrame(X_train_aug, columns=feature_cols)
    scaler_aug = StandardScaler().fit(df_train_aug)
    encoder_aug = LabelEncoder().fit(y_train_aug)
    labels = [str(c) for c in encoder_aug.classes_]
    model_aug = build_model()
    model_aug.fit(scaler_aug.transform(df_train_aug), encoder_aug.transform(y_train_aug))
    print(f"augmented training: {len(X_train_aug)} sampel, {len(labels)} kelas")

    # ---- 10. Simpan artefak baru (tanpa menimpa model lama) ----
    joblib.dump(model_aug, MLP_AUG)
    joblib.dump(scaler_aug, SCALER_AUG)
    joblib.dump(encoder_aug, ENCODER_AUG)

    # ---- 11-12. Evaluasi baseline & augmented ----
    model_base, scaler_base, encoder_base, labels_base = rt.load_artifacts("baseline")
    labels_base = list(labels_base)

    df_test_orig = pd.DataFrame(test_orig, columns=feature_cols)
    df_test_mir = pd.DataFrame(np.array(test_mir, dtype=np.float64), columns=feature_cols)

    results = {
        "baseline/original": evaluate(model_base, scaler_base, encoder_base,
                                      df_test_orig, test_labels, feature_cols),
        "baseline/mirrored": evaluate(model_base, scaler_base, encoder_base,
                                      df_test_mir, test_mir_labels, feature_cols),
        "augmented/original": evaluate(model_aug, scaler_aug, encoder_aug,
                                       df_test_orig, test_labels, feature_cols),
        "augmented/mirrored": evaluate(model_aug, scaler_aug, encoder_aug,
                                       df_test_mir, test_mir_labels, feature_cols),
    }

    # confusion matrices
    label_order = labels
    cm_paths = {}
    for key, res in results.items():
        cm = confusion_matrix(np.array(test_labels if "original" in key else test_mir_labels),
                              res["preds"], labels=label_order)
        fname = "aug_cm_" + key.replace("/", "_") + ".png"
        p = REPORTS_DIR / fname
        save_cm(cm, label_order, p, f"Confusion Matrix - {key}")
        cm_paths[key] = fname
        if key == "augmented/original":
            res["per_class"].to_csv(PER_CLASS_CSV, encoding="utf-8")

    # metrics CSV
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "test_set", "n", "accuracy", "macro_precision",
                    "macro_recall", "macro_f1", "rejection_rate", "coverage",
                    "accepted_accuracy"])
        for key, res in results.items():
            m, t = key.split("/")
            w.writerow([m, t, res["n"], f"{res['accuracy']:.4f}",
                        f"{res['macro_precision']:.4f}", f"{res['macro_recall']:.4f}",
                        f"{res['macro_f1']:.4f}", f"{res['rejection_rate']:.4f}",
                        f"{res['coverage']:.4f}", f"{res['accepted_accuracy']:.4f}"])

    # ---- 3. Simpan daftar file split ----
    with TRAIN_LIST.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(train_paths) + "\n")
    with TEST_LIST.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(test_paths) + "\n")
    with SPLIT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "class", "path", "handedness"])
        for p, c, h in zip(train_paths, train_labels, train_handed):
            w.writerow(["train", c, p, h])
        for p, c, h in zip(test_paths, test_labels, test_handed):
            w.writerow(["test", c, p, h])

    # ---- Left/Right performance (original test) ----
    handed_arr = np.array(test_handed)
    lr = {}
    for key in ("baseline/original", "augmented/original"):
        preds = results[key]["preds"]
        row = {}
        for grp in ("Left", "Right"):
            m = handed_arr == grp
            row[grp] = (int(m.sum()),
                        float((preds[m] == np.array(test_labels)[m]).mean()) if m.any() else 0.0)
        lr[key] = row

    # ---- 17. Validasi loading ulang ----
    m2 = joblib.load(MLP_AUG)
    s2 = joblib.load(SCALER_AUG)
    e2 = joblib.load(ENCODER_AUG)
    sample = df_test_orig.iloc[:5]
    reload_pred = e2.inverse_transform(m2.predict(s2.transform(sample)))
    reload_ok = bool(len(reload_pred) == 5
                     and getattr(m2, "n_features_in_", None) == 63
                     and len(e2.classes_) == 24)
    rt_aug_ok = False
    try:
        _, _, _, labels_rt = rt.load_artifacts("augmented")
        rt_aug_ok = len(labels_rt) == 24
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] realtime load augmented gagal: {exc}")

    elapsed = time.perf_counter() - start

    # ---- 16. Laporan ----
    def pct(x):
        return f"{x * 100:.2f}%"

    L: list[str] = []
    a = L.append
    a("# HANDEDNESS AUGMENTATION REPORT - SIBI")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Dasar: `reports/HANDEDNESS_REPORT.md`")
    a(f"- Script: `scripts/train_mlp_augmented.py`")
    a(f"- Waktu total: {elapsed:.1f} detik")
    a("")
    a("## 1. Tujuan & Metode")
    a("")
    a("Memperbaiki bias handedness dengan **image-level horizontal flip augmentation**:")
    a("citra training dibalik horizontal secara *in-memory*, lalu MediaPipe **di-detect "
      "ulang** (bukan sekadar negasi koordinat x). Model baru dibandingkan dengan model "
      "baseline pada test set yang sama.")
    a("")
    a("## 2. Split (Anti-Leakage)")
    a("")
    a(f"- Split direproduksi dari baseline: `train_test_split(test_size={TEST_SIZE}, "
      f"stratify=y, random_state={RANDOM_STATE})` di atas `{FEATURES_CSV.name}` "
      "lalu dipetakan ke path gambar.")
    a(f"- Train: **{len(train_idx)}** gambar | Test: **{len(test_idx)}** gambar.")
    a(f"- Augmentasi flip hanya dari **train**; test original tidak disentuh.")
    a(f"- Daftar file: `{TRAIN_LIST.name}`, `{TEST_LIST.name}`, `{SPLIT_CSV.name}`.")
    a(f"- Mirrored test dibuat dari {len(test_paths)} gambar test untuk evaluasi "
      "robustness saja (tidak untuk training).")
    a("")
    a("## 3. Augmentasi (Flip Citra + Re-deteksi)")
    a("")
    a("| Tahap | Jumlah | Gagal deteksi MediaPipe |")
    a("|:------|-------:|------------------------:|")
    a(f"| Train original | {len(train_orig)} | - |")
    a(f"| Train flipped | {len(train_flip)} | {len(train_flip_fail)} |")
    a(f"| Test original | {len(test_orig)} | - |")
    a(f"| Test mirrored (eval) | {len(test_mir)} | {len(test_mir_fail)} |")
    a("")
    a(f"- Total sampel training augmented: **{len(X_train_aug)}** "
      f"({len(train_orig)} original + {len(train_flip)} flip).")
    a("")
    a("## 4. Hasil Evaluasi")
    a("")
    a("| Model | Test set | N | Accuracy | Macro Precision | Macro Recall | Macro F1 |")
    a("|:------|:--------:|--:|---------:|----------------:|-------------:|---------:|")
    for key, res in results.items():
        m, t = key.split("/")
        a(f"| {m} | {t} | {res['n']} | {pct(res['accuracy'])} | "
          f"{pct(res['macro_precision'])} | {pct(res['macro_recall'])} | "
          f"{pct(res['macro_f1'])} |")
    a("")
    a("### Dengan Rejection Mechanism (threshold "
      f"{THRESHOLD:.2f}, margin {MARGIN:.2f})")
    a("")
    a("| Model | Test set | Rejection rate | Coverage | Accuracy pada yang diterima |")
    a("|:------|:--------:|---------------:|---------:|----------------------------:|")
    for key, res in results.items():
        m, t = key.split("/")
        a(f"| {m} | {t} | {pct(res['rejection_rate'])} | {pct(res['coverage'])} | "
          f"{pct(res['accepted_accuracy'])} |")
    a("")
    a("## 5. Performa Left/Right (Test Original)")
    a("")
    a("| Model | Grup | N | Akurasi |")
    a("|:------|:-----|--:|--------:|")
    for key, row in lr.items():
        for grp in ("Left", "Right"):
            n_grp, acc_grp = row[grp]
            a(f"| {key.split('/')[0]} | {grp} | {n_grp} | {pct(acc_grp)} |")
    a("")
    a(f"> Distribusi handedness di test: Left={int((handed_arr == 'Left').sum())}, "
      f"Right={int((handed_arr == 'Right').sum())}. Grup Right kecil sehingga angkanya "
      "perlu dibaca hati-hati.")
    a("")
    a("## 6. Confusion Matrix")
    a("")
    for key, fname in cm_paths.items():
        a(f"- `reports/{fname}` - {key}")
    a("")
    a("Per-class lengkap (model augmented pada original test): "
      f"`reports/{PER_CLASS_CSV.name}`; metrik ringkas: `reports/{METRICS_CSV.name}`.")
    a("")
    a("## 7. Validasi Loading Model")
    a("")
    a(f"- Reload `{MLP_AUG.name}` + scaler + encoder: **{'OK' if reload_ok else 'GAGAL'}**")
    a(f"- `realtime.load_artifacts('augmented')`: **{'OK' if rt_aug_ok else 'GAGAL'}**")
    a(f"- Model lama TIDAK ditimpa (`{rt.MLP_PATH.name}` tetap ada).")
    a("")
    a("## 8. Pemilihan Model di Realtime")
    a("")
    a("```bash")
    a("python realtime.py --model baseline    # default (belum diubah)")
    a("python realtime.py --model augmented")
    a("python realtime.py --check --model augmented")
    a("```")
    a("")
    a("## 9. Kesimpulan")
    a("")
    base_o = results["baseline/original"]["accuracy"]
    aug_o = results["augmented/original"]["accuracy"]
    base_m = results["baseline/mirrored"]["accuracy"]
    aug_m = results["augmented/mirrored"]["accuracy"]
    base_right = lr["baseline/original"]["Right"]
    aug_right = lr["augmented/original"]["Right"]
    a(f"- Original test: baseline {pct(base_o)} vs augmented {pct(aug_o)} "
      f"({(aug_o - base_o) * 100:+.2f} poin).")
    a(f"- Mirrored test (robustness): baseline {pct(base_m)} vs augmented {pct(aug_m)} "
      f"({(aug_m - base_m) * 100:+.2f} poin).")
    a(f"- Test Right/minoritas (n={aug_right[0]}): baseline {pct(base_right[1])} vs "
      f"augmented {pct(aug_right[1])}.")
    a(f"- Augmented **jauh lebih robust** pada mirrored test: "
      f"{'YA' if aug_m > base_m + 0.05 else 'BELUM'}.")
    if aug_o >= base_o - 0.02 and aug_m > base_m + 0.05:
        a("- Augmented unggul jelas pada robustness handedness dengan penurunan tipis "
          "pada original test. **Namun default tetap baseline** sampai divalidasi "
          "dengan webcam nyata (test mirror bersifat sintetis).")
    else:
        a("- **Default tetap baseline.** Augmented belum terbukti lebih baik secara "
          "keseluruhan; jangan diganti sebagai default.")
    a("")
    a("## 10. Catatan Penting")
    a("")
    a("- Mirrored test adalah **test sintetis** (flip citra). Ini **bukan** bukti "
      "invariansi handedness sesungguhnya. Hasil ini hanya menunjukkan robustness "
      "terhadap flip citra.")
    a("- **Validasi akhir wajib memakai webcam nyata dengan tangan kiri dan kanan** "
      "pada kondisi pencahayaan nyata, karena distribusi landmark webcam dapat berbeda "
      "dari gambar dataset.")
    a("- Model tetap hanya mengenali 24 gesture statis (A-I, K-Y); J dan Z tidak didukung.")
    a("")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"baseline  orig={base_o:.3f} mirror={base_m:.3f}")
    print(f"augmented orig={aug_o:.3f} mirror={aug_m:.3f}")
    print(f"Report : {REPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
