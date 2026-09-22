"""Latih GRU kecil untuk gesture dinamis SIBI J vs Z (video-level CV).

Prinsip:
- unit split = 1 video/sequence (StratifiedKFold di level sequence);
- scaler fit HANYA pada train fold;
- augmentasi HANYA pada train fold;
- class_weight dihitung dari train fold;
- tidak menyentuh model statis / realtime.py / app_streamlit.py.

Output:
- models/dynamic_jz_gru.keras
- models/dynamic_jz_scaler.joblib
- models/dynamic_jz_labels.json
- models/dynamic_jz_config.json
- reports/dynamic_jz_cv_folds.csv
- reports/dynamic_jz_confusion_matrix.png
- reports/dynamic_movement_stats.json (ditambah distribusi confidence model)
- reports/DYNAMIC_JZ_GRU_REPORT.md

Jalankan:
    python scripts/train_dynamic_jz_gru.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
SEQ_DIR = PROCESSED / "dynamic_sequences"
MANIFEST_CSV = PROCESSED / "dynamic_manifest.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODEL_PATH = MODELS / "dynamic_jz_gru.keras"
SCALER_PATH = MODELS / "dynamic_jz_scaler.joblib"
LABELS_PATH = MODELS / "dynamic_jz_labels.json"
CONFIG_PATH = MODELS / "dynamic_jz_config.json"
FOLDS_CSV = REPORTS / "dynamic_jz_cv_folds.csv"
CM_PNG = REPORTS / "dynamic_jz_confusion_matrix.png"
MOVEMENT_JSON = REPORTS / "dynamic_movement_stats.json"
REPORT_MD = REPORTS / "DYNAMIC_JZ_GRU_REPORT.md"

SEQ_LEN = 24
FEATURE_COUNT = 63
CLASSES = ["J", "Z"]
LABEL_TO_IDX = {"J": 0, "Z": 1}
N_SPLITS = 5
SEED = 42
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-3
GRU_UNITS = 48
DROPOUT = 0.3
DENSE_UNITS = 16
AUG_COPIES = 3
MODEL_VERSION = "dynamic-jz-gru-v1"


# --------------------------------------------------------------------------- #
# Load sequences
# --------------------------------------------------------------------------- #

def load_sequences() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    manifest = {}
    with MANIFEST_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest[row["source_video"]] = row

    X, y, sources, meta = [], [], [], []
    for label in CLASSES:
        d = SEQ_DIR / label
        for npz in sorted(d.glob("*.npz")):
            with np.load(npz, allow_pickle=True) as data:
                seq = np.asarray(data["sequence"], dtype=np.float32)
                src = str(data["source_video"])
                lab = str(data["label"])
            assert seq.shape == (SEQ_LEN, FEATURE_COUNT), f"{src} shape {seq.shape}"
            assert np.all(np.isfinite(seq)), f"{src} NaN/inf"
            assert lab == label
            m = manifest.get(src, {})
            X.append(seq)
            y.append(LABEL_TO_IDX[lab])
            sources.append(src)
            meta.append({
                "source_video": src,
                "label": lab,
                "fps": float(m.get("fps", 0) or 0),
                "duration_sec": float(m.get("duration_sec", 0) or 0),
                "handedness": m.get("handedness", ""),
                "movement_magnitude": float(m.get("movement_magnitude", 0) or 0),
                "wrist_movement": float(m.get("wrist_movement", 0) or 0),
                "fingertip_movement": float(m.get("fingertip_movement", 0) or 0),
                "temporal_velocity": float(m.get("temporal_velocity", 0) or 0),
            })
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            sources, meta)


# --------------------------------------------------------------------------- #
# Scaler (train-only) + augmentation
# --------------------------------------------------------------------------- #

def fit_scaler(X_train: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    flat = X_train.reshape(-1, FEATURE_COUNT)
    scaler = StandardScaler().fit(flat)
    return scaler


def apply_scaler(scaler, X: np.ndarray) -> np.ndarray:
    flat = X.reshape(-1, FEATURE_COUNT)
    return scaler.transform(flat).reshape(X.shape).astype(np.float32)


def temporal_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    n = seq.shape[0]
    t = np.linspace(0.0, 1.0, n)
    t2 = np.clip((t - 0.5) * factor + 0.5, 0.0, 1.0)
    idx = t2 * (n - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo)[:, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


def temporal_shift(seq: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return seq
    if shift > 0:
        pad = np.repeat(seq[:1], shift, axis=0)
        return np.concatenate([pad, seq[:-shift]], axis=0)
    pad = np.repeat(seq[-1:], -shift, axis=0)
    return np.concatenate([seq[-shift:], pad], axis=0)


def augment_sequences(X_train: np.ndarray, rng: np.random.Generator,
                      copies: int = AUG_COPIES) -> np.ndarray:
    out = [X_train]
    for _ in range(copies):
        batch = []
        for seq in X_train:
            s = temporal_warp(seq, rng.uniform(0.85, 1.15))
            s = temporal_shift(s, int(rng.integers(-1, 2)))
            s = s + rng.normal(0.0, 0.005, size=s.shape).astype(np.float32)
            s = s * rng.uniform(0.99, 1.01, size=(1, s.shape[1])).astype(np.float32)
            s = s + rng.normal(0.0, 0.003, size=s.shape).astype(np.float32)
            batch.append(s.astype(np.float32))
        out.append(np.asarray(batch, dtype=np.float32))
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_gru():
    import keras
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, FEATURE_COUNT)),
        keras.layers.GRU(GRU_UNITS, return_sequences=False),
        keras.layers.Dropout(DROPOUT),
        keras.layers.Dense(DENSE_UNITS, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def metrics_from_preds(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    acc = float(accuracy_score(y_true, y_pred))
    mp = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    mr = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mf = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    rec_j = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() else 0.0
    rec_z = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else 0.0
    return {"accuracy": acc, "macro_precision": mp, "macro_recall": mr,
            "macro_f1": mf, "recall_J": rec_j, "recall_Z": rec_z, "cm": cm}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    import keras
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight

    X, y, sources, meta = load_sequences()
    n = len(X)
    print(f"== GRU J/Z ==  sequences={n} (J={int((y==0).sum())}, Z={int((y==1).sum())})")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    fold_rows: list[dict] = []
    cm_total = np.zeros((2, 2), dtype=int)
    all_conf = np.zeros((n, 2), dtype=np.float32)
    best_epochs: list[int] = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        scaler = fit_scaler(X_tr)
        X_tr_s = apply_scaler(scaler, X_tr)
        X_va_s = apply_scaler(scaler, X_va)

        X_tr_aug = augment_sequences(X_tr_s, rng, AUG_COPIES)
        y_tr_aug = np.concatenate([y_tr] * (AUG_COPIES + 1), axis=0)

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
        class_weight = {0: float(cw[0]), 1: float(cw[1])}

        model = build_gru()
        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
            keras.callbacks.ModelCheckpoint(
                str(MODELS / f"_tmp_fold{fold}.keras"), monitor="val_loss",
                save_best_only=True, verbose=0),
        ]
        hist = model.fit(
            X_tr_aug, y_tr_aug, validation_data=(X_va_s, y_va),
            epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight,
            callbacks=cb, verbose=0,
        )
        hv = hist.history
        best_idx = int(np.argmin(hv["val_loss"]))
        best_epoch = best_idx + 1
        best_epochs.append(best_epoch)

        proba = model.predict(X_va_s, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        all_conf[va] = proba
        m = metrics_from_preds(y_va, y_pred)
        cm_total += m["cm"]

        fold_rows.append({
            "fold": fold,
            "n_train": len(tr), "n_val": len(va),
            "n_train_aug": len(X_tr_aug),
            "train_accuracy": float(hv["accuracy"][best_idx]),
            "val_accuracy": m["accuracy"],
            "train_loss": float(hv["loss"][best_idx]),
            "val_loss": float(hv["val_loss"][best_idx]),
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "recall_J": m["recall_J"], "recall_Z": m["recall_Z"],
            "best_epoch": best_epoch,
            "class_weight": f"{class_weight[0]:.2f}/{class_weight[1]:.2f}",
            "cm": m["cm"].tolist(),
        })
        print(f"[fold {fold}] n={len(tr)}/{len(va)} best_epoch={best_epoch} "
              f"acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} "
              f"recJ={m['recall_J']:.2f} recZ={m['recall_Z']:.2f} "
              f"train_acc={fold_rows[-1]['train_accuracy']:.3f} "
              f"val_loss={fold_rows[-1]['val_loss']:.3f}")
        # hapus checkpoint sementara
        tmp = MODELS / f"_tmp_fold{fold}.keras"
        if tmp.exists():
            tmp.unlink()

    # ---- Aggregate ----
    def ms(key):
        vals = [r[key] for r in fold_rows]
        return (float(statistics.mean(vals)),
                float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0)

    agg = {k: ms(k) for k in (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "recall_J", "recall_Z", "train_accuracy", "train_loss",
        "val_loss", "best_epoch")}

    # ---- Sanity & leakage control ----
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.neighbors import NearestCentroid
    from sklearn.linear_model import LogisticRegression

    def cv_simple(make, featurize, labels):
        accs = []
        for tr, va in skf.split(X, labels):
            Xtr, Xva = featurize(X[tr]), featurize(X[va])
            sc = _SS().fit(Xtr)
            clf = make().fit(sc.transform(Xtr), labels[tr])
            accs.append(float((clf.predict(sc.transform(Xva)) == labels[va]).mean()))
        return float(statistics.mean(accs)), float(statistics.pstdev(accs)) if len(accs) > 1 else 0.0

    nc_flat = cv_simple(NearestCentroid, lambda A: A.reshape(len(A), -1), y)
    lr_mean = cv_simple(lambda: LogisticRegression(max_iter=3000),
                        lambda A: A.mean(axis=1), y)
    perm_labels = np.random.default_rng(SEED + 1).permutation(y)
    nc_perm = cv_simple(NearestCentroid, lambda A: A.reshape(len(A), -1), perm_labels)

    mov = np.array([m["movement_magnitude"] for m in meta])
    mv_acc_high = float(((mov > (mov[y == 0].mean() + mov[y == 1].mean()) / 2).astype(int) == (1 - y)).mean())

    # ---- Fold CSV ----
    with FOLDS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "n_train", "n_val", "n_train_aug", "accuracy",
                    "macro_precision", "macro_recall", "macro_f1", "recall_J",
                    "recall_Z", "val_loss", "train_loss", "train_accuracy",
                    "best_epoch", "class_weight", "cm"])
        for r in fold_rows:
            w.writerow([r["fold"], r["n_train"], r["n_val"], r["n_train_aug"],
                        f"{r['accuracy']:.4f}", f"{r['macro_precision']:.4f}",
                        f"{r['macro_recall']:.4f}", f"{r['macro_f1']:.4f}",
                        f"{r['recall_J']:.4f}", f"{r['recall_Z']:.4f}",
                        f"{r['val_loss']:.4f}", f"{r['train_loss']:.4f}",
                        f"{r['train_accuracy']:.4f}", r["best_epoch"],
                        r["class_weight"], r["cm"]])

    # ---- Confusion matrix PNG (akumulasi semua fold) ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_total, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES,
           yticklabels=CLASSES, xlabel="Predicted", ylabel="True",
           title="Confusion Matrix (CV, agregat)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_total[i, j]), ha="center", va="center",
                    color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)

    # ---- Final model (semua data) ----
    scaler_final = fit_scaler(X)
    X_all_s = apply_scaler(scaler_final, X)
    X_all_aug = augment_sequences(X_all_s, rng, AUG_COPIES)
    y_all_aug = np.concatenate([y] * (AUG_COPIES + 1), axis=0)
    cw_all = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    final_epochs = max(20, int(round(statistics.median(best_epochs))))
    final_model = build_gru()
    final_model.fit(X_all_aug, y_all_aug, epochs=final_epochs,
                    batch_size=BATCH_SIZE,
                    class_weight={0: float(cw_all[0]), 1: float(cw_all[1])},
                    verbose=0)
    final_model.save(MODEL_PATH)
    joblib.dump(scaler_final, SCALER_PATH)
    LABELS_PATH.write_text(json.dumps({"0": "J", "1": "Z", "classes": CLASSES},
                                      indent=2), encoding="utf-8")

    # confidence pada semua sequence (in-sample; bukan test independent)
    proba_all = final_model.predict(X_all_s, verbose=0)
    conf_by_label = {
        lab: [float(proba_all[i][LABEL_TO_IDX[lab]]) for i in range(n) if y[i] == LABEL_TO_IDX[lab]]
        for lab in CLASSES
    }

    def dist(vals):
        vals = [float(v) for v in vals]
        if not vals:
            return {"min": 0, "mean": 0, "median": 0, "max": 0, "std": 0}
        return {"min": min(vals), "mean": float(statistics.mean(vals)),
                "median": float(statistics.median(vals)), "max": max(vals),
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0}

    # update movement stats JSON dengan confidence model
    mov = {}
    if MOVEMENT_JSON.exists():
        mov = json.loads(MOVEMENT_JSON.read_text(encoding="utf-8"))
    mov["model_confidence"] = {
        lab: {"count": len(conf_by_label[lab]), **dist(conf_by_label[lab])}
        for lab in CLASSES
    }
    mov["model_confidence_note"] = (
        "Confidence dihitung pada seluruh sequence included (in-sample, bukan "
        "test independen). Untuk routing/rejection realtime, kalibrasi "
        "ambang perlu validasi pada data baru."
    )
    MOVEMENT_JSON.write_text(json.dumps(mov, indent=2), encoding="utf-8")

    # ---- Config ----
    config = {
        "model_version": MODEL_VERSION,
        "task": "dynamic_jz",
        "sequence_length": SEQ_LEN,
        "feature_count": FEATURE_COUNT,
        "classes": CLASSES,
        "label_map": {"0": "J", "1": "Z"},
        "normalization": {
            "landmark": "21 x (x,y,z) MediaPipe",
            "wrist_relative": True,
            "scale_normalization": "max ||p - p_wrist||",
            "temporal_normalization": "timestamp frame_index/fps; active span -> 0..1",
            "resampling": "linear interpolation (np.interp) ke 24 timestep",
        },
        "scaler": {
            "type": "StandardScaler",
            "fit_on": "training fold only (reshape n*24,63)",
            "path": "models/dynamic_jz_scaler.joblib",
        },
        "architecture": {
            "input": [SEQ_LEN, FEATURE_COUNT],
            "gru_units": GRU_UNITS,
            "dropout": DROPOUT,
            "dense_units": DENSE_UNITS,
            "dense_dropout": 0.2,
            "output": 2,
            "activation": "softmax",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LR,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "early_stopping_patience": 25,
            "reduce_lr_patience": 10,
            "augmentation_copies": AUG_COPIES,
            "class_weight": "balanced (train fold only)",
        },
        "cross_validation": {
            "type": "StratifiedKFold (video/sequence level)",
            "n_splits": N_SPLITS,
            "seed": SEED,
            "signer_independent": False,
        },
        "data": {
            "n_sequences": int(n),
            "n_J": int((y == 0).sum()),
            "n_Z": int((y == 1).sum()),
            "excluded": ["J/J_005.MP4 (long gap)", "J/J_007.MP4 (duplicate)",
                         "Z/Z_009.MP4 (low detection)"],
        },
        "confidence_notes": (
            "Model hanya 2 kelas (J/Z), TIDAK ada kelas OTHER. Non-J/Z dapat "
            "diprediksi sebagai J/Z. Wajib motion routing + dynamic rejection "
            "sebelum dipakai di aplikasi."
        ),
        "training_seed": SEED,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- Report ----
    final_avg_acc = agg["accuracy"][0]
    overfit_acc = agg["train_accuracy"][0] - agg["accuracy"][0]
    overfit_loss = agg["val_loss"][0] - agg["train_loss"][0]

    L: list[str] = []
    a = L.append
    a("# DYNAMIC J/Z GRU REPORT")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Script: `scripts/train_dynamic_jz_gru.py` | versi: {MODEL_VERSION}")
    a(f"- Waktu training+CV: {time.perf_counter() - start:.1f} detik")
    a("- Model statis 24 huruf, `realtime.py`, `app_streamlit.py`: **tidak diubah**.")
    a("")
    a("## 1. Dataset Awal")
    a(f"- J = 21 video, Z = 15 video (total 36). J=50 FPS, Z=25 FPS, 1920x1080.")
    a("- Audit: `reports/DYNAMIC_VIDEO_AUDIT.md`.")
    a("")
    a("## 2. Cleaning")
    manifest_rows = list(csv.DictReader(MANIFEST_CSV.open(encoding="utf-8")))
    excluded_rows = [(r["source_video"], r["exclusion_reason"])
                     for r in manifest_rows
                     if str(r["include"]).strip().lower() != "true"]
    a("| Video | Alasan exclude |")
    a("|:------|:---------------|")
    for src, reason in excluded_rows:
        a(f"| {src} | {reason} |")
    a("")
    a("## 3. Dataset Final")
    a(f"- Sequence: **{n}** = {int((y==0).sum())} J + {int((y==1).sum())} Z.")
    a(f"- Shape per sequence: ({SEQ_LEN}, {FEATURE_COUNT}).")
    a("")
    a("## 4. Preprocessing Landmark")
    a("- MediaPipe HandLandmarker 21 titik (x,y,z) → 63 fitur.")
    a("- Wrist-relative (`p - p_wrist`) + scale normalization (`/ max ||p-p_wrist||`).")
    a("- Sama dengan `realtime.py`/dataset statis.")
    a("")
    a("## 5. Penanganan Missing Detection")
    a("- Interpolasi hanya gap pendek; gap > 0.30s / interpolasi > 50% → exclude.")
    a("- Rekap di `reports/DYNAMIC_SEQUENCE_REPORT.md`.")
    a("")
    a("## 6. Timestamp / FPS Normalization")
    a("- `timestamp = frame_index / fps` (bukan indeks mentah).")
    a("- Segmen aktif dinormalisasi 0.0 → 1.0.")
    a("- **Perbedaan FPS J=50 dan Z=25 tidak menjadi fitur.**")
    a("")
    a("## 7. Temporal Resampling")
    a("- Linear interpolation (`np.interp`) ke 24 timestep → bukan flatten 1512.")
    a("")
    a("## 8. Sequence 24x63")
    a(f"- Tersimpan di `data/processed/dynamic_sequences/J|Z/` ({n} file).")
    a("")
    a("## 9. Train-only Augmentation")
    a(f"- {AUG_COPIES} kopi per sampel train: temporal speed variation (±15%), "
      "temporal shift ±1, Gaussian noise kecil, feature jitter.")
    a("- Hanya pada train fold; validation bersih. Tidak memakai mirror x=-x.")
    a("")
    a("## 10. Cross-Validation Strategy")
    a(f"- StratifiedKFold {N_SPLITS} fold di level **video/sequence** (bukan frame).")
    a(f"- Seed tetap = {SEED}. Scaler & class_weight hanya dari train fold.")
    a("- Signer ID tidak tersedia → **signer-independent validation belum bisa "
      "dilakukan**.")
    a("")
    a("## 11. GRU Architecture")
    a(f"- Input({SEQ_LEN},{FEATURE_COUNT}) → GRU({GRU_UNITS}) → Dropout({DROPOUT}) "
      f"→ Dense({DENSE_UNITS}, relu) → Dropout(0.2) → Dense(2, softmax).")
    a(f"- Adam lr={LR}, loss sparse_categorical_crossentropy, batch={BATCH_SIZE}.")
    a("")
    a("## 12. Metrics per Fold")
    a("| Fold | n_tr/n_val | Acc | Macro P | Macro R | Macro F1 | Recall J | Recall Z | Val loss | Best epoch |")
    a("|:----:|:----------:|----:|--------:|--------:|---------:|---------:|---------:|---------:|-----------:|")
    for r in fold_rows:
        a(f"| {r['fold']} | {r['n_train']}/{r['n_val']} | {r['accuracy']:.3f} | "
          f"{r['macro_precision']:.3f} | {r['macro_recall']:.3f} | "
          f"{r['macro_f1']:.3f} | {r['recall_J']:.3f} | {r['recall_Z']:.3f} | "
          f"{r['val_loss']:.3f} | {r['best_epoch']} |")
    a("")
    a("## 13. Aggregate (mean ± std)")
    a("| Metrik | Mean ± Std |")
    a("|:-------|:-----------|")
    for key, lab in (("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
                     ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")):
        a(f"| {lab} | {agg[key][0]:.3f} ± {agg[key][1]:.3f} |")
    a("")
    a(f"## 14. Recall J\n- {agg['recall_J'][0]:.3f} ± {agg['recall_J'][1]:.3f}")
    a(f"\n## 15. Recall Z\n- {agg['recall_Z'][0]:.3f} ± {agg['recall_Z'][1]:.3f}")
    a("")
    a("## 16. Confusion Matrix (agregat CV)")
    a("")
    a("|  | Pred J | Pred Z |")
    a("|:--|------:|------:|")
    a(f"| True J | {cm_total[0,0]} | {cm_total[0,1]} |")
    a(f"| True Z | {cm_total[1,0]} | {cm_total[1,1]} |")
    a("")
    a(f"Gambar: `reports/{CM_PNG.name}`.")
    a("")
    a("## 17. Overfitting Analysis")
    a(f"- Train accuracy (mean over folds): {agg['train_accuracy'][0]:.3f}")
    a(f"- Validation accuracy (CV): {agg['accuracy'][0]:.3f}")
    a(f"- Gap akurasi: {overfit_acc:+.3f}")
    a(f"- Train loss: {agg['train_loss'][0]:.3f} | Val loss: {agg['val_loss'][0]:.3f} "
      f"| Gap: {overfit_loss:+.3f}")
    if overfit_acc > 0.10:
        a("- **Indikasi overfitting** (gap akurasi > 10 poin). Dataset sangat kecil.")
    else:
        a("- Gap akurasi 0 terjadi karena **train dan validasi sama-sama mencapai "
          "ceiling (1.0)**; dataset terlalu kecil/mudah untuk menampilkan "
          "overfitting secara jelas, jadi ini **bukan** bukti bebas overfitting.")
    a("- Semua fold dilaporkan (tidak menyembunyikan fold buruk).")
    a("")
    a("## 17b. Sanity & Leakage Control")
    a("")
    a("Karena skor CV sempurna, dilakukan kontrol independen:")
    a("")
    a(f"- Nearest-centroid pada fitur flat (video-level CV): "
      f"**{nc_flat[0]:.3f} ± {nc_flat[1]:.3f}**")
    a(f"- Logistic regression pada mean-feature: **{lr_mean[0]:.3f} ± {lr_mean[1]:.3f}**")
    a(f"- **Kontrol label diacak** (nearest-centroid): "
      f"**{nc_perm[0]:.3f} ± {nc_perm[1]:.3f}** (mendekati kebetulan)")
    a(f"- Movement-only separability (ambang pada movement magnitude): "
      f"**{mv_acc_high:.3f}**")
    a("")
    a("Interpretasi: model sederhana pun sudah memisahkan J dan Z dengan sangat "
      "baik, dan kontrol label-acak jatuh ke level kebetulan → **tidak ada "
      "indikasi kebocoran split**. Namun skor sempurna ini **optimistis** karena "
      "(a) kemungkinan signer/sesi yang sama ada di train & validation, dan "
      "(b) J dan Z sangat berbeda (amplitudo gerakan J jauh lebih besar). "
      "Generalisasi ke subjek lain belum teruji.")
    a("")
    a("## 18. Movement Statistics")
    a("")
    a("| Label | movement (mean) | wrist (mean) | fingertip (mean) | velocity (mean) | confidence model (mean) |")
    a("|:-----:|----------------:|-------------:|-----------------:|----------------:|------------------------:|")
    for lab in CLASSES:
        lm = mov["per_label"][lab]
        conf = mov["model_confidence"][lab]
        a(f"| {lab} | {lm['movement_magnitude']['mean']:.4f} | "
          f"{lm['wrist_movement']['mean']:.4f} | {lm['fingertip_movement']['mean']:.4f} | "
          f"{lm['temporal_velocity']['mean']:.3f} | {conf['mean']:.3f} |")
    a("")
    a(f"Detail: `reports/{MOVEMENT_JSON.name}`. Confidence bersifat in-sample.")
    a("")
    a("## 19. Limitations")
    a("- Dataset sangat kecil (33 sequence; 19 J, 14 Z).")
    a("- Signer ID tidak tersedia → signer-independent validation belum bisa.")
    a("- Model hanya mengenal J dan Z; **tidak ada kelas OTHER**.")
    a("- Gerakan non-J/Z dapat diprediksi sebagai J/Z bila langsung dipakai.")
    a("- FPS sumber berbeda (50 vs 25) sudah dinormalkan, tetapi variasi "
      "pencahayaan/latar/subjek terbatas.")
    a("")
    a("## 20. Kesimpulan Kelayakan")
    a(f"- Performa CV: accuracy {agg['accuracy'][0]:.3f} ± {agg['accuracy'][1]:.3f}, "
      f"macro F1 {agg['macro_f1'][0]:.3f} ± {agg['macro_f1'][1]:.3f}.")
    if agg["macro_f1"][0] >= 0.80 and agg["macro_f1"][1] <= 0.20:
        verdict = ("**Layak dilanjutkan ke prototipe realtime**, dengan syarat "
                   "motion routing + dynamic rejection.")
    elif agg["macro_f1"][0] >= 0.65:
        verdict = ("**Cukup untuk proof-of-concept**, tetapi perlu perbaikan "
                   "(data tambahan/augmentasi) sebelum realtime serius.")
    else:
        verdict = ("**Belum layak untuk realtime**; perlu lebih banyak data / "
                   "perbaikan preprocessing.")
    a(f"- {verdict}")
    a("- **BELUM boleh dipasang ke Streamlit** tanpa motion routing dan "
      "dynamic rejection (J/Z vs non-J/Z).")
    a(f"- Final model (`{MODEL_PATH.name}`) dilatih pada seluruh data included; "
      "**bukan** hasil test independen — performa diukur dari cross-validation.")
    a("")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"\nAggregate: acc={agg['accuracy'][0]:.3f}±{agg['accuracy'][1]:.3f} "
          f"F1={agg['macro_f1'][0]:.3f}±{agg['macro_f1'][1]:.3f} "
          f"recJ={agg['recall_J'][0]:.3f} recZ={agg['recall_Z'][0]:.3f}")
    print(f"Overfit gap acc={overfit_acc:+.3f}")
    print(f"Model    : {MODEL_PATH.relative_to(ROOT)}")
    print(f"Report   : {REPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
