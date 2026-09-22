"""Kandidat model statis V2 dengan augmentasi geometrik fokus kelas sulit.

Train-only augmentation:
- horizontal flip (proxy x-negation)
- rotasi kecil (±8°) untuk kelas sulit K,N,R,U,V,X
- gaussian noise ringan

Tidak menimpa model current. Output:
- models/sibi_mlp_augmented_v2.joblib
- models/scaler_augmented_v2.joblib
- models/label_encoder_augmented_v2.joblib
- reports/static_v2_comparison.csv

Evaluasi current vs V2 pada original test + mirrored (proxy) test, per-class
recall 24 kelas, memastikan 18 kelas lain tidak rusak.

Jalankan:
    python scripts/train_static_v2_hardaug.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realtime as rt  # noqa: E402

CSV_FEATURES = ROOT / "data" / "processed" / "sibi_landmarks.csv"
MODELS = ROOT / "models"
OUT_CMP = ROOT / "reports" / "static_v2_comparison.csv"
HARD = {"K", "N", "R", "U", "V", "X"}
SEED = 42
TEST_SIZE = 0.2


def mirror_x(X):
    X = np.array(X, dtype=np.float64, copy=True)
    X[..., 0::3] *= -1.0
    return X


def rotate_xy(feats, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    out = np.array(feats, dtype=np.float64, copy=True)
    pts = out.reshape(21, 3)
    x, y = pts[:, 0].copy(), pts[:, 1].copy()
    pts[:, 0] = x * c - y * s
    pts[:, 1] = x * s + y * c
    return out.reshape(-1)


def augment(X, y, rng):
    copies = []
    for feats, lab in zip(X, y):
        n_extra = 4 if lab in HARD else 2
        base = [feats, mirror_x(feats)]
        copies.append(feats)
        copies.append(mirror_x(feats))
        for _ in range(n_extra):
            f = rotate_xy(feats, rng.uniform(-8, 8))
            f = f + rng.normal(0, 0.004, size=f.shape)
            copies.append(f)
    return np.asarray(copies, dtype=np.float64)


def evaluate(model, scaler, labels, X, y, threshold=0.85, margin=0.20):
    probs = model.predict_proba(scaler.transform(X))
    pred = np.array([labels[i] for i in np.argmax(probs, axis=1)])
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, average="macro"))
    recall = {c: float((pred[y == c] == c).mean()) if (y == c).any() else 0.0
              for c in labels}
    return acc, f1, recall, pred


def main() -> int:
    df = pd.read_csv(CSV_FEATURES)
    fcols = [c for c in df.columns if c != "label"]
    X_all = df[fcols].to_numpy(dtype=np.float64)
    y_all = df["label"].astype(str).to_numpy()
    labels = sorted(set(y_all))
    tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                              stratify=y_all, random_state=SEED)
    X_tr, y_tr = X_all[tr], y_all[tr]
    X_te, y_te = X_all[te], y_all[te]

    rng = np.random.default_rng(SEED)
    X_aug = augment(X_tr, y_tr, rng)
    y_aug = np.asarray([lab for lab in y_tr
                        for _ in range(1 + 1 + (4 if lab in HARD else 2))])

    scaler = StandardScaler().fit(X_aug)
    enc = LabelEncoder().fit(y_aug)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          solver="adam", alpha=1e-4, batch_size=32,
                          learning_rate_init=1e-3, max_iter=500,
                          early_stopping=True, n_iter_no_change=15,
                          validation_fraction=0.1, random_state=SEED)
    model.fit(scaler.transform(X_aug), enc.transform(y_aug))
    joblib.dump(model, MODELS / "sibi_mlp_augmented_v2.joblib")
    joblib.dump(scaler, MODELS / "scaler_augmented_v2.joblib")
    joblib.dump(enc, MODELS / "label_encoder_augmented_v2.joblib")

    # current
    cur = rt.load_artifacts("augmented")
    rows = []
    for name, mdl, scl, labs in (("current", cur[0], cur[1], list(cur[3])),
                                 ("v2", model, scaler, list(enc.classes_))):
        for tag, Xt in (("original", X_te), ("mirrored_proxy", mirror_x(X_te))):
            acc, f1, rec, _ = evaluate(mdl, scl, labs, Xt, y_te)
            row = {"model": name, "test": tag, "accuracy": round(acc, 4),
                   "macro_f1": round(f1, 4)}
            for c in HARD:
                row[f"recall_{c}"] = round(rec.get(c, 0.0), 3)
            rows.append(row)
            print(f"{name:8s} {tag:14s} acc={acc:.3f} macroF1={f1:.3f} "
                  + " ".join(f"{c}={rec.get(c,0):.2f}" for c in sorted(HARD)))

    cols = list(rows[0].keys())
    with OUT_CMP.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # cek 18 kelas lain (original)
    _, _, rec_cur, _ = evaluate(cur[0], cur[1], list(cur[3]), X_te, y_te)
    _, _, rec_v2, _ = evaluate(model, scaler, list(enc.classes_), X_te, y_te)
    others = [c for c in labels if c not in HARD]
    drop = {c: round(rec_v2[c] - rec_cur[c], 3) for c in others}
    worst = min(drop.items(), key=lambda x: x[1])
    print(f"\n18 kelas lain: penurunan terbesar {worst[0]} {worst[1]:+.3f}")
    print(f"CSV: {OUT_CMP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
