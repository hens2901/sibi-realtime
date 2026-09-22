"""Derivasi threshold untuk hybrid router (motion + dynamic rejection).

Menghasilkan:
- reports/dynamic_router_thresholds.json  (dipakai hybrid_router.py)
- reports/dynamic_router_thresholds.csv
- models/dynamic_jz_prototypes.npz        (prototipe kelas di ruang scaled)

Metode:
- Out-of-fold (OOF) probability dari protokol CV IDENTIK training
  (StratifiedKFold 5, seed 42) untuk distribusi confidence.
- Distribusi movement (raw, dari manifest) + movement fitur (normalized).
- Prototipe kelas (mean sequence) + jarak leave-one-out untuk rejection
  berbasis jarak.

Catatan: script ini menjalankan protokol CV yang sama untuk KEPERLUAN ANALISIS
threshold. Artefak model final (models/dynamic_jz_gru.keras) TIDAK diubah.

Jalankan:
    python scripts/derive_dynamic_thresholds.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_dynamic_jz_gru as T  # noqa: E402
import joblib  # noqa: E402

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT_JSON = REPORTS / "dynamic_router_thresholds.json"
OUT_CSV = REPORTS / "dynamic_router_thresholds.csv"
PROTO_NPZ = MODELS / "dynamic_jz_prototypes.npz"

CONF_CANDIDATES = (0.70, 0.80, 0.85, 0.90, 0.95)
PCTS = (0, 5, 10, 25, 50, 75, 90, 95, 100)


def pct_stats(vals) -> dict:
    vals = np.asarray([float(v) for v in vals], dtype=np.float64)
    if vals.size == 0:
        return {f"p{p}": 0.0 for p in PCTS}
    q = np.percentile(vals, PCTS)
    return {f"p{p}": float(q[i]) for i, p in enumerate(PCTS)}


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    import keras
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight

    X, y, sources, meta = T.load_sequences()
    n = len(X)
    print(f"sequences={n} (J={int((y==0).sum())}, Z={int((y==1).sum())})")

    # wrist/fingertip/velocity tidak ada di manifest -> ambil dari npz.
    for i, src in enumerate(sources):
        npz = ROOT / "data" / "processed" / "dynamic_sequences" / src.split("/")[0] / (
            Path(src).stem + ".npz")
        with np.load(npz, allow_pickle=True) as d:
            meta[i]["movement_magnitude"] = float(d["movement_magnitude"])
            meta[i]["wrist_movement"] = float(d["wrist_movement"])
            meta[i]["fingertip_movement"] = float(d["fingertip_movement"])
            meta[i]["temporal_velocity"] = float(d["temporal_velocity"])

    # ---------------- OOF probabilities (protokol CV identik) ----------------
    skf = StratifiedKFold(n_splits=T.N_SPLITS, shuffle=True, random_state=T.SEED)
    rng = np.random.default_rng(T.SEED)
    oof = np.zeros((n, 2), dtype=np.float64)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        scaler = T.fit_scaler(X[tr])
        Xtr = T.apply_scaler(scaler, X[tr])
        Xva = T.apply_scaler(scaler, X[va])
        Xtr_aug = T.augment_sequences(Xtr, rng, T.AUG_COPIES)
        ytr_aug = np.concatenate([y[tr]] * (T.AUG_COPIES + 1))
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
        model = T.build_gru()
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                            restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=10, min_lr=1e-5)]
        model.fit(Xtr_aug, ytr_aug, validation_data=(Xva, y[va]),
                  epochs=T.EPOCHS, batch_size=T.BATCH_SIZE,
                  class_weight={0: float(cw[0]), 1: float(cw[1])},
                  callbacks=cb, verbose=0)
        oof[va] = model.predict(Xva, verbose=0)
        print(f"  fold {fold} done (n_val={len(va)})")

    preds = np.argmax(oof, axis=1)
    correct = preds == y
    conf_all = oof[np.arange(n), preds]      # confidence pada prediksi
    conf_true = oof[np.arange(n), y]         # confidence pada label benar
    oof_acc = float(correct.mean())

    # ---------------- Threshold sweep (coverage vs accuracy) ----------------
    sweep = []
    for thr in CONF_CANDIDATES:
        acc_mask = conf_all >= thr
        coverage = float(acc_mask.mean())
        if acc_mask.any():
            acc_on = float((preds[acc_mask] == y[acc_mask]).mean())
        else:
            acc_on = 0.0
        sweep.append({"threshold": thr, "coverage": coverage,
                      "accuracy_accepted": acc_on,
                      "accepted": int(acc_mask.sum()), "total": n})

    # ---------------- Movement distributions ----------------
    raw_move = {"J": [], "Z": []}
    raw_wrist = {"J": [], "Z": []}
    raw_tip = {"J": [], "Z": []}
    raw_vel = {"J": [], "Z": []}
    norm_move = {"J": [], "Z": []}
    norm_vel = {"J": [], "Z": []}
    for i in range(n):
        lab = "J" if y[i] == 0 else "Z"
        m = meta[i]
        raw_move[lab].append(m["movement_magnitude"])
        raw_wrist[lab].append(m["wrist_movement"])
        raw_tip[lab].append(m["fingertip_movement"])
        raw_vel[lab].append(m["temporal_velocity"])
        d = np.linalg.norm(np.diff(X[i], axis=0), axis=1)  # (23,) norm per step
        norm_move[lab].append(float(d.sum()))
        dur = m["duration_sec"] or 1.0
        norm_vel[lab].append(float(d.sum() / dur))

    # ---------------- Prototipe + jarak leave-one-out (scaled space) --------
    scaler_all = T.fit_scaler(X)
    Xs = T.apply_scaler(scaler_all, X)
    protos = {lab: Xs[y == (0 if lab == "J" else 1)].mean(axis=0)
              for lab in ("J", "Z")}
    # LOO distance: jarak ke prototipe kelas lain (atau kelas sendiri tanpa dirinya)
    dist = {"J": [], "Z": []}
    for i in range(n):
        lab = "J" if y[i] == 0 else "Z"
        dmin = None
        for cl in ("J", "Z"):
            idx = np.where(y == (0 if cl == "J" else 1))[0]
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            proto = Xs[idx].mean(axis=0)
            d = float(np.mean(np.linalg.norm(Xs[i] - proto, axis=1)))
            dmin = d if dmin is None else min(dmin, d)
        dist[lab].append(dmin)

    # ---------------- Temporal variation (per-sequence std of step motion) --
    variation = {"J": [], "Z": []}
    for i in range(n):
        lab = "J" if y[i] == 0 else "Z"
        d = np.linalg.norm(np.diff(X[i], axis=0), axis=1)
        variation[lab].append(float(np.std(d)))

    # ---------------- Pilih threshold awal ----------------
    # Confidence: pilih yang coverage tinggi dengan accuracy_accepted ~ 1.0
    best_conf = CONF_CANDIDATES[0]
    for row in sweep:
        if row["coverage"] >= 0.60 and row["accuracy_accepted"] >= 0.99:
            best_conf = row["threshold"]
    # Distance: pakai persentil 95 jarak LOO semua kelas
    all_dist = dist["J"] + dist["Z"]
    distance_max = float(np.percentile(all_dist, 95)) * 1.1 if all_dist else 1e9
    # Movement: Z p05 sebagai batas bawah kandidat dynamic
    z_move = np.asarray(raw_move["Z"])
    move_min = float(np.percentile(z_move, 5)) if z_move.size else 0.05
    static_max = float(move_min * 0.5)
    move_max = float(np.percentile(np.asarray(raw_move["J"] + raw_move["Z"]), 100) * 1.5)
    # Variasi minimal (menolak sequence nyaris datar)
    var_min = float(np.percentile(np.asarray(variation["J"] + variation["Z"]), 5) * 0.5)

    result = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_sequences": n,
        "oof_accuracy": oof_acc,
        "confidence": {
            "oof_predicted_confidence": pct_stats(conf_all),
            "oof_true_label_confidence": pct_stats(conf_true),
            "sweep": sweep,
        },
        "movement": {
            "raw_movement_magnitude": {lab: pct_stats(raw_move[lab]) for lab in ("J", "Z")},
            "raw_wrist": {lab: pct_stats(raw_wrist[lab]) for lab in ("J", "Z")},
            "raw_fingertip": {lab: pct_stats(raw_tip[lab]) for lab in ("J", "Z")},
            "raw_velocity": {lab: pct_stats(raw_vel[lab]) for lab in ("J", "Z")},
            "norm_feature_movement": {lab: pct_stats(norm_move[lab]) for lab in ("J", "Z")},
            "norm_feature_velocity": {lab: pct_stats(norm_vel[lab]) for lab in ("J", "Z")},
        },
        "distance": {
            "metric": "mean per-timestep L2 to class mean (scaled space), LOO",
            "J": pct_stats(dist["J"]),
            "Z": pct_stats(dist["Z"]),
        },
        "temporal_variation": {lab: pct_stats(variation[lab]) for lab in ("J", "Z")},
        "chosen": {
            "static_motion_max": static_max,
            "dynamic_motion_min": move_min,
            "dynamic_motion_max": move_max,
            "dynamic_conf_min": best_conf,
            "dynamic_margin_min": 0.20,
            "distance_max": distance_max,
            "variation_min": var_min,
            "min_peak_confidence": best_conf,
        },
        "notes": (
            "Threshold awal diturunkan dari distribusi training + OOF. "
            "Confidence OOF saturasi (~1.0) sehingga softmax saja TIDAK cukup "
            "untuk OOD rejection; dipakai juga jarak prototipe + rentang "
            "movement + variasi temporal."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(REPORTS / "dynamic_jz_oof.npz", oof=oof,
                        y=y, sources=np.array(sources))

    # CSV ringkas sweep + movement
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value"])
        w.writerow(["oof", "accuracy", f"{oof_acc:.4f}"])
        for row in sweep:
            w.writerow(["confidence_sweep", f"thr={row['threshold']:.2f}",
                        f"coverage={row['coverage']:.3f};acc={row['accuracy_accepted']:.3f}"])
        for lab in ("J", "Z"):
            for k, v in result["movement"]["raw_movement_magnitude"][lab].items():
                w.writerow(["movement", f"{lab}.{k}", f"{v:.4f}"])
            for k, v in result["distance"][lab].items():
                w.writerow(["distance", f"{lab}.{k}", f"{v:.4f}"])

    np.savez_compressed(PROTO_NPZ, J=protos["J"].astype(np.float32),
                        Z=protos["Z"].astype(np.float32),
                        metric=np.array("mean per-timestep L2 (scaled space)"))

    print(f"OOF accuracy       : {oof_acc:.3f}")
    print(f"OOF confidence p5/p50/p95: "
          f"{np.percentile(conf_all,5):.4f}/{np.percentile(conf_all,50):.4f}/{np.percentile(conf_all,95):.4f}")
    print(f"chosen conf_min    : {best_conf:.2f}")
    print(f"chosen motion min/max: {move_min:.4f} / {move_max:.4f} (static_max {static_max:.4f})")
    print(f"chosen distance_max: {distance_max:.4f}")
    print(f"chosen variation_min: {var_min:.4f}")
    print(f"Saved: {OUT_JSON.relative_to(ROOT)}")
    print(f"Saved: {PROTO_NPZ.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
