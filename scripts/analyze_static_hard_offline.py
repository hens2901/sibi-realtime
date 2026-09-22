"""Analisis OFFLINE kelas statis sulit (K,N,R,U,V,X) memakai dataset.

Reproduksi split baseline (random_state=42, test_size=0.2) pada
data/processed/sibi_landmarks.csv, lalu evaluasi model augmented (current) pada
test original dan mirrored (proxy x-negation). Fokus: correct-but-rejected,
misclassification, confusion pairs, per-class recall.

Ini analisis DATASET (bukan live). Live tetap wajib untuk keputusan akhir.

Output:
- reports/static_hard_offline_analysis.csv
- reports/static_hard_offline_confusion.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realtime as rt  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

CSV_FEATURES = ROOT / "data" / "processed" / "sibi_landmarks.csv"
OUT_A = ROOT / "reports" / "static_hard_offline_analysis.csv"
OUT_C = ROOT / "reports" / "static_hard_offline_confusion.csv"
HARD = ["K", "N", "R", "U", "V", "X"]
SEED = 42
TEST_SIZE = 0.2


def mirror_x(feats: np.ndarray) -> np.ndarray:
    out = np.array(feats, dtype=np.float64, copy=True)
    out[..., 0::3] *= -1.0
    return out


def evaluate(model, scaler, labels, X, y, threshold=0.85, margin=0.20):
    probs = model.predict_proba(scaler.transform(X))
    idx = np.argmax(probs, axis=1)
    pred = np.array([labels[i] for i in idx])
    accepted = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        t = rt.top2_from_probs(probs[i], list(labels))
        accepted[i] = rt.passes_rejection(t.top1_prob, t.margin, threshold, margin)
    return probs, pred, accepted


def main() -> int:
    df = pd.read_csv(CSV_FEATURES)
    fcols = [c for c in df.columns if c != "label"]
    X_all = df[fcols].to_numpy(dtype=np.float64)
    y_all = df["label"].astype(str).to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    model, scaler, encoder, labels = rt.load_artifacts("augmented")
    labels = list(labels)

    results = {}
    for tag, Xm in (("original", X_test), ("mirrored_proxy", mirror_x(X_test))):
        probs, pred, accepted = evaluate(model, scaler, labels, Xm, y_test)
        results[tag] = (probs, pred, accepted)

    # per-class recall + hard-class failure breakdown (original test)
    probs, pred, accepted = results["original"]
    print(f"== Offline hard-class analysis (augmented model, test n={len(y_test)}) ==")
    rows = []
    for cls in labels:
        m = y_test == cls
        n = int(m.sum())
        top1_ok = (pred[m] == cls)
        recall = float(top1_ok.mean()) if n else 0.0
        correct_rej = int(np.sum(top1_ok & ~accepted[m]))
        miscls = int(np.sum(~top1_ok))
        rows.append({"class": cls, "n": n, "recall_top1": round(recall, 3),
                     "correct_but_rejected": correct_rej, "misclassified": miscls,
                     "accepted_correct": int(np.sum(top1_ok & accepted[m]))})
        if cls in HARD:
            print(f"  {cls}: n={n} recall={recall:.2f} "
                  f"correct-but-rejected={correct_rej} misclassified={miscls}")

    with OUT_A.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # confusion pairs involving hard classes
    cm = Counter()
    for t, p in zip(y_test, pred):
        if t in HARD:
            cm[(t, p)] += 1
    with OUT_C.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "pred", "count", "type"])
        for (t, p), c in sorted(cm.items(), key=lambda x: -x[1]):
            w.writerow([t, p, c, "correct" if t == p else "confusion"])

    print("\nConfusion pairs (hard classes, target->pred) terbanyak:")
    for (t, p), c in sorted(cm.items(), key=lambda x: -x[1])[:15]:
        print(f"  {t} -> {p}: {c}  [{'OK' if t==p else 'CONF'}]")

    # mirrored proxy comparison
    _, pred_m, acc_m = results["mirrored_proxy"]
    print("\nMirrored (proxy x-negation) per-class recall hard classes:")
    for cls in HARD:
        m = y_test == cls
        print(f"  {cls}: recall={(pred_m[m]==cls).mean():.2f}")

    # ---- V2 comparison (offline) ----
    from pathlib import Path as _P
    import joblib as _joblib
    MODELS = ROOT / "models"
    if (MODELS / "sibi_mlp_augmented_v2.joblib").exists():
        v2_model = _joblib.load(MODELS / "sibi_mlp_augmented_v2.joblib")
        v2_scaler = _joblib.load(MODELS / "scaler_augmented_v2.joblib")
        v2_enc = _joblib.load(MODELS / "label_encoder_augmented_v2.joblib")
        v2_labels = [str(x) for x in v2_enc.classes_]
        probs_v2, pred_v2, acc_v2 = evaluate(v2_model, v2_scaler, v2_labels,
                                             X_test, y_test)
        comp = []
        print("\nHARD CLASS COMPARISON (offline original test) "
              "class | cur_top1 | v2_top1 | cur_accepted | v2_accepted")
        for cls in HARD:
            m = y_test == cls
            n = int(m.sum())
            cur_rec = float((pred[m] == cls).mean())
            v2_rec = float((pred_v2[m] == cls).mean())
            cur_ac = int(np.sum((pred[m] == cls) & accepted[m]))
            v2_ac = int(np.sum((pred_v2[m] == cls) & acc_v2[m]))
            comp.append({"class": cls, "n": n, "current_top1": round(cur_rec, 3),
                         "v2_top1": round(v2_rec, 3),
                         "current_accepted_correct": cur_ac,
                         "v2_accepted_correct": v2_ac})
            print(f"  {cls}: {cur_rec:.2f} | {v2_rec:.2f} | {cur_ac}/{n} | {v2_ac}/{n}")
        with (ROOT / "reports" / "static_hard_offline_comparison.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(comp[0].keys()))
            w.writeheader()
            w.writerows(comp)
        # V2 confusion pairs for hard classes
        cm_v2 = Counter((t, p) for t, p in zip(y_test, pred_v2) if t in HARD)
        with (ROOT / "reports" / "static_hard_offline_confusion_v2.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["target", "pred", "count", "type"])
            for (t, p), c in sorted(cm_v2.items(), key=lambda x: -x[1]):
                w.writerow([t, p, c, "correct" if t == p else "confusion"])
        print("\nV2 confusion pairs (hard classes):")
        for (t, p), c in sorted(cm_v2.items(), key=lambda x: -x[1]):
            if t in ("R", "U", "V"):
                print(f"  {t} -> {p}: {c} [{'OK' if t==p else 'CONF'}]")

    print(f"\nCSV: {OUT_A.relative_to(ROOT)}")
    print(f"CSV: {OUT_C.relative_to(ROOT)}")
    print("Catatan: mirrored memakai proxy x-negation (bukan image flip+re-detect).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
