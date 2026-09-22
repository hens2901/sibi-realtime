"""Analisis live kelas statis sulit: CURRENT vs V2 pada landmark yang sama.

Membaca reports/static_hard_live.csv (+ npz features). Fitur 63 dari npz
dievaluasi dengan KEDUA model (current & V2) memakai threshold sama
(confidence >= 0.85, margin >= 0.20) agar perbandingan adil.

Metrik per kelas per model: top1 accuracy, accepted-correct rate,
correct-but-rejected, misclassification, ambiguity rate (margin<0.20),
mean confidence, mean margin. Plus confusion comparison R/U/V.

Output:
- reports/static_hard_live_comparison.csv
- reports/static_hard_live_confusion.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realtime as rt  # noqa: E402

CSV_PATH = ROOT / "reports" / "static_hard_live.csv"
OUT_CMP = ROOT / "reports" / "static_hard_live_comparison.csv"
OUT_CONF = ROOT / "reports" / "static_hard_live_confusion.csv"
HARD = ["K", "N", "R", "U", "V", "X"]
MODELS = ROOT / "models"
CONF_TH, MARGIN_TH = 0.85, 0.20


def load_models():
    cur = rt.load_artifacts("augmented")
    v2_model = joblib.load(MODELS / "sibi_mlp_augmented_v2.joblib")
    v2_scaler = joblib.load(MODELS / "scaler_augmented_v2.joblib")
    v2_enc = joblib.load(MODELS / "label_encoder_augmented_v2.joblib")
    return {"current": cur,
            "v2": (v2_model, v2_scaler, v2_enc, [str(x) for x in v2_enc.classes_])}


def classify_one(bundle, feats, target):
    model, scaler, enc, labels = bundle
    probs = rt.predict_proba(model, scaler, feats)
    t2 = rt.top2_from_probs(probs, list(labels))
    accepted = rt.passes_rejection(t2.top1_prob, t2.margin, CONF_TH, MARGIN_TH)
    correct = t2.top1_label == target
    if t2.margin < MARGIN_TH:
        ftype = "AMBIGUOUS"
    elif correct and accepted:
        ftype = "CORRECT_ACCEPTED"
    elif correct:
        ftype = "CORRECT_REJECTED"
    else:
        ftype = "MISCLASSIFIED"
    return {"top1": t2.top1_label, "conf": float(t2.top1_prob),
            "margin": float(t2.margin), "accepted": bool(accepted),
            "correct": bool(correct), "ftype": ftype}


def main() -> int:
    if not CSV_PATH.exists():
        print("Belum ada data live reports/static_hard_live.csv.")
        print("Jalankan: python scripts/diagnose_static_hard_classes.py")
        return 0
    bundles = load_models()
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    print(f"== Live comparison CURRENT vs V2 ==  n={len(rows)}")

    per = {m: defaultdict(list) for m in ("current", "v2")}
    conf = {m: Counter() for m in ("current", "v2")}
    for r in rows:
        target = r["target"]
        npz = ROOT / r.get("npz_path", "")
        if npz.exists():
            with np.load(npz, allow_pickle=True) as d:
                feats = np.asarray(d["features"], dtype=np.float64)
        else:
            # fallback: tidak ada fitur -> pakai kolom CSV (hanya model yg ada)
            for m in ("current", "v2"):
                key = f"{m}_top1"
                if key in r and r[key]:
                    per[m][target].append({
                        "top1": r[key], "conf": float(r.get(f"{m}_conf") or 0),
                        "margin": float(r.get(f"{m}_margin") or 0),
                        "accepted": str(r.get(f"{m}_accepted")).lower() == "true",
                        "correct": str(r.get(f"{m}_correct")).lower() == "true"})
            continue
        for m in ("current", "v2"):
            res = classify_one(bundles[m], feats, target)
            per[m][target].append(res)
            conf[m][(target, res["top1"])] += 1

    out_rows = []
    for cls in HARD:
        for m in ("current", "v2"):
            rs = per[m].get(cls, [])
            n = len(rs)
            if n == 0:
                continue
            top1_acc = sum(r["correct"] for r in rs) / n
            acc_correct = sum(1 for r in rs if r["correct"] and r["accepted"]) / n
            cbr = sum(1 for r in rs if r["correct"] and not r["accepted"])
            mis = sum(1 for r in rs if not r["correct"])
            amb = sum(1 for r in rs if r["ftype"] == "AMBIGUOUS")
            row = {"class": cls, "model": m, "n": n,
                   "top1_accuracy": round(top1_acc, 3),
                   "accepted_correct_rate": round(acc_correct, 3),
                   "correct_but_rejected": cbr,
                   "misclassified": mis,
                   "ambiguity_rate": round(amb / n, 3),
                   "mean_confidence": round(float(np.mean([r["conf"] for r in rs])), 3),
                   "mean_margin": round(float(np.mean([r["margin"] for r in rs])), 3)}
            out_rows.append(row)
            print(f"{cls} {m:8s} n={n} top1={top1_acc:.2f} acc_correct={acc_correct:.2f} "
                  f"CBR={cbr} mis={mis} amb={amb} conf={row['mean_confidence']:.2f} "
                  f"margin={row['mean_margin']:.2f}")

    if out_rows:
        with OUT_CMP.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    # Confusion R/U/V per model
    with OUT_CONF.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "target", "top1", "count", "type"])
        for m in ("current", "v2"):
            for (t, p), c in sorted(conf[m].items(), key=lambda x: -x[1]):
                if t in ("R", "U", "V"):
                    w.writerow([m, t, p, c, "correct" if t == p else "confusion"])
        for m in ("current", "v2"):
            print(f"\n{m.upper()} confusion R/U/V:")
            for (t, p), c in sorted(conf[m].items(), key=lambda x: -x[1]):
                if t in ("R", "U", "V"):
                    print(f"  {t} -> {p}: {c} [{'OK' if t==p else 'CONF'}]")

    print(f"\nCSV: {OUT_CMP.relative_to(ROOT)}")
    print(f"CSV: {OUT_CONF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
