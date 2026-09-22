"""Bandingkan dynamic event webcam (live) dengan distribusi training J/Z.

Membaca metadata dari reports/realtime_dynamic_debug/*.json (hasil
`hybrid_realtime_test.py --save-dynamic-debug`) dan membandingkan:
- movement, wrist, fingertip, velocity (dari metadata live)
- prototype distance, variation (dari metadata live)
- feature range (bila npz sequence tersimpan)

Menghasilkan ringkasan: posisi persentil live terhadap distribusi training,
serta jumlah dimensi yang out-of-distribution (di luar [p5, p95]).

Jika belum ada data live, script menjelaskan cara mengumpulkan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEQ_DIR = ROOT / "data" / "processed" / "dynamic_sequences"
DEBUG_DIR = ROOT / "reports" / "realtime_dynamic_debug"
PROTO_NPZ = ROOT / "models" / "dynamic_jz_prototypes.npz"
SCALER = ROOT / "models" / "dynamic_jz_scaler.joblib"
OUT = ROOT / "reports" / "live_vs_train_dynamic.csv"

DIMS = ("movement_magnitude", "wrist_movement", "fingertip_movement",
        "mean_velocity", "max_velocity", "distance", "variation")


def load_train() -> dict:
    scaler = joblib.load(SCALER)
    proto = np.load(PROTO_NPZ, allow_pickle=True)
    protos = {"J": proto["J"].astype(np.float64), "Z": proto["Z"].astype(np.float64)}
    data = {"J": {d: [] for d in DIMS}, "Z": {d: [] for d in DIMS}}
    for lab in ("J", "Z"):
        for f in sorted((SEQ_DIR / lab).glob("*.npz")):
            with np.load(f, allow_pickle=True) as d:
                seq = np.asarray(d["sequence"], dtype=np.float64)
                data[lab]["movement_magnitude"].append(float(d["movement_magnitude"]))
                data[lab]["wrist_movement"].append(float(d["wrist_movement"]))
                data[lab]["fingertip_movement"].append(float(d["fingertip_movement"]))
                data[lab]["mean_velocity"].append(float(d["temporal_velocity"]))
                step = np.linalg.norm(np.diff(seq, axis=0), axis=1)
                data[lab]["max_velocity"].append(float(step.max()))
                data[lab]["variation"].append(float(np.std(step)))
                scaled = scaler.transform(seq.reshape(-1, 63)).reshape(24, 63)
                data[lab]["distance"].append(float(min(
                    np.mean(np.linalg.norm(scaled - protos[cl], axis=1))
                    for cl in ("J", "Z"))))
    return {lab: {d: np.asarray(v, dtype=np.float64) for d, v in data[lab].items()}
            for lab in ("J", "Z")}


def percentile(train_vals: np.ndarray, v: float) -> float:
    if train_vals.size == 0:
        return float("nan")
    return float((train_vals <= v).mean() * 100.0)


def main() -> int:
    print("== Compare live vs train (dynamic J/Z) ==")
    if not DEBUG_DIR.exists() or not list(DEBUG_DIR.glob("*.json")):
        print("Belum ada data live.")
        print("Kumpulkan dulu: python hybrid_realtime_test.py --save-dynamic-debug")
        print("Lakukan >=5 percobaan J dan >=5 percobaan Z, lalu jalankan ulang script ini.")
        return 0

    train = load_train()
    rows = []
    for jf in sorted(DEBUG_DIR.glob("*.json")):
        if jf.name == "events.jsonl":
            continue
        m = json.loads(jf.read_text(encoding="utf-8"))
        pred = m.get("predicted") or "J"
        dist = train.get(pred)
        row = {"file": jf.name, "predicted": pred, "accepted": m.get("accepted"),
               "reason": m.get("reason", "")}
        for d in DIMS:
            v = m.get(d)
            if v is None or dist is None or d not in dist:
                row[d] = None
                row[d + "_pct"] = None
                continue
            v = float(v)
            pct = percentile(dist[d], v)
            row[d] = round(v, 4)
            row[d + "_pct"] = round(pct, 1)
        rows.append(row)

    # CSV
    cols = ["file", "predicted", "accepted", "reason"]
    for d in DIMS:
        cols += [d, d + "_pct"]
    import csv as _csv
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])

    # Ringkasan out-of-distribution (di luar p5-p95)
    ood_counts = {d: 0 for d in DIMS}
    low_counts = {d: 0 for d in DIMS}
    high_counts = {d: 0 for d in DIMS}
    for r in rows:
        for d in DIMS:
            p = r.get(d + "_pct")
            if p is None:
                continue
            if p < 5:
                ood_counts[d] += 1
                low_counts[d] += 1
            elif p > 95:
                ood_counts[d] += 1
                high_counts[d] += 1

    n = len(rows)
    n_acc = sum(1 for r in rows if r["accepted"])
    print(f"total events={n} accepted={n_acc} rejected={n - n_acc}")
    print("OOD (di luar p5-p95) per dimensi:")
    for d in DIMS:
        print(f"  {d:22s} ood={ood_counts[d]:3d}/{n} "
              f"(low={low_counts[d]}, high={high_counts[d]})")
    # Rekomendasi hipotesis
    worst = max(ood_counts, key=ood_counts.get)
    print(f"Dimensi paling OOD: {worst}")

    # Crosstab rejection reason
    reasons = {}
    for r in rows:
        if not r["accepted"]:
            for token in ("CONF", "MARGIN", "MOTION", "DIST", "VARIATION", "STEPS"):
                if token in r["reason"]:
                    reasons[token] = reasons.get(token, 0) + 1
    print(f"Alasan rejection dominan: {sorted(reasons.items(), key=lambda x: -x[1])}")
    print(f"CSV: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
