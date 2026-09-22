"""Smoke test model GRU dinamis J/Z.

Menguji:
- model/scaler/label dapat dimuat;
- input (1,24,63) diterima; output 2 probabilitas, finite, sum ~ 1;
- label decode benar (0->J, 1->Z);
- shape selain (24,63) ditolak;
- preprocessing/sequence tidak menghasilkan NaN/inf;
- prediksi pada seluruh sequence included (in-sample, informasional).

Output: reports/dynamic_jz_smoke.txt

Jalankan:
    python scripts/smoke_test_dynamic_jz.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEQ_DIR = ROOT / "data" / "processed" / "dynamic_sequences"

MODEL_PATH = MODELS / "dynamic_jz_gru.keras"
SCALER_PATH = MODELS / "dynamic_jz_scaler.joblib"
LABELS_PATH = MODELS / "dynamic_jz_labels.json"
OUT = REPORTS / "dynamic_jz_smoke.txt"


class Result:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name, passed, detail=""):
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    res = Result()
    print("== Smoke test GRU J/Z ==")

    import keras
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    res.add("Model/scaler/label dimuat", True,
            f"classes={labels.get('classes')}")
    res.add("Label map 0/1 -> J/Z",
            labels.get("0") == "J" and labels.get("1") == "Z")

    # Kontrak shape via helper inference (juga dipakai tahap berikutnya)
    import dynamic_jz_inference as dyn
    predictor = dyn.DynamicJZPredictor()

    # input valid
    rng = np.random.default_rng(0)
    seq = rng.normal(0, 0.3, size=(1, 24, 63)).astype(np.float32)
    seq_s = scaler.transform(seq.reshape(-1, 63)).reshape(1, 24, 63).astype(np.float32)
    proba = model.predict(seq_s, verbose=0)
    res.add("Input (1,24,63) diterima & output (1,2)", proba.shape == (1, 2),
            f"shape={proba.shape}")
    res.add("Probabilitas finite", bool(np.all(np.isfinite(proba))))
    res.add("Probabilitas sum ~ 1", abs(float(proba.sum()) - 1.0) < 1e-5,
            f"sum={float(proba.sum()):.6f}")
    idx = int(np.argmax(proba))
    res.add("Label decode benar", labels[str(idx)] in ("J", "Z"),
            f"pred={labels[str(idx)]} p={float(proba[0, idx]):.3f}")

    # shape salah ditolak (kontrak (24,63) diberlakukan oleh helper inference).
    # Catatan: Keras GRU secara teknis menerima panjang waktu bervariasi,
    # sehingga kontrak ditegakkan di DynamicJZPredictor.
    for bad in ((1, 20, 63), (1, 30, 63), (1, 24, 62), (1, 63), (2, 24, 63, 1)):
        try:
            predictor.predict(np.zeros(bad, dtype=np.float32))
            rejected = False
        except ValueError:
            rejected = True
        res.add(f"Shape {bad} ditolak", rejected)
    res.add("Sequence tunggal (24,63) diterima",
            predictor.predict(seq_s[0]).shape == (1, 2))
    try:
        predictor.predict(np.full((1, 24, 63), np.nan, dtype=np.float32))
        rejected_nan = False
    except ValueError:
        rejected_nan = True
    res.add("Input NaN ditolak", rejected_nan)
    res.add("Helper predict_label bekerja",
            predictor.predict_label(seq_s[0])[0] in ("J", "Z"),
            f"{predictor.predict_label(seq_s[0])}")

    # sequence nyata: tidak NaN/inf + traceability + akurasi in-sample
    X, y = [], []
    for lab, yi in (("J", 0), ("Z", 1)):
        for f in sorted((SEQ_DIR / lab).glob("*.npz")):
            with np.load(f, allow_pickle=True) as d:
                s = np.asarray(d["sequence"], dtype=np.float32)
                src = str(d["source_video"])
            res.add(f"Sequence {src} shape (24,63) & finite",
                    s.shape == (24, 63) and bool(np.all(np.isfinite(s))))
            X.append(s)
            y.append(yi)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    Xs = scaler.transform(X.reshape(-1, 63)).reshape(X.shape).astype(np.float32)
    res.add("Scaling tidak menghasilkan NaN/inf", bool(np.all(np.isfinite(Xs))))

    proba_all = model.predict(Xs, verbose=0)
    pred = np.argmax(proba_all, axis=1)
    acc = float((pred == y).mean())
    res.add("Prediksi seluruh sequence (in-sample) >= 0.80", acc >= 0.80,
            f"acc={acc:.3f} (n={len(y)})")

    elapsed = time.perf_counter() - start
    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    lines = [
        "SMOKE TEST - GRU DINAMIS J/Z",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Waktu: {elapsed:.2f} detik",
        f"Status: {'LULUS' if passed == total else 'ADA GAGAL'} ({passed}/{total})",
        "",
    ]
    for i, (name, p, detail) in enumerate(res.items, 1):
        lines.append(f"{i:2d}. [{'PASS' if p else 'FAIL'}] {name}"
                     + (f" - {detail}" if detail else ""))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nHasil: {passed}/{total} lulus")
    print(f"Rincian: {OUT.relative_to(ROOT)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
