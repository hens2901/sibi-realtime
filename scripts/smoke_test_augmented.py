"""Smoke test model augmented + pemilihan model di realtime.py.

Memvalidasi:
1. model baseline & augmented dapat dimuat (63 fitur, 24 kelas, urutan konsisten);
2. jalur `--model baseline|augmented` berfungsi;
3. rejection mechanism tetap utuh;
4. prediksi end-to-end pada sampel dataset untuk kedua model;
5. model baseline lama TIDAK berubah (masih bisa dimuat & memprediksi).

Output:
- reports/augmented_smoke_test.txt

Jalankan:
    python scripts/smoke_test_augmented.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realtime as rt  # noqa: E402

FEATURES_CSV = ROOT / "data" / "processed" / "sibi_landmarks.csv"
REPORTS_DIR = ROOT / "reports"
REPORT_TXT = REPORTS_DIR / "augmented_smoke_test.txt"


class Result:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.items)


def check_artifacts(res: Result, name: str):
    model, scaler, encoder, labels = rt.load_artifacts(name)
    res.add(f"[{name}] artefak termuat", True, f"kelas={len(labels)}")
    res.add(f"[{name}] 24 kelas", len(labels) == 24, f"ditemukan {len(labels)}")
    res.add(f"[{name}] model menerima 63 fitur",
            getattr(model, "n_features_in_", None) == 63)
    res.add(f"[{name}] scaler 63 fitur",
            getattr(scaler, "n_features_in_", None) == 63)
    res.add(f"[{name}] urutan kelas model == encoder",
            np.array_equal(model.classes_, np.arange(len(labels))))
    res.add(f"[{name}] J dan Z tidak didukung",
            "J" not in labels and "Z" not in labels)
    return model, scaler, encoder, labels


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    res = Result()
    print("== Smoke test augmented model ==")

    base = check_artifacts(res, "baseline")
    aug = check_artifacts(res, "augmented")

    # Prediksi pada sampel nyata (fitur CSV).
    df = pd.read_csv(FEATURES_CSV)
    fc = [c for c in df.columns if c != "label"]
    sample = df.iloc[::37][fc].to_numpy(dtype=np.float64)[:40]
    y_sample = df.iloc[::37]["label"].astype(str).to_numpy()[:40]
    for name, (model, scaler, encoder, labels) in (("baseline", base), ("augmented", aug)):
        probs = model.predict_proba(scaler.transform(pd.DataFrame(sample, columns=fc)))
        res.add(f"[{name}] predict_proba shape benar",
                probs.shape == (len(sample), 24), f"{probs.shape}")
        res.add(f"[{name}] probabilitas valid (sum~1)",
                np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
        preds = encoder.inverse_transform(np.argmax(probs, axis=1))
        acc = float((preds == y_sample).mean())
        res.add(f"[{name}] akurasi sampel >= 0.80", acc >= 0.80, f"acc={acc:.3f}")

    # CLI: pemilihan model.
    parser = rt.build_arg_parser()
    a_default = parser.parse_args([])
    res.add("Default --model = baseline",
            a_default.model == "baseline", f"model={a_default.model}")
    a_aug = parser.parse_args(["--model", "augmented"])
    res.add("Argumen --model augmented diproses", a_aug.model == "augmented")

    # Self-check realtime untuk kedua model.
    for name in ("baseline", "augmented"):
        rc = rt.run_check(name)
        res.add(f"realtime --check --model {name} lulus", rc == 0, f"rc={rc}")

    # Rejection mechanism tetap utuh.
    res.add("Rejection: lolos (0.90, 0.30)", rt.passes_rejection(0.90, 0.30))
    res.add("Rejection: tolak margin kecil", not rt.passes_rejection(0.95, 0.10))
    res.add("Rejection: tolak confidence rendah", not rt.passes_rejection(0.80, 0.50))
    res.add("Threshold default 0.85 / margin 0.20",
            abs(rt.CONFIDENCE_THRESHOLD - 0.85) < 1e-9
            and abs(rt.MARGIN_THRESHOLD - 0.20) < 1e-9)

    # Model lama tetap ada dan tidak berubah (masih memprediksi).
    res.add("File model baseline lama masih ada",
            rt.MLP_PATH.exists() and rt.SCALER_PATH.exists() and rt.ENCODER_PATH.exists())
    res.add("File model augmented ada",
            rt.MLP_PATH_AUGMENTED.exists() and rt.SCALER_PATH_AUGMENTED.exists()
            and rt.ENCODER_PATH_AUGMENTED.exists())

    elapsed = time.perf_counter() - start
    lines = [
        "SMOKE TEST - AUGMENTED MODEL & MODEL SELECTION",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Waktu: {elapsed:.2f} detik",
        f"Status: {'LULUS' if res.passed else 'GAGAL'}",
        "",
    ]
    for i, (name, passed, detail) in enumerate(res.items, 1):
        lines.append(f"{i:2d}. [{'PASS' if passed else 'FAIL'}] {name}"
                     + (f" - {detail}" if detail else ""))
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    print(f"\nHasil: {passed}/{total} pemeriksaan lulus")
    print(f"Rincian: {REPORT_TXT.relative_to(ROOT)}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
