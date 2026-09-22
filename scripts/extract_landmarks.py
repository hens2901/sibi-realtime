"""Ekstraksi hand landmark MediaPipe untuk dataset SIBI Mono_Background.

- Read-only terhadap dataset asli.
- 1 tangan per gambar, 21 landmark, fitur x,y,z => 63 fitur.
- Normalisasi: translasi relatif wrist + skala ukuran tangan.

Output:
- data/processed/sibi_landmarks.csv
- reports/landmark_failures.csv
- reports/landmark_detection_summary.csv
- reports/LANDMARK_REPORT.md
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

LANDMARKS_CSV = PROCESSED_DIR / "sibi_landmarks.csv"
FAILURES_CSV = REPORTS_DIR / "landmark_failures.csv"
SUMMARY_CSV = REPORTS_DIR / "landmark_detection_summary.csv"
REPORT_MD = REPORTS_DIR / "LANDMARK_REPORT.md"

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # 63


def build_landmarker() -> "vision.HandLandmarker":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model HandLandmarker tidak ditemukan: {MODEL_PATH}\n"
            "Unduh dari https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks) -> list[float]:
    """Translasi relatif wrist lalu skala berdasarkan ukuran tangan."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    wrist = pts[0]
    translated = pts - wrist
    scale = float(np.max(np.linalg.norm(translated, axis=1)))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    normalized = translated / scale
    return normalized.reshape(-1).tolist()


def extract() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)
    landmarker = build_landmarker()

    feature_rows: list[tuple[list[float], str]] = []
    failures: list[str] = []
    per_class_total: dict[str, int] = {}
    per_class_detected: dict[str, int] = {}

    total = 0
    for cdir in class_dirs:
        label = cdir.name
        images = sorted([f for f in cdir.iterdir() if f.is_file()])
        per_class_total[label] = len(images)
        per_class_detected[label] = 0

        for img_path in images:
            total += 1
            rel = img_path.relative_to(ROOT).as_posix()
            try:
                with Image.open(img_path) as im:
                    rgb = np.asarray(im.convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                failures.append(rel)
                print(f"[READ-FAIL] {rel}: {type(exc).__name__}", file=sys.stderr)
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                failures.append(rel)
                continue

            feats = normalize_landmarks(result.hand_landmarks[0])
            if len(feats) != NUM_FEATURES:
                failures.append(rel)
                continue

            feature_rows.append((feats, label))
            per_class_detected[label] += 1

    landmarker.close()

    detected = len(feature_rows)
    return {
        "class_dirs": [d.name for d in class_dirs],
        "feature_rows": feature_rows,
        "failures": failures,
        "per_class_total": per_class_total,
        "per_class_detected": per_class_detected,
        "total": total,
        "detected": detected,
        "failed": total - detected,
        "detection_rate": detected / total if total else 0.0,
    }


def write_landmarks(res: dict) -> dict:
    checks = {
        "rows": 0,
        "feature_cols": NUM_FEATURES,
        "nan_rows": 0,
        "missing_value_rows": 0,
        "wrong_feature_count_rows": 0,
        "non_finite_values": 0,
        "class_distribution": defaultdict(int),
    }
    header = [f"f{i}" for i in range(1, NUM_FEATURES + 1)] + ["label"]

    with LANDMARKS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for feats, label in res["feature_rows"]:
            checks["rows"] += 1
            checks["class_distribution"][label] += 1
            if len(feats) != NUM_FEATURES:
                checks["wrong_feature_count_rows"] += 1
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in feats):
                checks["nan_rows"] += 1
            if any(v is None for v in feats):
                checks["missing_value_rows"] += 1
            if any(not math.isfinite(float(v)) for v in feats):
                checks["non_finite_values"] += 1
            writer.writerow([f"{v:.8f}" for v in feats] + [label])

    checks["class_distribution"] = dict(checks["class_distribution"])
    return checks


def write_failures(res: dict) -> None:
    with FAILURES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path"])
        for path in res["failures"]:
            writer.writerow([path])


def write_summary(res: dict) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "class",
                "jumlah_gambar",
                "berhasil_dideteksi",
                "gagal_dideteksi",
                "detection_rate",
                "persentase_dataset",
            ]
        )
        total = res["total"] or 1
        for label in res["class_dirs"]:
            tot = res["per_class_total"][label]
            det = res["per_class_detected"][label]
            rate = det / tot if tot else 0.0
            writer.writerow(
                [label, tot, det, tot - det, f"{rate:.4f}", f"{tot / total * 100:.2f}%"]
            )
        writer.writerow(
            [
                "TOTAL",
                res["total"],
                res["detected"],
                res["failed"],
                f"{res['detection_rate']:.4f}",
                "100.00%",
            ]
        )


def write_report(res: dict, checks: dict) -> None:
    lines: list[str] = []
    a = lines.append
    a("# LANDMARK REPORT - SIBI (MediaPipe Hands)")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Sumber: `{DATASET_DIR.relative_to(ROOT).as_posix()}`")
    a(f"- Model: MediaPipe HandLandmarker (`{MODEL_PATH.relative_to(ROOT).as_posix()}`)")
    a("- Mode: read-only (dataset asli tidak diubah), 1 tangan/gambar, 21 landmark x (x,y,z) = 63 fitur")
    a("- Normalisasi: translasi relatif wrist + skala max jarak landmark terhadap wrist")
    a("")
    a("## 1. Ringkasan Deteksi")
    a("")
    a(f"- Jumlah gambar total: **{res['total']}**")
    a(f"- Berhasil dideteksi: **{res['detected']}**")
    a(f"- Gagal dideteksi: **{res['failed']}**")
    a(f"- Detection rate keseluruhan: **{res['detection_rate'] * 100:.2f}%**")
    a("")
    a("## 2. Detection Rate per Kelas")
    a("")
    a("| Kelas | Jumlah | Berhasil | Gagal | Detection Rate |")
    a("|:-----:|-------:|---------:|------:|---------------:|")
    for label in res["class_dirs"]:
        tot = res["per_class_total"][label]
        det = res["per_class_detected"][label]
        rate = det / tot if tot else 0.0
        a(f"| {label} | {tot} | {det} | {tot - det} | {rate * 100:.2f}% |")
    a(
        f"| **TOTAL** | **{res['total']}** | **{res['detected']}** | "
        f"**{res['failed']}** | **{res['detection_rate'] * 100:.2f}%** |"
    )
    a("")
    a("## 3. Pemeriksaan Kualitas Fitur")
    a("")
    a(f"- Jumlah baris fitur: **{checks['rows']}**")
    a(f"- Jumlah kolom fitur (harus 63): **{checks['feature_cols']}**")
    a(f"- Baris dengan jumlah fitur != 63: **{checks['wrong_feature_count_rows']}**")
    a(f"- Baris dengan NaN: **{checks['nan_rows']}**")
    a(f"- Baris dengan missing value: **{checks['missing_value_rows']}**")
    a(f"- Nilai non-finite (inf): **{checks['non_finite_values']}**")
    a("")
    a("## 4. Distribusi Kelas Setelah Ekstraksi")
    a("")
    a("| Kelas | Jumlah Sampel | Persentase |")
    a("|:-----:|--------------:|-----------:|")
    detected_total = checks["rows"] or 1
    for label in res["class_dirs"]:
        cnt = checks["class_distribution"].get(label, 0)
        a(f"| {label} | {cnt} | {cnt / detected_total * 100:.2f}% |")
    a(f"| **TOTAL** | **{checks['rows']}** | **100.00%** |")
    a("")
    a("## 5. File Gagal Deteksi")
    a("")
    if res["failures"]:
        a(f"Terdapat **{len(res['failures'])}** gambar gagal dideteksi. Daftar path lengkap tersimpan di "
          f"`{FAILURES_CSV.relative_to(ROOT).as_posix()}` (hanya path, file tidak disalin).")
    else:
        a("Tidak ada gambar yang gagal dideteksi (detection rate 100%).")
    a("")
    a("## 6. Kelayakan MediaPipe Landmark")
    a("")
    rate = res["detection_rate"]
    if rate >= 0.95:
        verdict = "SANGAT LAYAK"
        note = "Hampir seluruh gambar berhasil dideteksi."
    elif rate >= 0.90:
        verdict = "LAYAK"
        note = "Detection rate tinggi; sebagian kecil gambar perlu ditinjau."
    elif rate >= 0.75:
        verdict = "CUKUP LAYAK"
        note = "Detection rate sedang; perlu penanganan pada gambar gagal."
    else:
        verdict = "KURANG LAYAK"
        note = "Banyak gambar gagal dideteksi; pendekatan ini perlu dievaluasi ulang."
    a(f"- Detection rate: **{rate * 100:.2f}%**")
    a(f"- Kesimpulan: **{verdict}** - {note}")
    a("")
    a("> Tidak ada training, augmentasi, data sintetis, atau CNN pada tahap ini.")
    a("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    res = extract()
    checks = write_landmarks(res)
    write_failures(res)
    write_summary(res)
    write_report(res, checks)
    print(f"Total images     : {res['total']}")
    print(f"Detected         : {res['detected']}")
    print(f"Failed           : {res['failed']}")
    print(f"Detection rate   : {res['detection_rate'] * 100:.2f}%")
    print(f"NaN rows         : {checks['nan_rows']}")
    print(f"Missing rows     : {checks['missing_value_rows']}")
    print(f"Wrong feat count : {checks['wrong_feature_count_rows']}")
    print(f"Landmarks CSV    : {LANDMARKS_CSV.relative_to(ROOT)}")
    print(f"Failures CSV     : {FAILURES_CSV.relative_to(ROOT)}")
    print(f"Summary CSV      : {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Report MD        : {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
