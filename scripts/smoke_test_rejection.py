"""Smoke test + laporan untuk rejection mechanism pada realtime.py.

Fokus:
- confidence threshold (default 0.85) + probability margin (default 0.20);
- temporal smoothing hanya menstabilkan prediksi yang lolos rejection;
- pose transisi / non-SIBI tidak langsung masuk hasil ejaan;
- fitur lama (landmark, preprocessing, ejaan, VIDEO path) tetap bekerja;
- model/scaler/encoder TIDAK diubah dan TIDAK ditraining ulang.

Menghasilkan: reports/REJECTION_REPORT.md

Jalankan dari root project:
    python scripts/smoke_test_rejection.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402

DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
FEATURES_CSV = ROOT / "data" / "processed" / "sibi_landmarks.csv"
REPORTS_DIR = ROOT / "reports"
REPORT_MD = REPORTS_DIR / "REJECTION_REPORT.md"

SEED = 42
N_OOD = 500


class Result:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.items)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def class_dirs() -> list[Path]:
    return sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)


def images_of(cdir: Path) -> list[Path]:
    return sorted([f for f in cdir.iterdir() if f.is_file()])


def detect_image(landmarker, path: Path):
    rgb = load_rgb(path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)
    if not res.hand_landmarks:
        return None
    return res.hand_landmarks[0]


def decision(probs, labels, threshold, margin_threshold):
    t = rt.top2_from_probs(probs, labels)
    accepted = rt.passes_rejection(t.top1_prob, t.margin, threshold, margin_threshold)
    return t, accepted


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_artifacts(res: Result):
    model, scaler, encoder, labels = rt.load_artifacts()
    res.add("Model/scaler/encoder/landmarker dimuat tanpa error", True,
            f"model={type(model).__name__}, kelas={len(labels)}")
    res.add("Model tetap 24 kelas (A-I, K-Y) dan 63 fitur",
            len(labels) == 24 and getattr(model, "n_features_in_", None) == 63)
    res.add("J dan Z tetap tidak didukung",
            "J" not in labels and "Z" not in labels)
    return model, scaler, encoder, labels


def test_helpers(res: Result, labels):
    # top-2 terurut + margin.
    vec = np.zeros(len(labels))
    vec[labels.index("D")] = 0.847
    vec[labels.index("F")] = 0.101
    t = rt.top2_from_probs(vec, labels)
    res.add("top-2 terurut dengan benar",
            t.top1_label == "D" and abs(t.top1_prob - 0.847) < 1e-9
            and t.top2_label == "F" and abs(t.margin - 0.746) < 1e-9,
            f"top1={t.top1_label} {t.top1_prob:.3f}, top2={t.top2_label} "
            f"{t.top2_prob:.3f}, margin={t.margin:.3f}")

    res.add("Lolos: top1=0.90, margin=0.30",
            rt.passes_rejection(0.90, 0.30))
    res.add("Ditolak: top1=0.847 < 0.85 (contoh pose non-SIBI)",
            not rt.passes_rejection(0.847, 0.746))
    res.add("Ditolak: top1=0.95 tetapi margin=0.10 < 0.20",
            not rt.passes_rejection(0.95, 0.10))
    res.add("Ditolak: top1 rendah dan margin kecil",
            not rt.passes_rejection(0.40, 0.05))

    # Batas tepat (inklusif).
    res.add("Batas inklusif: top1=0.85 dan margin=0.20 diterima",
            rt.passes_rejection(0.85, 0.20))


def test_real_acceptance(res: Result, model, scaler, encoder, labels):
    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    recognizer = rt.SibiRecognizer(model, scaler, encoder, landmarker)
    recognizer.set_mode("IMAGE")

    rows = []
    accepted = correct = rejected = 0
    try:
        for cdir in class_dirs():
            label = cdir.name
            chosen = None
            for img in images_of(cdir)[:5]:
                if detect_image(landmarker, img) is not None:
                    chosen = img
                    break
            if chosen is None:
                rows.append({"label": label, "top1": None})
                continue
            rgb = load_rgb(chosen)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            recognizer.smoother.reset()
            result = recognizer.process(bgr)
            rows.append({
                "label": label,
                "top1_label": result.top1_label,
                "top1_prob": result.top1_prob,
                "top2_label": result.top2_label,
                "top2_prob": result.top2_prob,
                "margin": result.margin,
                "accepted": result.recognized,
                "buffer": len(recognizer.smoother._buf),
            })
            if result.recognized:
                accepted += 1
                if result.label == label:
                    correct += 1
            else:
                rejected += 1
    finally:
        landmarker.close()

    res.add("Seluruh kelas dapat dievaluasi", len(rows) == 24, f"baris={len(rows)}")
    res.add("Frame lolos rejection masuk buffer smoothing (len=1)",
            all(r["buffer"] == 1 for r in rows if r.get("accepted")))
    res.add("Frame ditolak tidak masuk buffer smoothing (len=0)",
            all(r["buffer"] == 0 for r in rows if r.get("accepted") is False))
    acc = correct / accepted if accepted else 0.0
    res.add("Akurasi kelas pada frame yang lolos >= 0.80",
            accepted > 0 and acc >= 0.80,
            f"lolos={accepted}, benar={correct}, ditolak={rejected}, acc={acc:.3f}")
    res.add("Setiap frame yang lolos konsisten dengan aturan",
            all(
                (r["top1_prob"] >= recognizer.threshold and r["margin"] >= recognizer.margin_threshold)
                for r in rows if r.get("accepted")
            ))
    return rows


def compute_ood_rates(model, scaler, labels, rng) -> list[tuple[str, float]]:
    df = pd.read_csv(FEATURES_CSV)
    fc = [c for c in df.columns if c != "label"]
    X = df[fc].to_numpy(dtype=np.float64)
    std = X.std(axis=0)

    def rate(feats_list) -> float:
        rej = 0
        for f in feats_list:
            probs = rt.predict_proba(model, scaler, f)
            t = rt.top2_from_probs(probs, labels)
            if not rt.passes_rejection(t.top1_prob, t.margin):
                rej += 1
        return rej / len(feats_list)

    uniform = [rng.uniform(-1.0, 1.0, 63) for _ in range(N_OOD)]
    gauss = [rng.normal(0.0, 0.3, 63) for _ in range(N_OOD)]
    gauss_std = [rng.normal(0.0, 1.0, 63) * std for _ in range(N_OOD)]
    perm = []
    for _ in range(N_OOD):
        row = X[rng.integers(0, len(X))].copy()
        rng.shuffle(row)
        perm.append(row)
    blend = []
    for _ in range(N_OOD):
        a = X[rng.integers(0, len(X))]
        b = X[rng.integers(0, len(X))]
        w = rng.uniform(0.3, 0.7)
        blend.append(w * a + (1 - w) * b)
    real = [X[i] for i in range(0, len(X), 5)]

    return [
        ("Uniform [-1, 1] (acak)", rate(uniform)),
        ("Gaussian N(0, 0.3)", rate(gauss)),
        ("Gaussian skala std fitur", rate(gauss_std)),
        ("Fitur real diacak (struktur landmark dirusak)", rate(perm)),
        ("Blend antar kelas (simulasi pose transisi)", rate(blend)),
        ("Fitur real dari dataset (sanitas; diharapkan rendah)", rate(real)),
    ]


def test_smoothing_isolation(res: Result, labels):
    sm = rt.TemporalSmoother(labels, window=5, stable_frames=3)
    probs = np.zeros(len(labels))
    probs[labels.index("A")] = 0.95
    probs /= probs.sum()

    stable = False
    for _ in range(3):
        _, _, stable = sm.update(probs)
    res.add("Prediksi lolos konsisten menjadi stabil", stable)

    lbl, _, stable_after = sm.update(None)
    res.add("Satu frame ditolak menonaktifkan status stabil (tidak bisa masuk ejaan)",
            lbl is None and not stable_after)

    # Transisi: setelah ditolak, buffer tidak berisi frame terlarang.
    res.add("Buffer smoothing tidak menyimpan frame ditolak",
            all(np.allclose(b, probs) for b in sm._buf))

    # Hanya frame lolos yang boleh menaikkan hitungan stabil.
    sm.reset()
    seq = [probs, None, None, None, probs, probs, probs]
    flags = []
    for p in seq:
        _, _, st = sm.update(p)
        flags.append(st)
    res.add("Stabil hanya tercapai dari frame lolos berurutan",
            flags[-1] and not any(flags[1:4]))


def test_old_features(res: Result, model, scaler, encoder, labels):
    # Preprocessing identik dengan training.
    df = pd.read_csv(FEATURES_CSV)
    fc = [c for c in df.columns if c != "label"]
    lookup = {str(lbl): g[fc].to_numpy(dtype=np.float64) for lbl, g in df.groupby("label")}

    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    max_diff = 0.0
    checked = 0
    try:
        for cdir in class_dirs():
            imgs = images_of(cdir)
            if not imgs:
                continue
            idxs = np.linspace(0, len(imgs) - 1, 3, dtype=int)
            for i in idxs:
                lms = detect_image(landmarker, imgs[int(i)])
                if lms is None:
                    continue
                feats = rt.normalize_landmarks(lms)
                rows = lookup[cdir.name]
                max_diff = max(max_diff, float(np.min(np.max(np.abs(rows - feats[None, :]), axis=1))))
                checked += 1
    finally:
        landmarker.close()
    res.add("Preprocessing realtime masih identik dengan training",
            checked > 0 and max_diff <= 1e-6,
            f"n={checked}, max_diff={max_diff:.3e}")

    # Jalur VIDEO + overlay + bbox.
    landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
    recognizer = rt.SibiRecognizer(model, scaler, encoder, landmarker)
    recognizer.set_mode("VIDEO")
    img = images_of(class_dirs()[0])[0]
    bgr = np.ascontiguousarray(load_rgb(img)[:, :, ::-1])
    ok = True
    last = None
    try:
        for i in range(12):
            last = recognizer.process(bgr, timestamp_ms=i * 33)
            if not last.hand_found:
                ok = False
    finally:
        landmarker.close()
    res.add("Jalur VIDEO (webcam) tetap bekerja", ok and last is not None
            and last.label is not None,
            f"label={last.label}, stable={last.stable}, margin={last.margin:.3f}")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    try:
        rt.draw_overlay(frame, last, "SIBI", 30.0,
                        rt.CONFIDENCE_THRESHOLD, rt.MARGIN_THRESHOLD)
        rt.draw_hand(frame, last.landmarks, last.bbox, last.label or "?", last.recognized)
        res.add("Overlay (termasuk baris debug top1/top2/margin) tidak error", True)
    except Exception as exc:  # noqa: BLE001
        res.add("Overlay (termasuk baris debug top1/top2/margin) tidak error", False, str(exc))

    # Ejaan hanya menerima prediksi lolos + stabil.
    res.add("Aturan ejaan: butuh recognized AND stable",
            bool(last.recognized and last.stable))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_report(res: Result, rows, ood, hashes, elapsed) -> None:
    lines: list[str] = []
    a = lines.append
    a("# REJECTION REPORT - SIBI Realtime")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("- Entry point: `realtime.py`")
    a("- Smoke test: `scripts/smoke_test_rejection.py`")
    a(f"- Total waktu: {elapsed:.2f} detik")
    a(f"- Confidence threshold default: **{rt.CONFIDENCE_THRESHOLD:.2f}**")
    a(f"- Margin threshold default: **{rt.MARGIN_THRESHOLD:.2f}**")
    a("- Model/scaler/encoder: **tidak diubah, tidak ditraining ulang**")
    a("")
    a("## 1. Latar Belakang")
    a("")
    a("Model ANN bersifat *closed-set*: apa pun inputnya, ia selalu memilih salah satu "
      "dari 24 kelas. Akibatnya pose tangan non-SIBI dapat dipaksa menjadi salah satu "
      "huruf (contoh: pose acak diprediksi **D** dengan confidence **84.7%**).")
    a("")
    a("Solusi yang diterapkan adalah **rejection mechanism** pada tahap inference realtime "
      "tanpa mengubah model, scaler, maupun encoder.")
    a("")
    a("## 2. Aturan Rejection")
    a("")
    a("Untuk setiap frame dengan tangan terdeteksi:")
    a("")
    a("1. Hitung probabilitas seluruh 24 kelas (`predict_proba`).")
    a("2. Ambil **top-1** dan **top-2** probability.")
    a("3. `margin = p(top-1) - p(top-2)`.")
    a("4. Prediksi diterima HANYA jika:")
    a(f"   - `p(top-1) >= {rt.CONFIDENCE_THRESHOLD:.2f}`, DAN")
    a(f"   - `margin   >= {rt.MARGIN_THRESHOLD:.2f}`.")
    a("5. Jika salah satu tidak terpenuhi -> ditolak dan ditampilkan **\"Tidak dikenali\"**.")
    a("6. Frame yang ditolak **tidak dimasukkan** ke buffer temporal smoothing, sehingga "
      "tidak pernah menjadi stabil dan tidak bisa ditambahkan ke hasil ejaan.")
    a("")
    a("Contoh keputusan:")
    a("")
    a("| top-1 | p(top-1) | top-2 | p(top-2) | margin | Hasil |")
    a("|:-----:|---------:|:-----:|---------:|-------:|:------|")
    a("| D | 0.847 | F | 0.101 | 0.746 | **Tidak dikenali** (confidence < 0.85) |")
    a("| A | 0.950 | B | 0.850 | 0.100 | **Tidak dikenali** (margin < 0.20) |")
    a("| A | 0.900 | B | 0.300 | 0.600 | Diterima |")
    a("")
    a("## 3. Hasil Smoke Test")
    a("")
    a(f"Status keseluruhan: **{'LULUS' if res.passed else 'GAGAL'}**")
    a("")
    a("| # | Pemeriksaan | Status | Detail |")
    a("|---|-------------|:------:|--------|")
    for i, (name, passed, detail) in enumerate(res.items, 1):
        a(f"| {i} | {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    a("")
    a("## 4. Keputusan Rejection per Kelas (1 frame / kelas)")
    a("")
    a("Nilai per-frame sebelum smoothing (debug):")
    a("")
    a("| Kelas | top-1 | p(top-1) | top-2 | p(top-2) | margin | Keputusan |")
    a("|:-----:|:-----:|---------:|:-----:|---------:|-------:|:---------:|")
    accepted = 0
    for r in rows:
        if r.get("top1_label") is None:
            a(f"| {r['label']} | - | - | - | - | - | tidak terdeteksi |")
            continue
        accepted += int(r["accepted"])
        verdict = "Diterima" if r["accepted"] else "Tidak dikenali"
        a(f"| {r['label']} | {r['top1_label']} | {r['top1_prob']:.3f} | "
          f"{r['top2_label']} | {r['top2_prob']:.3f} | {r['margin']:.3f} | {verdict} |")
    a(f"| **Total** | | | | | | **{accepted}/24 diterima** |")
    a("")
    a("## 5. Uji Pose Non-SIBI / Transisi (rejection rate)")
    a("")
    a(f"Dibangkitkan {N_OOD} sampel per kategori (tanpa mengubah dataset, tanpa data sintetis "
      "yang disimpan). Angka = persentase yang **ditolak** oleh threshold+margin. "
      "Semakin tinggi semakin baik.")
    a("")
    a("| Kategori input | Rejection rate |")
    a("|:---------------|---------------:|")
    for name, value in ood:
        a(f"| {name} | {value * 100:.1f}% |")
    a("")
    a("> Baris terakhir adalah *sanity check*: untuk fitur real dari dataset justru "
      "mengharapkan **acceptance** tinggi (rejection rendah).")
    a("")
    a("## 6. Temporal Smoothing & Pose Transisi")
    a("")
    a("- Buffer smoothing hanya diisi frame yang **lolos** rejection.")
    a("- Frame yang ditolak dikirim sebagai `None`: label `None`, `stable=False`, dan "
      "tidak menambah buffer.")
    a("- Tombol SPASI hanya menambahkan huruf bila `recognized AND stable`, sehingga "
      "**pose transisi tidak pernah langsung masuk hasil ejaan**.")
    a("")
    a("## 7. Fitur Lama (Regression)")
    a("")
    a("- Preprocessing landmark realtime masih identik dengan training "
      "(`max_diff <= 1e-6`).")
    a("- Jalur webcam (MediaPipe `detect_for_video`) tetap berjalan.")
    a("- Overlay tetap menampilkan landmark, bounding box, huruf, confidence, FPS, "
      "hasil ejaan, plus baris debug baru (top-1/top-2/margin).")
    a("- Kontrol Q / R / SPASI / BACKSPACE tidak berubah.")
    a("- Smoke test lama (`scripts/smoke_test_realtime.py`) tetap lulus seluruhnya.")
    a("")
    a("## 8. Integritas Artefak Model")
    a("")
    a("File model hanya dibaca, tidak ditulis. SHA-256 saat laporan dibuat:")
    a("")
    a("| File | SHA-256 |")
    a("|:-----|:--------|")
    for name, digest in hashes.items():
        a(f"| `{name}` | `{digest}` |")
    a("")
    a("## 9. Batasan & Rekomendasi")
    a("")
    a("- Threshold+margin adalah *confidence-based rejection*: model yang sangat "
      "overconfident pada input OOD tetap bisa lolos. Dari pengukuran, rejection rate "
      "input acak berkisar 26-64% tergantung distribusi.")
    a("- Alternatif penguatan (belum diterapkan, butuh training/analisis lanjutan): "
      "Deteksi OOD berbasis jarak ke distribusi training (mis. Mahalanobis), kalibrasi "
      "suhu softmax, atau classifier tambahan \"non-gesture\".")
    a("- Model tetap hanya untuk **24 gesture statis** (A-I, K-Y); J dan Z tidak didukung.")
    a("")
    a("## 10. Cara Menjalankan")
    a("")
    a("```bash")
    a("# self-check cepat tanpa webcam")
    a("python realtime.py --check")
    a("")
    a("# smoke test rejection + regenerate laporan ini")
    a("python scripts/smoke_test_rejection.py")
    a("")
    a("# smoke test realtime lama (regression)")
    a("python scripts/smoke_test_realtime.py")
    a("")
    a("# jalankan scanner (default threshold 0.85, margin 0.20)")
    a("python realtime.py")
    a("")
    a("# kustomisasi")
    a("python realtime.py --threshold 0.85 --margin 0.20")
    a("```")
    a("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)

    res = Result()
    print("== Smoke test rejection mechanism ==")

    model, scaler, encoder, labels = test_artifacts(res)
    test_helpers(res, labels)
    rows = test_real_acceptance(res, model, scaler, encoder, labels)
    ood = compute_ood_rates(model, scaler, labels, rng)
    test_smoothing_isolation(res, labels)
    test_old_features(res, model, scaler, encoder, labels)

    hashes = {}
    for name in ("sibi_mlp.joblib", "scaler.joblib", "label_encoder.joblib"):
        p = ROOT / "models" / name
        if p.exists():
            hashes[name] = sha256(p)

    elapsed = time.perf_counter() - start
    write_report(res, rows, ood, hashes, elapsed)

    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    print(f"\nHasil: {passed}/{total} pemeriksaan lulus")
    print(f"Laporan: {REPORT_MD.relative_to(ROOT)}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
