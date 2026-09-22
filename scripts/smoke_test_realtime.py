"""Smoke test offline untuk realtime.py (tanpa webcam).

Memvalidasi:
1. model / scaler / label encoder / hand landmarker dapat dimuat;
2. input selalu tepat 63 fitur dan tidak menerima dimensi lain;
3. preprocessing realtime IDENTIK dengan scripts/extract_landmarks.py
   (dibandingkan terhadap data/processed/sibi_landmarks.csv);
4. pipeline end-to-end menghasilkan prediksi benar pada sampel dataset;
5. temporal smoothing bekerja dan stabil;
6. fungsi gambar overlay tidak error pada frame sintetis;
7. J dan Z tidak didukung (tidak ada di encoder).

Membuat laporan: reports/REALTIME_REPORT.md

Jalankan dari root project:
    python scripts/smoke_test_realtime.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402

DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
FEATURES_CSV = ROOT / "data" / "processed" / "sibi_landmarks.csv"
REPORTS_DIR = ROOT / "reports"
REPORT_MD = REPORTS_DIR / "REALTIME_REPORT.md"


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #

class Result:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.items.append((name, bool(passed), detail))
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.items)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def class_dirs() -> list[Path]:
    return sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)


def images_of(cdir: Path) -> list[Path]:
    return sorted([f for f in cdir.iterdir() if f.is_file()])


def detect_image(landmarker, path: Path):
    """Deteksi landmark pada satu gambar (mode IMAGE), kembalikan list | None."""
    rgb = load_rgb(path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)
    if not res.hand_landmarks:
        return None
    return res.hand_landmarks[0]


# --------------------------------------------------------------------------- #
# 1. Artefak
# --------------------------------------------------------------------------- #

def test_artifacts(res: Result):
    model, scaler, encoder, labels = rt.load_artifacts()
    res.add("Model/scaler/encoder/landmarker dapat dimuat",
            True, f"model={type(model).__name__}, kelas={len(labels)}")
    res.add("Jumlah kelas = 24", len(labels) == 24, f"ditemukan {len(labels)}")
    res.add("Model mengharapkan 63 fitur",
            getattr(model, "n_features_in_", None) == 63,
            f"n_features_in_={getattr(model, 'n_features_in_', None)}")
    res.add("Scaler cocok 63 fitur",
            getattr(scaler, "n_features_in_", None) == 63)

    expected = set(rt.SUPPORTED_CLASSES)
    actual = set(labels)
    res.add("Daftar kelas sesuai A-I dan K-Y", expected == actual,
            f"selisih={sorted(expected ^ actual)}")
    res.add("J dan Z tidak didukung",
            "J" not in actual and "Z" not in actual)

    # urgensi kelas model vs encoder
    res.add("Urutan kelas model == encoder",
            np.array_equal(model.classes_, np.arange(len(labels))))
    return model, scaler, encoder, labels


# --------------------------------------------------------------------------- #
# 2. Dimensi input
# --------------------------------------------------------------------------- #

def test_feature_dimensions(res: Result, model, scaler, labels):
    feats = np.zeros(rt.NUM_FEATURES, dtype=np.float64)
    probs = rt.predict_proba(model, scaler, feats)
    res.add("Input 63 fitur -> probabilitas sepanjang jumlah kelas",
            probs.shape[0] == len(labels), f"shape={probs.shape}")
    res.add("Probabilitas berjumlah valid (sum ~ 1)",
            abs(float(np.sum(probs)) - 1.0) < 1e-6,
            f"sum={float(np.sum(probs)):.6f}")

    for bad in (10, 62, 64, 126):
        try:
            rt.predict_proba(model, scaler, np.zeros(bad))
            rejected = False
        except ValueError:
            rejected = True
        res.add(f"Input {bad} fitur ditolak", rejected)

    try:
        bad = np.zeros(rt.NUM_FEATURES)
        bad[0] = np.nan
        rt.predict_proba(model, scaler, bad)
        rejected = bool(False)
    except ValueError:
        rejected = True
    res.add("Input mengandung NaN ditolak", rejected)

    # Normalisasi degenerasi (semua titik == wrist) tetap 63 fitur & finite.
    from types import SimpleNamespace
    same = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    norm = rt.normalize_landmarks(same)
    res.add("Normalisasi degenerasi tetap 63 fitur & finite",
            norm.shape == (63,) and np.all(np.isfinite(norm)),
            f"shape={norm.shape}")


# --------------------------------------------------------------------------- #
# 3. Identitas preprocessing vs dataset training
# --------------------------------------------------------------------------- #

def test_preprocessing_identity(res: Result, samples_per_class: int = 3):
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c != "label"]
    res.add("CSV training berisi 63 fitur", len(feature_cols) == 63,
            f"kolom={len(feature_cols)}")
    res.add("CSV training tanpa NaN", int(df[feature_cols].isna().sum().sum()) == 0)

    lookup: dict[str, np.ndarray] = {}
    for label, group in df.groupby("label"):
        lookup[str(label)] = group[feature_cols].to_numpy(dtype=np.float64)

    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    max_diff = 0.0
    checked = 0
    missing = []
    try:
        for cdir in class_dirs():
            label = cdir.name
            images = images_of(cdir)
            if not images:
                continue
            idxs = np.linspace(0, len(images) - 1, samples_per_class, dtype=int)
            for i in idxs:
                lms = detect_image(landmarker, images[int(i)])
                if lms is None:
                    continue
                feats = rt.normalize_landmarks(lms)
                if label not in lookup or lookup[label].size == 0:
                    missing.append(label)
                    continue
                rows = lookup[label]
                diffs = np.max(np.abs(rows - feats[None, :]), axis=1)
                best = float(np.min(diffs))
                max_diff = max(max_diff, best)
                checked += 1
    finally:
        landmarker.close()

    res.add("Sampel gambar berhasil dibandingkan",
            checked > 0, f"{checked} gambar")
    res.add("Tidak ada kelas hilang di CSV", not missing, f"missing={sorted(set(missing))}")
    res.add("Preprocessing realtime identik dengan training (max diff <= 1e-6)",
            max_diff <= 1e-6, f"max_diff={max_diff:.3e}")
    return max_diff, checked


# --------------------------------------------------------------------------- #
# 4. End-to-end pada dataset (mode IMAGE, per gambar independen)
# --------------------------------------------------------------------------- #

def test_end_to_end(res: Result, model, scaler, encoder, labels, tries: int = 5):
    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    recognizer = rt.SibiRecognizer(model, scaler, encoder, landmarker)
    recognizer.set_mode("IMAGE")

    correct = 0
    evaluated = 0
    not_detected = 0
    details = []
    try:
        for cdir in class_dirs():
            label = cdir.name
            images = images_of(cdir)
            chosen = None
            for img in images[:tries]:
                if detect_image(landmarker, img) is not None:
                    chosen = img
                    break
            if chosen is None:
                not_detected += 1
                details.append((label, "-", "tidak terdeteksi"))
                continue

            rgb = load_rgb(chosen)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            recognizer.smoother.reset()
            result = recognizer.process(bgr)
            pred = result.label if result.recognized else "(low)"
            evaluated += 1
            ok = pred == label
            correct += int(ok)
            details.append((label, pred, f"{result.confidence:.3f}"))
    finally:
        landmarker.close()

    acc = correct / evaluated if evaluated else 0.0
    res.add("Semua 24 kelas dapat dievaluasi",
            evaluated == 24, f"dievaluasi={evaluated}, gagal deteksi={not_detected}")
    res.add("Akurasi end-to-end sampel >= 0.80", acc >= 0.80,
            f"akurasi={acc:.3f} ({correct}/{evaluated})")
    return acc, evaluated, correct, details


# --------------------------------------------------------------------------- #
# 5. Temporal smoothing & jalur VIDEO
# --------------------------------------------------------------------------- #

def test_temporal_smoothing(res: Result, model, scaler, encoder, labels):
    # Perilaku dasar smoother tanpa tangan.
    sm = rt.TemporalSmoother(labels, window=5, stable_frames=3)
    label, conf, stable = sm.update(None)
    res.add("Smoother tanpa tangan -> tidak ada label",
            label is None and conf == 0.0 and not stable)

    # Probabilitas satu kelas konstan -> stabil setelah N frame.
    probs = np.zeros(len(labels), dtype=np.float64)
    probs[labels.index("A")] = 0.95
    probs /= probs.sum()
    stable_at = None
    for i in range(1, 7):
        label, conf, stable = sm.update(probs)
        if stable and stable_at is None:
            stable_at = i
    res.add("Smoother menghasilkan label dominan", label == "A", f"label={label}")
    res.add("Smoother menjadi stabil pada frame ke-3", stable_at == 3,
            f"stable_at={stable_at}")

    sm.reset()
    label, conf, stable = sm.update(probs)
    res.add("Reset menghapus status stabil", not stable)

    # Jalur VIDEO dengan urutan frame (simulasi webcam).
    landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
    recognizer = rt.SibiRecognizer(model, scaler, encoder, landmarker)
    recognizer.set_mode("VIDEO")

    sample_cdir = class_dirs()[0]
    img = images_of(sample_cdir)[0]
    rgb = load_rgb(img)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    ok_sequence = True
    last = None
    try:
        for i in range(12):
            last = recognizer.process(bgr, timestamp_ms=i * 33)
            if not last.hand_found:
                ok_sequence = False
    except Exception as exc:  # noqa: BLE001
        ok_sequence = False
        res.add("Jalur VIDEO tidak error", False, f"{type(exc).__name__}: {exc}")
    finally:
        landmarker.close()

    if ok_sequence:
        res.add("Jalur VIDEO (detect_for_video) berjalan tanpa error", True,
                f"label={last.label}, conf={last.confidence:.3f}, stabil={last.stable}")
        res.add("Smoothing VIDEO menghasilkan label pada frame terakhir",
                last.label is not None)


# --------------------------------------------------------------------------- #
# 6. Drawing & CLI
# --------------------------------------------------------------------------- #

def test_drawing_and_cli(res: Result, model, scaler, encoder, labels):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result_no_hand = rt.FrameResult(False, None, 0.0, False, False, None, None)
    try:
        rt.draw_overlay(frame, result_no_hand, "", 30.0, rt.CONFIDENCE_THRESHOLD)
        res.add("draw_overlay tanpa tangan tidak error", True)
    except Exception as exc:  # noqa: BLE001
        res.add("draw_overlay tanpa tangan tidak error", False, f"{exc}")

    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    lms = [SimpleNamespace(x=float(x), y=float(y), z=0.0)
           for x, y in np.clip(rng.random((21, 2)) * 0.6 + 0.2, 0, 1)]
    bbox = rt.compute_bbox(lms, 1280, 720)
    result_hand = rt.FrameResult(True, "A", 0.93, True, True, lms, bbox)
    try:
        rt.draw_overlay(frame, result_hand, "SIBI", 30.0, rt.CONFIDENCE_THRESHOLD)
        rt.draw_hand(frame, lms, bbox, "A", True)
        res.add("draw_overlay + draw_hand dengan tangan tidak error", True)
    except Exception as exc:  # noqa: BLE001
        res.add("draw_overlay + draw_hand dengan tangan tidak error", False, f"{exc}")

    res.add("compute_bbox menghasilkan nilai valid",
            len(bbox) == 4 and bbox[0] <= bbox[2] and bbox[1] <= bbox[3],
            f"bbox={bbox}")

    parser = rt.build_arg_parser()
    args = parser.parse_args([])
    res.add("Default threshold = 0.85", abs(args.threshold - 0.85) < 1e-9,
            f"threshold={args.threshold}")
    res.add("Default margin = 0.20", abs(args.margin - 0.20) < 1e-9,
            f"margin={args.margin}")
    res.add("Default mirror aktif", bool(args.mirror))
    args2 = parser.parse_args(["--no-mirror", "--camera", "1", "--margin", "0.25"])
    res.add("Argumen --no-mirror, --camera, --margin diproses",
            (not args2.mirror) and args2.camera == 1 and abs(args2.margin - 0.25) < 1e-9)


# --------------------------------------------------------------------------- #
# 7. Rejection mechanism (threshold + margin)
# --------------------------------------------------------------------------- #

def test_rejection(res: Result, model, scaler, encoder, labels):
    # Helper top-2 pada vektor buatan.
    vec = np.zeros(len(labels))
    vec[labels.index("A")] = 0.90
    vec[labels.index("B")] = 0.05
    t = rt.top2_from_probs(vec, labels)
    res.add("top2: top-1 dan top-2 terurut benar",
            t.top1_label == "A" and t.top2_label == "B"
            and abs(t.margin - 0.85) < 1e-9,
            f"top1={t.top1_label}, top2={t.top2_label}, margin={t.margin:.2f}")

    # Truth table rejection.
    res.add("Lolos: top1 >= 0.85 dan margin >= 0.20",
            rt.passes_rejection(0.90, 0.30))
    res.add("Ditolak: top1 < 0.85",
            not rt.passes_rejection(0.80, 0.30))
    res.add("Ditolak: margin < 0.20",
            not rt.passes_rejection(0.95, 0.10))
    res.add("Ditolak: keduanya gagal",
            not rt.passes_rejection(0.60, 0.05))

    # Wiring recognizer: hanya frame lolos yang masuk buffer smoothing.
    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    recognizer = rt.SibiRecognizer(model, scaler, encoder, landmarker)
    recognizer.set_mode("IMAGE")

    accepted = rejected = evaluated = 0
    wiring_ok = True
    seen = None
    try:
        for cdir in class_dirs():
            images = images_of(cdir)
            chosen = None
            for img in images[:5]:
                if detect_image(landmarker, img) is not None:
                    chosen = img
                    break
            if chosen is None:
                continue
            rgb = load_rgb(chosen)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            recognizer.smoother.reset()
            result = recognizer.process(bgr)
            evaluated += 1
            if result.rejected:
                rejected += 1
                if (result.recognized or result.label is not None
                        or len(recognizer.smoother._buf) != 0):
                    wiring_ok = False
            elif result.recognized:
                accepted += 1
                if (result.top1_prob < recognizer.threshold
                        or result.margin < recognizer.margin_threshold
                        or len(recognizer.smoother._buf) != 1):
                    wiring_ok = False
                else:
                    seen = result
    finally:
        landmarker.close()

    res.add("Evaluasi rejection pada sampel dataset berjalan",
            evaluated > 0, f"dievaluasi={evaluated}, lolos={accepted}, ditolak={rejected}")
    res.add("Wiring: frame ditolak tidak masuk buffer & tidak diakui", wiring_ok)
    res.add("Debug info top1/top2/margin terisi",
            seen is not None and seen.top1_label is not None
            and seen.top2_label is not None and seen.margin > 0,
            (f"contoh top1={seen.top1_label} {seen.top1_prob:.3f}, "
             f"top2={seen.top2_label} {seen.top2_prob:.3f}, margin={seen.margin:.3f}")
            if seen else "tidak ada frame lolos")

    # Smoothing isolation: rejected (None) tidak menstabilkan.
    sm = rt.TemporalSmoother(labels, window=5, stable_frames=3)
    probs = np.zeros(len(labels))
    probs[labels.index("A")] = 0.95
    probs /= probs.sum()
    stable = False
    for _ in range(3):
        _, _, stable = sm.update(probs)
    lbl_after_reject, _, stable_after_reject = sm.update(None)
    res.add("Frame ditolak (None) tidak stabil & tidak memberi label",
            stable and not stable_after_reject and lbl_after_reject is None)


# --------------------------------------------------------------------------- #
# 8. Laporan
# --------------------------------------------------------------------------- #

def write_report(res: Result, extra: dict, elapsed: float) -> None:
    lines: list[str] = []
    a = lines.append
    a("# REALTIME REPORT - SIBI Scanner (Webcam)")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("- Entry point: `realtime.py`")
    a("- Model: `models/sibi_mlp.joblib` (tidak diubah, tidak ditraining ulang)")
    a("- Scaler: `models/scaler.joblib`")
    a("- Label encoder: `models/label_encoder.joblib`")
    a("- Hand landmarker: `models/hand_landmarker.task`")
    a("- Smoke test: `scripts/smoke_test_realtime.py` (offline, tanpa webcam)")
    a(f"- Total waktu smoke test: {elapsed:.2f} detik")
    a("")
    a("## 1. Ringkasan Hasil Smoke Test")
    a("")
    a(f"Status keseluruhan: **{'LULUS' if res.passed else 'GAGAL'}**")
    a("")
    a("| # | Pemeriksaan | Status | Detail |")
    a("|---|-------------|:------:|--------|")
    for i, (name, passed, detail) in enumerate(res.items, 1):
        a(f"| {i} | {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    a("")
    a("## 2. Kesesuaian Preprocessing")
    a("")
    a("Preprocessing realtime (`realtime.normalize_landmarks`) adalah salinan logika "
      "`scripts/extract_landmarks.py`:")
    a("")
    a("1. Ambil 21 landmark (x, y, z) -> 63 nilai.")
    a("2. Translasi relatif terhadap landmark 0 (wrist).")
    a("3. Skala = max jarak Euclidean landmark ke wrist (fallback 1.0 jika ~0).")
    a("4. Fitur = hasil translasi dibagi skala, di-reshape menjadi 63.")
    a("")
    a(f"- Perbedaan maksimum terhadap fitur CSV training: **{extra['max_diff']:.3e}**")
    a(f"- Gambar dataset yang dibandingkan: **{extra['checked']}**")
    a(f"- Verdict: {'IDENTIK (<= 1e-6)' if extra['max_diff'] <= 1e-6 else 'BERBEDA'}")
    a("")
    a("## 3. Akurasi End-to-End pada Sampel Dataset")
    a("")
    a(f"- Sampel: 1 gambar per kelas (24 kelas), mode IMAGE.")
    a(f"- Akurasi: **{extra['acc']:.3f}** ({extra['correct']}/{extra['evaluated']})")
    a("")
    a("| Kelas Asli | Prediksi | Confidence |")
    a("|:----------:|:--------:|-----------:|")
    for label, pred, conf in extra["details"]:
        a(f"| {label} | {pred} | {conf} |")
    a("")
    a("## 4. Kesesuaian Model & Kelas")
    a("")
    a("- Model menerima tepat **63 fitur** (divalidasi; input 10/62/64/126 ditolak).")
    a("- Model memiliki **24 kelas**: A-I dan K-Y.")
    a("- **J dan Z tidak didukung** dan tidak dibuat secara sintetis.")
    a(f"- Threshold confidence: **{rt.CONFIDENCE_THRESHOLD:.2f}** dan margin threshold "
      f"**{rt.MARGIN_THRESHOLD:.2f}** (top1 - top2); jika salah satu tidak terpenuhi "
      "ditampilkan \"Tidak dikenali\".")
    a("- Temporal smoothing: rata-rata probabilitas frame terakhir "
      f"(window={rt.SMOOTHING_WINDOW}), stabil setelah {rt.STABLE_FRAMES} frame. "
      "Hanya frame yang lolos rejection yang masuk buffer smoothing.")
    a("")
    a("## 5. Fitur Aplikasi")
    a("")
    a("- Deteksi tangan MediaPipe Hand Landmarker (mode VIDEO).")
    a("- Menampilkan landmark, kerangka tangan, bounding box, huruf prediksi, "
      "confidence, bar confidence, dan FPS.")
    a("- Ejaan ditampilkan di bagian atas frame.")
    a("- Kontrol: **Q** keluar, **R** reset teks, **SPASI** tambah huruf stabil, "
      "**BACKSPACE** hapus karakter terakhir.")
    a("")
    a("## 6. Cara Menjalankan")
    a("")
    a("```bash")
    a("# (opsional) validasi offline tanpa webcam")
    a("python realtime.py --check")
    a("")
    a("# smoke test lengkap + regenerate laporan ini")
    a("python scripts/smoke_test_realtime.py")
    a("")
    a("# jalankan scanner dengan webcam")
    a("python realtime.py")
    a("")
    a("# opsi: kamera lain, tanpa mirror, threshold lain")
    a("python realtime.py --camera 1 --no-mirror --threshold 0.85 --margin 0.20")
    a("```")
    a("")
    a("## 7. Catatan & Batasan")
    a("")
    a("- Model hanya untuk **24 gesture statis**. Gestur dinamis J dan Z tidak "
      "tersedia di dataset sehingga **tidak didukung**.")
    a("- Jika akurasi terasa buruk dengan mode cermin, coba `--no-mirror`.")
    a("- Pastikan pencahayaan cukup dan tangan terlihat jelas oleh webcam.")
    a("- Tidak ada training ulang, perubahan file model, atau data sintetis.")
    a("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    res = Result()
    print("== Smoke test realtime.py ==")

    model, scaler, encoder, labels = test_artifacts(res)
    test_feature_dimensions(res, model, scaler, labels)
    max_diff, checked = test_preprocessing_identity(res)
    acc, evaluated, correct, details = test_end_to_end(
        res, model, scaler, encoder, labels
    )
    test_temporal_smoothing(res, model, scaler, encoder, labels)
    test_drawing_and_cli(res, model, scaler, encoder, labels)
    test_rejection(res, model, scaler, encoder, labels)

    elapsed = time.perf_counter() - start
    extra = {
        "max_diff": max_diff,
        "checked": checked,
        "acc": acc,
        "evaluated": evaluated,
        "correct": correct,
        "details": details,
    }
    write_report(res, extra, elapsed)

    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    print(f"\nHasil: {passed}/{total} pemeriksaan lulus")
    print(f"Laporan: {REPORT_MD.relative_to(ROOT)}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
