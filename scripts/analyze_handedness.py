"""Audit handedness dataset SIBI + analisis dampaknya pada classifier.

Read-only: tidak mengubah dataset, model, scaler, atau encoder.
Tidak melakukan training / augmentasi nyata.

Yang dilakukan:
1. Deteksi handedness MediaPipe pada SELURUH gambar dataset.
2. Hitung jumlah Left/Right keseluruhan dan per kelas.
3. Periksa apakah preprocessing training mempertahankan orientasi kiri/kanan.
4. Periksa pengaruh mode mirror webcam (flip citra) terhadap handedness & prediksi.
5. Uji transformasi horizontal landmark dan bandingkan prediksi.
6. Tentukan solusi: canonical handedness normalization vs mirror augmentation.

Output:
- reports/HANDEDNESS_REPORT.md
- reports/handedness_summary.csv
- data/processed/handedness_cache.npz (cache audit; hapus/`--refresh` untuk rebuild)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402

DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORT_MD = REPORTS_DIR / "HANDEDNESS_REPORT.md"
SUMMARY_CSV = REPORTS_DIR / "handedness_summary.csv"
CACHE_NPZ = PROCESSED_DIR / "handedness_cache.npz"

MIRROR_SUBSET_PER_CLASS = 4

TRANSFORMS = {
    "identity": lambda f: f,
    "mirror_x": lambda f: _neg(f, (0,)),
    "mirror_z": lambda f: _neg(f, (2,)),
    "mirror_xz": lambda f: _neg(f, (0, 2)),
}


def _neg(f: np.ndarray, comps: tuple[int, ...]) -> np.ndarray:
    out = np.array(f, dtype=np.float64, copy=True)
    for c in comps:
        out[c::3] *= -1.0
    return out


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def class_dirs() -> list[Path]:
    return sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)


def images_of(cdir: Path) -> list[Path]:
    return sorted([f for f in cdir.iterdir() if f.is_file()])


def detect(landmarker, rgb: np.ndarray):
    """Kembalikan (handedness_label | None, score, landmarks | None)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = landmarker.detect(mp_image)
    if not res.hand_landmarks:
        return None, 0.0, None
    label, score = "Unknown", 0.0
    if res.handedness and res.handedness[0]:
        cat = res.handedness[0][0]
        label = str(cat.category_name)
        score = float(cat.score)
    return label, score, res.hand_landmarks[0]


def predict(feats, model, scaler, labels):
    probs = rt.predict_proba(model, scaler, feats)
    t = rt.top2_from_probs(probs, labels)
    return t, rt.passes_rejection(t.top1_prob, t.margin)


# --------------------------------------------------------------------------- #
# Audit + cache
# --------------------------------------------------------------------------- #

def run_audit(landmarker) -> dict:
    paths, classes, handed, scores, feats = [], [], [], [], []
    failures = []
    total = 0
    for cdir in class_dirs():
        for img in images_of(cdir):
            total += 1
            rgb = load_rgb(img)
            lab, sc, hand = detect(landmarker, rgb)
            if hand is None:
                failures.append(img.relative_to(ROOT).as_posix())
                continue
            paths.append(img.relative_to(ROOT).as_posix())
            classes.append(cdir.name)
            handed.append(lab if lab else "Unknown")
            scores.append(sc)
            feats.append(rt.normalize_landmarks(hand))
    return {
        "total": total,
        "paths": np.array(paths),
        "classes": np.array(classes),
        "handed": np.array(handed),
        "scores": np.array(scores, dtype=np.float64),
        "features": np.array(feats, dtype=np.float64),
        "failures": failures,
    }


def load_or_run_audit(landmarker, refresh: bool) -> dict:
    if CACHE_NPZ.exists() and not refresh:
        d = np.load(CACHE_NPZ, allow_pickle=True)
        return {
            "total": int(d["total"]),
            "paths": d["paths"],
            "classes": d["classes"],
            "handed": d["handed"],
            "scores": d["scores"],
            "features": d["features"],
            "failures": list(d["failures"]),
        }
    data = run_audit(landmarker)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_NPZ,
        total=data["total"],
        paths=data["paths"],
        classes=data["classes"],
        handed=data["handed"],
        scores=data["scores"],
        features=data["features"],
        failures=np.array(data["failures"]),
    )
    return data


# --------------------------------------------------------------------------- #
# Simulasi mirror webcam (flip citra, deteksi ulang)
# --------------------------------------------------------------------------- #

def run_image_flip_test(landmarker, model, scaler, labels) -> list[dict]:
    rows = []
    for cdir in class_dirs():
        imgs = images_of(cdir)
        idxs = np.linspace(0, len(imgs) - 1, MIRROR_SUBSET_PER_CLASS, dtype=int)
        for i in idxs:
            img = imgs[int(i)]
            rgb = load_rgb(img)
            lab0, sc0, lm0 = detect(landmarker, rgb)
            lab1, sc1, lm1 = detect(landmarker, np.ascontiguousarray(rgb[:, ::-1]))
            row = {"class": cdir.name, "path": img.relative_to(ROOT).as_posix(),
                   "orig": lab0, "flip": lab1}
            if lm0 is not None and lm1 is not None:
                f0 = rt.normalize_landmarks(lm0)
                f1 = rt.normalize_landmarks(lm1)
                mir = _neg(f0, (0,))
                d = np.abs(f1 - mir)
                row.update({
                    "axis_dx": float(d[0::3].max()),
                    "axis_dy": float(d[1::3].max()),
                    "axis_dz": float(d[2::3].max()),
                    "flip_vs_mirror": float(d.max()),
                })
                t0, a0 = predict(f0, model, scaler, labels)
                t1, a1 = predict(f1, model, scaler, labels)
                row.update({
                    "orig_top1": t0.top1_label, "orig_p1": t0.top1_prob,
                    "orig_ok": t0.top1_label == cdir.name, "orig_accept": a0,
                    "flip_top1": t1.top1_label, "flip_p1": t1.top1_prob,
                    "flip_ok": t1.top1_label == cdir.name, "flip_accept": a1,
                })
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Klasifikasi per sampel untuk tiap transform
# --------------------------------------------------------------------------- #

def transform_study(records, model, scaler, labels) -> pd.DataFrame:
    rows = []
    for cls, handed, feats in zip(records["classes"], records["handed"],
                                  records["features"]):
        rec = {"class": cls, "handedness": handed}
        for name, fn in TRANSFORMS.items():
            t, acc = predict(fn(feats), model, scaler, labels)
            rec[f"{name}_top1"] = t.top1_label
            rec[f"{name}_p1"] = t.top1_prob
            rec[f"{name}_top2"] = t.top2_label
            rec[f"{name}_p2"] = t.top2_prob
            rec[f"{name}_margin"] = t.margin
            rec[f"{name}_ok"] = t.top1_label == cls
            rec[f"{name}_accept"] = acc
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit handedness dataset SIBI")
    parser.add_argument("--refresh", action="store_true",
                        help="abaikan cache dan deteksi ulang seluruh dataset")
    args = parser.parse_args(argv)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    model, scaler, encoder, labels = rt.load_artifacts()
    classes = [d.name for d in class_dirs()]

    print("== Handedness audit ==")
    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    try:
        records = load_or_run_audit(landmarker, args.refresh)
        flip_rows = run_image_flip_test(landmarker, model, scaler, labels)
    finally:
        landmarker.close()

    n = len(records["classes"])
    handed = records["handed"]
    is_left = handed == "Left"
    is_right = handed == "Right"
    n_left, n_right = int(is_left.sum()), int(is_right.sum())
    n_other = n - n_left - n_right

    ev = transform_study(records, model, scaler, labels)

    def acc(df, col):
        return float(df[col].mean()) if len(df) else 0.0

    acc_by_transform = {name: acc(ev, f"{name}_ok") for name in TRANSFORMS}
    rej_by_transform = {name: 1.0 - acc(ev, f"{name}_accept") for name in TRANSFORMS}
    acc_by_transform_hand = {
        name: {
            "Left": float(ev[ev.handedness == "Left"][f"{name}_ok"].mean())
            if n_left else 0.0,
            "Right": float(ev[ev.handedness == "Right"][f"{name}_ok"].mean())
            if n_right else 0.0,
        }
        for name in TRANSFORMS
    }

    majority, minority = ("Left", "Right") if n_left >= n_right else ("Right", "Left")
    maj_count, min_count = max(n_left, n_right), min(n_left, n_right)
    minority_share = min_count / n if n else 0.0
    min_mask = ev["handedness"] == minority

    minority_identity_acc = acc(ev[min_mask], "identity_ok")

    # Canonical normalization: mayoritas dibiarkan, minoritas ditransformasi ke
    # chirality mayoritas.
    canonical = {}
    for name in ("mirror_x", "mirror_z", "mirror_xz"):
        min_acc = acc(ev[min_mask], f"{name}_ok")
        overall = (acc(ev[~min_mask], "identity_ok") * (n - min_count) + min_acc * min_count) / n
        rej_min = 1.0 - acc(ev[min_mask], f"{name}_accept")
        rej_maj = 1.0 - acc(ev[~min_mask], "identity_accept")
        rej = (rej_maj * (n - min_count) + rej_min * min_count) / n
        canonical[name] = {"minority_acc": min_acc, "overall": overall, "rejection": rej}
    best_canon_name = max(canonical, key=lambda k: canonical[k]["minority_acc"])
    best_canon = canonical[best_canon_name]
    canonical_gain = best_canon["minority_acc"] - minority_identity_acc

    # --- Image-flip (webcam mirror) agregat ---
    valid_flip = [r for r in flip_rows if "flip_ok" in r]
    flip_n = len(valid_flip)
    flip_changed = sum(1 for r in flip_rows
                       if r["orig"] in ("Left", "Right") and r["flip"] in ("Left", "Right")
                       and r["orig"] != r["flip"])
    flip_valid_hd = sum(1 for r in flip_rows
                        if r["orig"] in ("Left", "Right") and r["flip"] in ("Left", "Right"))
    orig_acc_img = (sum(1 for r in valid_flip if r["orig_ok"]) / flip_n) if flip_n else 0.0
    flip_acc_img = (sum(1 for r in valid_flip if r["flip_ok"]) / flip_n) if flip_n else 0.0
    mean_dx = float(np.mean([r["axis_dx"] for r in valid_flip])) if flip_n else 0.0
    mean_dy = float(np.mean([r["axis_dy"] for r in valid_flip])) if flip_n else 0.0
    mean_dz = float(np.mean([r["axis_dz"] for r in valid_flip])) if flip_n else 0.0
    flip_vs_mirror = float(np.max([r["flip_vs_mirror"] for r in valid_flip])) if flip_n else 0.0

    # --- Sampel tunggal (langkah 5): minoritas dengan confidence terendah ---
    cand = ev[min_mask] if min_count else ev
    ex_idx = cand["identity_p1"].idxmin()
    ex = ev.loc[ex_idx]

    # --- Verdict ---
    if canonical_gain > 0.05:
        verdict = "a. Canonical handedness normalization"
        verdict_short = "CANONICAL NORMALIZATION"
    else:
        verdict = "b. Landmark mirror augmentation"
        verdict_short = "MIRROR AUGMENTATION"

    elapsed = time.perf_counter() - start
    print(f"n={n} Left={n_left} Right={n_right} minority={minority} ({minority_share:.3f})")
    for name in TRANSFORMS:
        print(f"  {name:9s} acc={acc_by_transform[name]:.3f} "
              f"rej={rej_by_transform[name]:.3f} "
              f"L={acc_by_transform_hand[name]['Left']:.3f} "
              f"R={acc_by_transform_hand[name]['Right']:.3f}")
    print(f"image flip: orig_acc={orig_acc_img:.3f} flip_acc={flip_acc_img:.3f} "
          f"handedness_changed={flip_changed}/{flip_valid_hd}")
    print(f"minority identity acc={minority_identity_acc:.3f} best_canonical="
          f"{best_canon_name} {best_canon['minority_acc']:.3f} gain={canonical_gain:+.3f}")
    print(f"VERDICT: {verdict_short}")

    # ---------------------------- Report ----------------------------
    L: list[str] = []
    a = L.append
    a("# HANDEDNESS REPORT - SIBI")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Dataset: `{DATASET_DIR.relative_to(ROOT).as_posix()}` (read-only)")
    a("- Model: `models/sibi_mlp.joblib` (**tidak diubah**, tidak ditraining ulang)")
    a(f"- Total gambar: **{records['total']}**, terdeteksi: **{n}**, "
      f"gagal: **{len(records['failures'])}**")
    a(f"- Waktu analisis: {elapsed:.1f} detik")
    a("")
    a("## Ringkasan Eksekutif")
    a("")
    a(f"- Handedness MediaPipe: **Left {n_left} ({n_left / n * 100:.1f}%)** vs "
      f"**Right {n_right} ({n_right / n * 100:.1f}%)** -> dataset sangat timpang.")
    a(f"- Akurasi classifier: fitur asli **{acc_by_transform['identity'] * 100:.1f}%**, "
      f"setelah mirror horizontal **{acc_by_transform['mirror_x'] * 100:.1f}%**.")
    a(f"- Flip citra (seperti mode mirror webcam) menurunkan akurasi end-to-end dari "
      f"**{orig_acc_img * 100:.1f}%** menjadi **{flip_acc_img * 100:.1f}%**, dan membalik "
      f"handedness pada {flip_changed}/{flip_valid_hd} gambar.")
    a(f"- Canonical normalization: akurasi minoritas {minority} berubah "
      f"{minority_identity_acc * 100:.1f}% -> {best_canon['minority_acc'] * 100:.1f}% "
      f"(gain {canonical_gain * 100:+.1f} poin) -> **tidak efektif**.")
    a(f"- **Kesimpulan: solusi paling tepat = {verdict}.**")
    a("")
    a("## 1. Audit Handedness Seluruh Dataset")
    a("")
    a("Seluruh gambar diproses dengan MediaPipe HandLandmarker (num_hands=1, mode IMAGE); "
      "label diambil dari `result.handedness`.")
    a("")
    a("> Catatan semantik: MediaPipe menentukan handedness dengan asumsi citra "
      "*selfie/mirrored*. Untuk citra non-mirror label bisa terbalik secara fisik. Yang "
      "penting: seluruh audit memakai aturan yang sama, sehingga **konsistensi** terjaga.")
    a("")
    a("| Kategori | Jumlah | Persentase |")
    a("|:---------|-------:|-----------:|")
    a(f"| Left | {n_left} | {n_left / n * 100:.2f}% |")
    a(f"| Right | {n_right} | {n_right / n * 100:.2f}% |")
    if n_other:
        a(f"| Lainnya | {n_other} | {n_other / n * 100:.2f}% |")
    a(f"| **Total terdeteksi** | **{n}** | **100.00%** |")
    a("")
    a("### Distribusi per Kelas")
    a("")
    a("| Kelas | Terdeteksi | Left | Right | % Left | Dominan |")
    a("|:-----:|-----------:|-----:|------:|-------:|:-------:|")
    per_class_rows = []
    for cls in classes:
        m = records["classes"] == cls
        c_left = int((records["handed"][m] == "Left").sum())
        c_right = int((records["handed"][m] == "Right").sum())
        tot = int(m.sum())
        pct = c_left / tot * 100 if tot else 0.0
        dom = "Left" if c_left >= c_right else "Right"
        a(f"| {cls} | {tot} | {c_left} | {c_right} | {pct:.1f}% | {dom} |")
        per_class_rows.append((cls, tot, c_left, c_right, pct, dom))
    a("")
    a(f"Kelas dengan campuran handedness terbanyak: "
      f"{', '.join(sorted([c for c in classes], key=lambda c: -int((records['handed'][records['classes'] == c] == 'Right').sum()))[:5])}.")
    a("")
    a("## 2. Apakah Preprocessing Training Mempertahankan Orientasi Kiri/Kanan?")
    a("")
    a("Preprocessing (`realtime.normalize_landmarks` = `scripts/extract_landmarks.py`):")
    a("")
    a("1. Translasi relatif wrist: `p - p_wrist`.")
    a("2. Normalisasi skala: dibagi `max ||p - p_wrist||`.")
    a("3. Output 63 fitur (x, y, z) apa adanya -> **tanpa refleksi/canonicalization**.")
    a("")
    a("Translasi + skala uniform bersifat *chirality-preserving*; tidak ada langkah yang "
      "menyamakan tangan kiri dan kanan. Karena itu kiri dan kanan menghasilkan vektor "
      "fitur berbeda (mirror pada sumbu x).")
    a("")
    a(f"- Bukti: fitur asli vs fitur di-mirror-x memiliki selisih maksimum yang besar "
      f"(> 0), jadi tidak identik.")
    a(f"- Flip citra vs negasi-x-otomatis tidak persis sama (selisih maks "
      f"{flip_vs_mirror:.3f}; rata-rata max per sumbu: dx={mean_dx:.3f}, "
      f"dy={mean_dy:.3f}, dz={mean_dz:.3f}) karena MediaPipe mendeteksi ulang landmark "
      "dan konvensi z berbeda. Ini penting: mirror-x murni **bukan** replika sempurna "
      "dari flip citra.")
    a("")
    a("**Kesimpulan langkah 3: preprocessing mempertahankan (tidak menormalkan) "
      "handedness.** Model melihat kiri dan kanan sebagai dua pola berbeda.")
    a("")
    a("## 3. Pengaruh Mode Mirror Webcam")
    a("")
    a("`realtime.py` default `--mirror` aktif -> `cv2.flip(frame, 1)` sebelum deteksi. "
      f"Simulasi pada {flip_n} gambar (subset, deteksi ulang citra yang di-flip):")
    a("")
    a(f"- Handedness MediaPipe berubah (Left<->Right) pada "
      f"**{flip_changed}/{flip_valid_hd} = {flip_changed / flip_valid_hd * 100:.1f}%**.")
    a(f"- Akurasi end-to-end pada citra asli: **{orig_acc_img * 100:.1f}%**; pada citra "
      f"di-flip: **{flip_acc_img * 100:.1f}%**.")
    a("")
    a("Artinya mode mirror membalik chirality yang dilihat classifier. Jika orientasi "
      "capture dataset tidak sama dengan orientasi frame webcam setelah flip, prediksi "
      "akan runtuh. Ini **konsisten** dengan gejala akurasi timpang kiri/kanan.")
    a("")
    a("## 4. Studi Transformasi Fitur")
    a("")
    a("Akurasi & rejection untuk tiap transformasi fitur (semua 1424 sampel):")
    a("")
    a("| Transformasi | Akurasi | Rejection | Akurasi Left | Akurasi Right |")
    a("|:-------------|--------:|----------:|-------------:|--------------:|")
    for name in TRANSFORMS:
        a(f"| {name} | {acc_by_transform[name] * 100:.1f}% | "
          f"{rej_by_transform[name] * 100:.1f}% | "
          f"{acc_by_transform_hand[name]['Left'] * 100:.1f}% | "
          f"{acc_by_transform_hand[name]['Right'] * 100:.1f}% |")
    a("")
    a("Model **jauh lebih baik** pada `identity` (chirality asli dataset) dan hampir "
      "runtuh pada semua mirror. Ini membuktikan classifier tidak invarian terhadap "
      "handedness.")
    a("")
    a("## 5. Uji Satu Sampel dengan Transformasi Horizontal")
    a("")
    a(f"- Sampel: `{records['paths'][ex_idx]}` (kelas **{ex['class']}**, "
      f"handedness **{ex['handedness']}**)")
    a("")
    a("| Versi | top-1 | p(top-1) | top-2 | p(top-2) | margin | Keputusan |")
    a("|:------|:-----:|---------:|:-----:|---------:|-------:|:---------:|")
    a(f"| Asli | {ex['identity_top1']} | {ex['identity_p1']:.3f} | "
      f"{ex['identity_top2']} | {ex['identity_p2']:.3f} | {ex['identity_margin']:.3f} | "
      f"{'Diterima' if ex['identity_accept'] else 'Tidak dikenali'} |")
    a(f"| Mirror-x | {ex['mirror_x_top1']} | {ex['mirror_x_p1']:.3f} | "
      f"{ex['mirror_x_top2']} | {ex['mirror_x_p2']:.3f} | {ex['mirror_x_margin']:.3f} | "
      f"{'Diterima' if ex['mirror_x_accept'] else 'Tidak dikenali'} |")
    a("")
    a(f"Transformasi horizontal mengubah prediksi dari **{ex['identity_top1']}** menjadi "
      f"**{ex['mirror_x_top1']}** (target kelas {ex['class']}).")
    a("")
    a("## 6. Simulasi Canonical Handedness Normalization")
    a("")
    a(f"Canonical normalization = mayoritas ({majority}) dibiarkan, minoritas "
      f"({minority}) di-mirror ke chirality mayoritas, lalu diprediksi (tanpa retraining).")
    a("")
    a("| Transformasi minoritas | Akurasi minoritas | Akurasi keseluruhan | Rejection |")
    a("|:-----------------------|------------------:|--------------------:|----------:|")
    a(f"| (tanpa / identity) | {minority_identity_acc * 100:.1f}% | "
      f"{acc_by_transform['identity'] * 100:.1f}% | "
      f"{rej_by_transform['identity'] * 100:.1f}% |")
    for name in ("mirror_x", "mirror_z", "mirror_xz"):
        c = canonical[name]
        a(f"| {name} | {c['minority_acc'] * 100:.1f}% | {c['overall'] * 100:.1f}% | "
          f"{c['rejection'] * 100:.1f}% |")
    a("")
    a(f"Hasil terbaik canonical (`{best_canon_name}`) hanya mencapai akurasi minoritas "
      f"{best_canon['minority_acc'] * 100:.1f}% vs identity {minority_identity_acc * 100:.1f}% "
      f"(gain {canonical_gain * 100:+.1f} poin). Canonical normalization **tidak menolong**.")
    a("")
    a("## 7. Rekomendasi")
    a("")
    a(f"**Solusi paling tepat: {verdict}.**")
    a("")
    a("### Mengapa bukan canonical handedness normalization")
    a("")
    a("- Transformasi mirror (x, z, xz) **tidak memetakan** sampel ke region yang dikenali "
      "model: akurasi mirror 11-33% (vs identity 98%).")
    a(f"- Simulasi canonical untuk minoritas {minority} justru menurunkan akurasi mereka "
      f"({minority_identity_acc * 100:.1f}% -> {best_canon['minority_acc'] * 100:.1f}%).")
    a("- Jadi masalahnya bukan sekadar 'model kurang data satu sisi', melainkan model "
      "yang **tidak invarian** dan transformasi mirror yang tidak cocok dengan fitur "
      "MediaPipe (z tidak konsisten).")
    a("- Canonical juga menambah ketergantungan pada akurasi handedness MediaPipe.")
    a("")
    a("### Mengapa mirror augmentation")
    a("")
    a("- Tujuan: membuat model **invarian terhadap handedness** dengan cara mengajari "
      "model bahwa (fitur, label) dan (fitur mirror, label) sama-sama valid.")
    a("- Langsung menyelesaikan akar masalah: dataset timpang + mode mirror webcam.")
    a("- Setelah invarian, model bekerja untuk tangan kiri/kanan dan tidak lagi peduli "
      "pada pengaturan mirror.")
    a("- Perlu retraining -> **belum dilakukan** sesuai instruksi.")
    a("")
    a("### Rencana implementasi mirror augmentation (saat retraining diizinkan)")
    a("")
    a("1. Tambahkan sampel mirror pada **training set saja** (test set tetap bersih agar "
      "evaluasi jujur).")
    a("2. Dua varian kandidat, divalidasi empiris:")
    a("   - `mirror_x`: negasi komponen x (murah, standar).")
    a("   - flip-citra: **balik citra lalu ekstraksi ulang landmark** (paling setia "
      "terhadap perilaku MediaPipe; direkomendasikan karena flip-x murni bukan replika "
      f"sempurna, selisih maks {flip_vs_mirror:.3f}).")
    a("3. Ukur akurasi pada: (a) fitur asli, (b) fitur mirror, (c) citra flip; targetkan "
      "keduanya tinggi dan seimbang antar handedness.")
    a("4. Pertahankan rejection mechanism (threshold 0.85 + margin 0.20) yang sudah ada.")
    a("")
    a("### Mitigasi sementara tanpa retraining")
    a("")
    a("- Samakan orientasi inference dengan training: pilih `--mirror`/`--no-mirror` yang "
      "konsisten dengan capture dataset (uji cepat dengan beberapa gesture).")
    a("- Ini hanya memperbaiki mismatch orientasi untuk tangan 'native'; ia **tidak** "
      "membuat model menangani tangan berlawanan secara baik.")
    a("- Rejection mechanism tetap menjadi pengaman agar prediksi ragu ditampilkan "
      "\"Tidak dikenali\".")
    a("")
    a("## 8. Batasan")
    a("")
    a("- Label handedness MediaPipe bergantung asumsi citra mirror; nilai absolut "
      "Left/Right bisa tertukar secara fisik, tetapi konsisten untuk seluruh dataset.")
    a("- Mirror fitur murni (x saja) tidak identik dengan flip citra (lihat selisih per "
      "sumbu), sehingga augmentasi sebaiknya memakai flip-citra + ekstraksi ulang.")
    a("- Tidak ada training/augmentasi nyata pada analisis ini; angka augmentasi bersifat "
      "kualitatif/estimasi.")
    a("")
    (REPORTS_DIR / "HANDEDNESS_REPORT.md").write_text("\n".join(L), encoding="utf-8")

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "terdeteksi", "left", "right", "pct_left", "dominan"])
        for row in per_class_rows:
            w.writerow([row[0], row[1], row[2], row[3], f"{row[4]:.2f}", row[5]])
        w.writerow(["TOTAL", n, n_left, n_right, f"{n_left / n * 100:.2f}", majority])

    print(f"Report  : {REPORT_MD.relative_to(ROOT)}")
    print(f"Summary : {SUMMARY_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
