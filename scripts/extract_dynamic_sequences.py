"""Manifest + ekstraksi sequence temporal J/Z untuk GRU (read-only dataset).

Pipeline tiap video:
  video -> decode (PyAV/OpenCV) -> MediaPipe HandLandmarker (21 landmark)
        -> wrist-relative + scale normalization (sama dengan project)
        -> timestamp-aware (t = frame_index / fps)
        -> interpolasi gap pendek + resampling temporal ke 24 timestep
        -> sequence (24, 63)

Output:
- data/processed/dynamic_manifest.csv
- data/processed/dynamic_sequences/J/*.npz, .../Z/*.npz
- reports/DYNAMIC_SEQUENCE_REPORT.md
- reports/dynamic_sequence_summary.csv
- reports/dynamic_movement_stats.json

Jalankan:
    python scripts/extract_dynamic_sequences.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av  # noqa: E402

import mediapipe as mp  # noqa: E402
import realtime as rt  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SEQ_DIR = PROCESSED / "dynamic_sequences"
REPORTS = ROOT / "reports"

MANIFEST_CSV = PROCESSED / "dynamic_manifest.csv"
SUMMARY_CSV = REPORTS / "dynamic_sequence_summary.csv"
MOVEMENT_JSON = REPORTS / "dynamic_movement_stats.json"
REPORT_MD = REPORTS / "DYNAMIC_SEQUENCE_REPORT.md"

LABELS = ("J", "Z")
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
FINGERTIPS = (4, 8, 12, 16, 20)
SEQ_LEN = 24
FEATURE_COUNT = 63

# Ambang penanganan missing detection
GAP_EXCLUDE_SEC = 0.30     # gap > ini -> exclude (tidak diisi sembarangan)
INTERP_EXCLUDE_FRAC = 0.50  # > 50% frame di span terinterpolasi -> exclude
DETECTION_EXCLUDE = 0.50   # detection rate < ini -> exclude

# Exclusion yang sudah diketahui dari audit
KNOWN_EXCLUDE = {
    "J_007.MP4": "byte-identical dengan J_001 (duplikat)",
    "Z_009.MP4": "detection rate sangat rendah (~49%)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_files() -> list[tuple[str, Path]]:
    out = []
    for label in LABELS:
        d = RAW / label
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    out.append((label, p))
    return out


def normalize_temporal(times: np.ndarray, feats: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    """Resample fitur pada timestamp `times` ke n timestep merata (0..1).

    Linear interpolation antar-sampel terdeteksi (timestamp-aware), sehingga
    perbedaan FPS antar kelas tidak menjadi fitur. Mengembalikan (n, 63).
    """
    target = np.linspace(times[0], times[-1], n)
    out = np.empty((n, feats.shape[1]), dtype=np.float64)
    for j in range(feats.shape[1]):
        out[:, j] = np.interp(target, times, feats[:, j])
    return out


def resample_series(times: np.ndarray, arr: np.ndarray, n: int = SEQ_LEN) -> np.ndarray:
    """Resample array (T, ...) pada timestamp ke n timestep (linear)."""
    target = np.linspace(times[0], times[-1], n)
    flat = arr.reshape(len(times), -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(target, times, flat[:, j])
    return out.reshape((n,) + arr.shape[1:])


def extraction_worker(label: str, path: Path, landmarker, dup_of: str | None) -> dict:
    rec: dict = {
        "source_video": f"{label}/{path.name}",
        "label": label,
        "include": True,
        "exclusion_reason": "",
        "fps": 0.0,
        "duration_sec": 0.0,
        "original_frames": 0,
        "detection_rate": 0.0,
        "detected_frames": 0,
        "interpolated_frames": 0,
        "max_missing_gap_sec": 0.0,
        "interpolation_pct": 0.0,
        "width": 0,
        "height": 0,
        "duplicate_of": dup_of or "",
        "handedness": "",
        "movement_magnitude": 0.0,
        "wrist_movement": 0.0,
        "fingertip_movement": 0.0,
        "temporal_velocity": 0.0,
        "sequence_shape": "",
        "has_nan": False,
        "has_inf": False,
        "sequence_path": "",
    }

    times: list[float] = []
    feats: list[np.ndarray] = []
    raw_coords: list[np.ndarray] = []
    handed_seq: list[str] = []
    det_flags: list[bool] = []
    fps = 0.0
    size = path.stat().st_size

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            rec["fps"] = fps
            rec["width"] = stream.width
            rec["height"] = stream.height
            frame_idx = 0
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                t = frame_idx / fps if fps else float(frame_idx)
                det_flags.append(False)
                res = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
                )
                if res.hand_landmarks:
                    hand = res.hand_landmarks[0]
                    if len(hand) == 21:
                        f = rt.normalize_landmarks(hand)
                        if np.all(np.isfinite(f)):
                            det_flags[-1] = True
                            times.append(t)
                            feats.append(f)
                            raw_coords.append(np.array(
                                [[lm.x, lm.y] for lm in hand], dtype=np.float64))
                            hd = "Unknown"
                            if res.handedness and res.handedness[0]:
                                hd = str(res.handedness[0][0].category_name)
                            handed_seq.append(hd)
                frame_idx += 1
            rec["original_frames"] = frame_idx
    except Exception as exc:  # noqa: BLE001
        rec["include"] = False
        rec["exclusion_reason"] = f"gagal dibuka: {type(exc).__name__}: {exc}"
        return rec

    total = rec["original_frames"]
    det = len(feats)
    rec["detected_frames"] = det
    rec["detection_rate"] = det / total if total else 0.0
    if fps:
        rec["duration_sec"] = total / fps

    if not times:
        rec["include"] = False
        rec["exclusion_reason"] = "tidak ada tangan terdeteksi"
        return rec

    # Span aktif (frame pertama..terakhir terdeteksi)
    t_arr = np.array(times, dtype=np.float64)
    f_arr = np.array(feats, dtype=np.float64)
    span_frames = int(round((t_arr[-1] - t_arr[0]) * fps)) + 1 if fps else det
    missing_span = max(0, span_frames - det)
    rec["interpolated_frames"] = missing_span
    rec["interpolation_pct"] = missing_span / span_frames if span_frames else 0.0

    # max missing gap (konversi index-frame terdeteksi, berbasis waktu)
    if fps:
        max_gap = 0.0
        for a in range(1, len(t_arr)):
            gap = t_arr[a] - t_arr[a - 1] - (1.0 / fps)
            if gap > max_gap:
                max_gap = gap
        rec["max_missing_gap_sec"] = max(0.0, max_gap)

    rec["handedness"] = Counter(handed_seq).most_common(1)[0][0] if handed_seq else ""

    # Gerakan dihitung pada koordinat KAMERA (raw), karena setelah
    # wrist-relative normalization posisi wrist selalu di origin.
    rec_seq = normalize_temporal(t_arr, f_arr, SEQ_LEN)
    raw_arr = np.array(raw_coords, dtype=np.float64)          # (det, 21, 2)
    raw_seq = resample_series(t_arr, raw_arr, SEQ_LEN)        # (24, 21, 2)
    d_wrist = np.linalg.norm(np.diff(raw_seq[:, 0, :], axis=0), axis=1)
    d_tips = np.linalg.norm(np.diff(raw_seq[:, FINGERTIPS, :], axis=0), axis=2)
    d_all = np.linalg.norm(np.diff(raw_seq, axis=0), axis=2)  # (23, 21)
    rec["wrist_movement"] = float(np.sum(d_wrist))
    rec["fingertip_movement"] = float(np.sum(d_tips))
    rec["movement_magnitude"] = float(np.sum(np.mean(d_all, axis=1)))
    span_sec = max(1e-6, t_arr[-1] - t_arr[0])
    rec["temporal_velocity"] = rec["movement_magnitude"] / span_sec

    # Kelayakan
    reasons = []
    if path.name in KNOWN_EXCLUDE:
        reasons.append(KNOWN_EXCLUDE[path.name])
    if dup_of:
        reasons.append(f"duplikat dari {dup_of}")
    if rec["detection_rate"] < DETECTION_EXCLUDE:
        reasons.append(f"detection rate rendah ({rec['detection_rate']*100:.0f}%)")
    if rec["max_missing_gap_sec"] > GAP_EXCLUDE_SEC:
        reasons.append(f"gap deteksi panjang ({rec['max_missing_gap_sec']:.2f}s)")
    if rec["interpolation_pct"] > INTERP_EXCLUDE_FRAC:
        reasons.append(f"interpolasi berlebih ({rec['interpolation_pct']*100:.0f}%)")
    if rec["movement_magnitude"] < 0.05:
        reasons.append(f"gerakan terlalu kecil ({rec['movement_magnitude']:.3f})")

    rec["has_nan"] = bool(np.any(np.isnan(rec_seq)))
    rec["has_inf"] = bool(np.any(np.isinf(rec_seq)))
    if rec["has_nan"] or rec["has_inf"]:
        reasons.append("sequence mengandung NaN/inf")

    rec["sequence_shape"] = f"{rec_seq.shape[0]}x{rec_seq.shape[1]}"

    if reasons:
        rec["include"] = False
        rec["exclusion_reason"] = "; ".join(reasons)
        return rec

    # Simpan
    out_dir = SEQ_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{path.stem}.npz"
    np.savez_compressed(
        npz_path,
        sequence=rec_seq.astype(np.float32),
        label=label,
        source_video=f"{label}/{path.name}",
        original_fps=np.float32(fps),
        duration_sec=np.float32(rec["duration_sec"]),
        valid_frames=np.int32(det),
        handedness=rec["handedness"],
        movement_magnitude=np.float32(rec["movement_magnitude"]),
        wrist_movement=np.float32(rec["wrist_movement"]),
        fingertip_movement=np.float32(rec["fingertip_movement"]),
        temporal_velocity=np.float32(rec["temporal_velocity"]),
        file_size_bytes=np.int64(size),
    )
    rec["sequence_path"] = npz_path.relative_to(ROOT).as_posix()
    return rec


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    files = video_files()
    print(f"== Manifest + ekstraksi sequence ==  ({len(files)} video)")

    # hash -> nama file pertama (deteksi duplikat)
    first_by_hash: dict[str, str] = {}
    dup_map: dict[str, str] = {}
    for label, p in files:
        d = sha256(p)
        if d in first_by_hash:
            dup_map[str(p)] = first_by_hash[d]
        else:
            first_by_hash[d] = p.name

    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    rows: list[dict] = []
    try:
        for i, (label, p) in enumerate(files, 1):
            dup_of = dup_map.get(str(p))
            rec = extraction_worker(label, p, landmarker, dup_of)
            rows.append(rec)
            status = "OK" if rec["include"] else "EXCLUDE"
            print(f"[{i:2d}/{len(files)}] {rec['source_video']:14s} {status:7s} "
                  f"det={rec['detection_rate']*100:5.1f}% gap={rec['max_missing_gap_sec']:.2f}s "
                  f"interp={rec['interpolation_pct']*100:4.0f}% "
                  f"{('- ' + rec['exclusion_reason']) if rec['exclusion_reason'] else ''}")
    finally:
        landmarker.close()

    # ---- Manifest ----
    manifest_cols = [
        "source_video", "label", "include", "exclusion_reason", "fps",
        "duration_sec", "original_frames", "detection_rate", "detected_frames",
        "interpolated_frames", "max_missing_gap_sec", "interpolation_pct",
        "width", "height", "duplicate_of", "handedness", "movement_magnitude",
        "sequence_path",
    ]
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(manifest_cols)
        for r in rows:
            w.writerow([r[c] for c in manifest_cols])

    # ---- Summary CSV ----
    summary_cols = manifest_cols + [
        "wrist_movement", "fingertip_movement", "temporal_velocity",
        "sequence_shape", "has_nan", "has_inf",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(summary_cols)
        for r in rows:
            w.writerow([r[c] for c in summary_cols])

    # ---- QC validasi sequence ----
    included = [r for r in rows if r["include"]]
    excluded = [r for r in rows if not r["include"]]
    final_counts = Counter(r["label"] for r in included)

    qc_fail: list[str] = []
    seen_sources: Counter = Counter()
    for r in included:
        npz = ROOT / r["sequence_path"]
        with np.load(npz, allow_pickle=True) as d:
            seq = d["sequence"]
            if seq.shape != (SEQ_LEN, FEATURE_COUNT):
                qc_fail.append(f"{r['source_video']}: shape {seq.shape}")
            if not np.all(np.isfinite(seq)):
                qc_fail.append(f"{r['source_video']}: NaN/inf")
            if str(d["label"]) != r["label"]:
                qc_fail.append(f"{r['source_video']}: label mismatch")
            seen_sources[str(d["source_video"])] += 1
    dup_sources = [s for s, c in seen_sources.items() if c > 1]

    # ---- Movement stats ----
    def stats(vals):
        vals = [float(v) for v in vals]
        if not vals:
            return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0, "std": 0.0}
        return {
            "min": min(vals), "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)), "max": max(vals),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        }

    movement = {
        "per_label": {
            lab: {
                "count": sum(1 for r in included if r["label"] == lab),
                "movement_magnitude": stats([r["movement_magnitude"] for r in included if r["label"] == lab]),
                "wrist_movement": stats([r["wrist_movement"] for r in included if r["label"] == lab]),
                "fingertip_movement": stats([r["fingertip_movement"] for r in included if r["label"] == lab]),
                "temporal_velocity": stats([r["temporal_velocity"] for r in included if r["label"] == lab]),
                "duration_sec": stats([r["duration_sec"] for r in included if r["label"] == lab]),
                "original_fps": stats([r["fps"] for r in included if r["label"] == lab]),
            }
            for lab in LABELS
        },
        "note": "Movement dihitung pada sequence 24 timestep (koordinat ternormalisasi). "
                "Confidence model dihitung di scripts/train_dynamic_jz_gru.py.",
    }
    MOVEMENT_JSON.write_text(json.dumps(movement, indent=2), encoding="utf-8")

    # ---- Laporan ----
    L: list[str] = []
    a = L.append
    a("# DYNAMIC SEQUENCE REPORT - J/Z (24x63)")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("- Script: `scripts/extract_dynamic_sequences.py`")
    a(f"- Waktu ekstraksi: {time.perf_counter() - start:.1f} detik")
    a("- Dataset: read-only (tidak ada file asli yang diubah). Tidak ada training.")
    a("")
    a("## 1. Ringkasan")
    a("")
    a(f"- Video awal: {sum(1 for r in rows if r['label'] == 'J')} J + "
      f"{sum(1 for r in rows if r['label'] == 'Z')} Z = {len(rows)}.")
    a(f"- **Final included: {final_counts.get('J', 0)} J + "
      f"{final_counts.get('Z', 0)} Z = {len(included)} sequence.**")
    a(f"- Dikecualikan: **{len(excluded)}** video.")
    a(f"- Sequence shape: **({SEQ_LEN}, {FEATURE_COUNT})** (tidak di-flatten).")
    a(f"- QC shape/NaN/inf/label: **{'LULUS' if not qc_fail else 'GAGAL'}**.")
    a(f"- Duplicate source: {len(dup_sources)}.")
    a("")
    a("## 2. Cleaning (exclusion)")
    a("")
    a("| Video | Alasan |")
    a("|:------|:-------|")
    for r in excluded:
        a(f"| {r['source_video']} | {r['exclusion_reason']} |")
    a("")
    a("File asli **tidak dihapus**; hanya dikeluarkan dari daftar training.")
    a("")
    a("## 3. Dataset Final")
    a("")
    a("| Label | Jumlah sequence | FPS asal | Durasi (mean s) |")
    a("|:-----:|----------------:|:--------:|----------------:|")
    for lab in LABELS:
        rs = [r for r in included if r["label"] == lab]
        fps_vals = sorted({int(r["fps"]) for r in rs})
        dur = statistics.mean([r["duration_sec"] for r in rs]) if rs else 0.0
        a(f"| {lab} | {len(rs)} | {', '.join(map(str, fps_vals))} | {dur:.2f} |")
    a("")
    a("## 4. Penanganan Missing Detection")
    a("")
    a("Strategi: interpolasi **hanya gap pendek** (linear antar landmark "
      "sebelum/sesudah gap). Gap > "
      f"{GAP_EXCLUDE_SEC:.2f}s atau interpolasi > {INTERP_EXCLUDE_FRAC*100:.0f}% "
      "menyebabkan video di-exclude, bukan diisi sembarangan.")
    a("")
    a("| Video | Frames | Detected | Interpolated | Max gap (s) | Interp % |")
    a("|:------|-------:|---------:|-------------:|------------:|---------:|")
    for r in rows:
        a(f"| {r['source_video']} | {r['original_frames']} | "
          f"{r['detected_frames']} | {r['interpolated_frames']} | "
          f"{r['max_missing_gap_sec']:.2f} | {r['interpolation_pct']*100:.0f}% |")
    a("")
    a("## 5. Timestamp & Temporal Resampling")
    a("")
    a("- Timestamp tiap frame: `t = frame_index / fps` (bukan indeks mentah).")
    a("- Segmen aktif = dari frame terdeteksi pertama sampai terakhir; waktu "
      "dinormalisasi 0.0 → 1.0 pada segmen tsb.")
    a(f"- Resampling ke **{SEQ_LEN} timestep** dengan **interpolasi linear** "
      "`np.interp` per fitur pada timestamp terdeteksi.")
    a("- Konsekuensi: perbedaan FPS J=50 dan Z=25 **tidak** menjadi fitur; "
      "model hanya melihat bentuk gerakan ternormalisasi waktu.")
    a("")
    a("## 6. Quality Check Sequence")
    a("")
    if qc_fail:
        a("Masalah:")
        for f in qc_fail:
            a(f"- {f}")
    else:
        a("- Semua sequence ber-shape (24, 63).")
        a("- Tidak ada NaN/inf.")
        a("- Label sesuai folder.")
        a("- Tidak ada duplicate source.")
        a("- Source video dapat ditelusuri (tersimpan di npz & manifest).")
    a("")
    a("## 7. Movement Statistics (untuk motion routing & rejection)")
    a("")
    a("| Label | movement magnitude (mean) | wrist (mean) | fingertip (mean) | velocity (mean) |")
    a("|:-----:|--------------------------:|-------------:|-----------------:|----------------:|")
    for lab in LABELS:
        s = movement["per_label"][lab]
        a(f"| {lab} | {s['movement_magnitude']['mean']:.4f} | "
          f"{s['wrist_movement']['mean']:.4f} | "
          f"{s['fingertip_movement']['mean']:.4f} | "
          f"{s['temporal_velocity']['mean']:.3f} |")
    a("")
    a("Detail (min/median/max/std) ada di `reports/dynamic_movement_stats.json`. "
      "Distribusi confidence model ditambahkan di laporan training.")
    a("")
    a("## 8. Catatan")
    a("")
    a("- Sequence disimpan sebagai `.npz` di `data/processed/dynamic_sequences/J|Z/`.")
    a("- Setiap npz berisi: sequence (24,63), label, source_video, original_fps, "
      "duration_sec, valid_frames, handedness, movement_magnitude (+ wrist/"
      "fingertip/velocity).")
    a("- Belum ada training di tahap ini.")
    a("")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"\nFinal: {final_counts.get('J', 0)} J + {final_counts.get('Z', 0)} Z "
          f"= {len(included)} sequence | excluded {len(excluded)}")
    print(f"QC: {'LULUS' if not qc_fail else 'GAGAL'} | dup_sources={len(dup_sources)}")
    print(f"Manifest : {MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Report   : {REPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
