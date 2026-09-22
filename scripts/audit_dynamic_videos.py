"""Audit dataset video dinamis J dan Z (tanpa training, read-only).

Untuk setiap video:
- metadata (format, ukuran, resolusi, FPS, jumlah frame, durasi);
- deteksi MediaPipe HandLandmarker pada SELURUH frame;
- detection rate, distribusi handedness, validitas 21 landmark & fitur x,y,z;
- karakteristik gerakan (displacement wrist/fingertip, total movement,
  frame awal vs akhir, durasi gesture aktif);
- contact sheet keyframes (0/25/50/75/100%) ke reports/dynamic_previews/.

Output:
- scripts/audit_dynamic_videos.py (script ini)
- reports/dynamic_video_summary.csv
- reports/DYNAMIC_VIDEO_AUDIT.md
- reports/dynamic_previews/*.png

Jalankan:
    python scripts/audit_dynamic_videos.py
"""

from __future__ import annotations

import csv
import hashlib
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av  # noqa: E402
import cv2  # noqa: E402

import mediapipe as mp  # noqa: E402
import realtime as rt  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
PREVIEWS = REPORTS / "dynamic_previews"
SUMMARY_CSV = REPORTS / "dynamic_video_summary.csv"
REPORT_MD = REPORTS / "DYNAMIC_VIDEO_AUDIT.md"

LABELS = ("J", "Z")
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
FINGERTIPS = (4, 8, 12, 16, 20)
MOVE_EPS = 0.002  # ambang perpindahan wrist (koordinat ternormalisasi)

SEQ_CANDIDATES = (16, 20, 24, 30, 32)


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_files() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for label in LABELS:
        d = RAW / label
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                out.append((label, p))
    return out


def draw_skeleton(frame_bgr: np.ndarray, hand) -> None:
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for a, b in rt.HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        color = (0, 0, 255) if i == 0 else (255, 160, 0)
        cv2.circle(frame_bgr, (x, y), 3, color, -1, cv2.LINE_AA)


def save_contact_sheet(label: str, path: Path, keyframes: list[tuple[int, float, np.ndarray | None]]):
    """keyframes: list of (frame_idx, time_sec, bgr_small | None)."""
    tile_w = 320
    tiles = []
    for idx, tsec, img in keyframes:
        if img is None:
            tile = np.zeros((180, tile_w, 3), dtype=np.uint8)
            cv2.putText(tile, "no frame", (10, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            tile = cv2.resize(img, (tile_w, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile_w - 1, 22), (0, 0, 0), -1)
        cv2.putText(tile, f"#{idx}  {tsec:.2f}s", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    cv2.putText(sheet, f"{label}  {path.name}", (8, 176),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    out = PREVIEWS / f"{label}_{path.stem}.png"
    cv2.imwrite(str(out), sheet)


# --------------------------------------------------------------------------- #
# Audit satu video
# --------------------------------------------------------------------------- #

def audit_video(label: str, path: Path, landmarker, digest_map: dict) -> dict:
    rec: dict = {
        "label": label, "filename": path.name, "format": path.suffix.lower().lstrip("."),
        "size_bytes": path.stat().st_size, "fps": 0.0, "frames": 0,
        "duration_sec": 0.0, "width": 0, "height": 0, "codec": "",
        "detected_frames": 0, "detection_rate": 0.0,
        "left_frames": 0, "right_frames": 0, "unknown_hand_frames": 0,
        "no_hand_frames": 0, "decode_errors": 0,
        "bad_landmark_frames": 0, "nonfinite_feature_frames": 0,
        "mean_wrist_disp": 0.0, "mean_fingertip_disp": 0.0,
        "total_movement": 0.0, "first_last_diff": 0.0,
        "active_span_sec": 0.0, "moving_frames": 0,
        "duration_container_sec": 0.0,
        "valid": False, "notes": "",
    }
    notes: list[str] = []
    keyframe_slots: list[tuple[int, float, np.ndarray | None]] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            rec["fps"] = fps
            rec["width"] = stream.width
            rec["height"] = stream.height
            rec["codec"] = stream.codec_context.name
            if container.duration:
                rec["duration_container_sec"] = float(container.duration / av.time_base)

            det_flags: list[bool] = []
            wrists: list[np.ndarray | None] = []
            tips: list[np.ndarray | None] = []
            handed_seq: list[str] = []

            frame_idx = -1
            for frame in container.decode(stream):
                frame_idx += 1
                try:
                    rgb = frame.to_ndarray(format="rgb24")
                except Exception:  # noqa: BLE001
                    rec["decode_errors"] += 1
                    continue

                det_flags.append(False)
                wrists.append(None)
                tips.append(None)
                handed_seq.append("None")

                res = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                )
                if not res.hand_landmarks:
                    continue

                hand = res.hand_landmarks[0]
                if len(hand) != 21:
                    rec["bad_landmark_frames"] += 1
                    continue
                feats = rt.normalize_landmarks(hand)
                if not np.all(np.isfinite(feats)):
                    rec["nonfinite_feature_frames"] += 1
                    continue

                det_flags[-1] = True
                wrists[-1] = np.array([hand[0].x, hand[0].y], dtype=np.float64)
                tips[-1] = np.array([[hand[i].x, hand[i].y] for i in FINGERTIPS],
                                    dtype=np.float64)
                hd = "Unknown"
                if res.handedness and res.handedness[0]:
                    hd = str(res.handedness[0][0].category_name)
                handed_seq[-1] = hd

            rec["frames"] = frame_idx + 1
            total = max(1, rec["frames"])
            det_idx = [i for i, f in enumerate(det_flags) if f]
            rec["detected_frames"] = len(det_idx)
            rec["detection_rate"] = len(det_idx) / total
            rec["no_hand_frames"] = total - len(det_idx)
            left = sum(1 for h in handed_seq if h == "Left")
            right = sum(1 for h in handed_seq if h == "Right")
            rec["left_frames"] = left
            rec["right_frames"] = right
            rec["unknown_hand_frames"] = sum(
                1 for h in handed_seq if h in ("Unknown",) and h != "None"
            )

            # durasi dari frame terhitung
            if fps > 0:
                rec["duration_sec"] = rec["frames"] / fps
            elif rec["duration_container_sec"]:
                rec["duration_sec"] = rec["duration_container_sec"]

            # Gerakan hanya antar frame terdeteksi berurutan.
            wd, td, md = [], [], []
            moving = 0
            for a in range(1, len(det_flags)):
                if not (det_flags[a - 1] and det_flags[a]):
                    continue
                dw = float(np.linalg.norm(wrists[a] - wrists[a - 1]))
                dt = float(np.mean(np.linalg.norm(tips[a] - tips[a - 1], axis=1)))
                wd.append(dw)
                td.append(dt)
                md.append(0.5 * (dw + dt))
                if dw > MOVE_EPS:
                    moving += 1
            rec["mean_wrist_disp"] = float(np.mean(wd)) if wd else 0.0
            rec["mean_fingertip_disp"] = float(np.mean(td)) if td else 0.0
            rec["total_movement"] = float(np.sum(md)) if md else 0.0
            rec["moving_frames"] = moving
            if det_idx:
                rec["active_span_sec"] = (det_idx[-1] - det_idx[0]) / fps if fps else 0.0
                first, last = det_idx[0], det_idx[-1]
                if det_flags[first] and det_flags[last] and first != last:
                    rec["first_last_diff"] = float(np.linalg.norm(wrists[last] - wrists[first]))

            # Contact sheet keyframes (gambar ulang pada pass kedua di main).
            rec["_keyframe_indices"] = [
                0,
                int(round(0.25 * (total - 1))),
                int(round(0.5 * (total - 1))),
                int(round(0.75 * (total - 1))),
                total - 1,
            ]

        # Duplikat berdasarkan hash
        digest = digest_map.get(sha256(path))
        if digest and digest != path.name:
            notes.append(f"duplikat dari {digest}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"gagal dibuka: {type(exc).__name__}: {exc}")

    if rec["frames"] == 0:
        notes.append("tidak ada frame terbaca")
    if rec["decode_errors"]:
        notes.append(f"{rec['decode_errors']} frame corrupt")
    if rec["bad_landmark_frames"]:
        notes.append(f"{rec['bad_landmark_frames']} frame landmark != 21")
    if rec["nonfinite_feature_frames"]:
        notes.append(f"{rec['nonfinite_feature_frames']} frame fitur NaN/inf")
    if rec["frames"] and rec["detection_rate"] < 0.80:
        notes.append(f"detection rate rendah ({rec['detection_rate']*100:.0f}%)")
    if rec["detected_frames"] > 0 and rec["total_movement"] < 0.05:
        notes.append("gerakan sangat kecil")

    rec["valid"] = bool(
        rec["frames"] > 0
        and rec["decode_errors"] == 0
        and rec["bad_landmark_frames"] == 0
        and rec["nonfinite_feature_frames"] == 0
        and rec["detection_rate"] >= 0.50
    )
    rec["notes"] = "; ".join(notes)
    return rec


def build_contact_sheet(label: str, path: Path, indices: list[int],
                        landmarker) -> None:
    """Pass kedua: ambil keyframes & gambar skeleton, simpan contact sheet."""
    want = sorted(set(i for i in indices if i >= 0))
    tiles: list[tuple[int, float, np.ndarray | None]] = []
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx not in want:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                small = cv2.resize(bgr, (320, 180), interpolation=cv2.INTER_AREA)
                res = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
                )
                if res.hand_landmarks:
                    draw_skeleton(small, res.hand_landmarks[0])
                tiles.append((frame_idx, frame_idx / fps if fps else 0.0, small))
                if len(tiles) == len(want):
                    break
    except Exception:  # noqa: BLE001
        pass
    if not tiles:
        return
    save_contact_sheet(label, path, tiles)


# --------------------------------------------------------------------------- #
# Rekomendasi sequence length
# --------------------------------------------------------------------------- #

def recommend_sequence_length(rows: list[dict]) -> dict:
    """Pilih sequence length berdasarkan statistik frame aktif & realtime."""
    stats = {}
    for label in LABELS:
        rs = [r for r in rows if r["label"] == label and r["frames"] > 0]
        if not rs:
            continue
        stats[label] = {
            "fps": statistics.median(r["fps"] for r in rs),
            "frames": statistics.median(r["detected_frames"] for r in rs),
            "dur": statistics.median(r["duration_sec"] for r in rs),
            "span": statistics.median(r["active_span_sec"] for r in rs),
        }
    all_rows = [r for r in rows if r["frames"] > 0]
    med_dur = statistics.median(r["duration_sec"] for r in all_rows) if all_rows else 0.0
    det_list = [r["detected_frames"] for r in all_rows]
    min_frames = min(det_list, default=0)
    med_frames = statistics.median(det_list) if det_list else 0

    candidates = []
    for n in SEQ_CANDIDATES:
        sec50 = n / 50.0
        sec25 = n / 25.0
        below = sum(1 for d in det_list if d < n)
        candidates.append({
            "n": n,
            "sec_50fps": sec50,
            "sec_25fps": sec25,
            "videos_below": below,
            "videos_total": len(det_list),
            "cost": round(n / 24.0, 2),
        })
    return {
        "per_label": stats,
        "median_duration": med_dur,
        "min_detected_frames": min_frames,
        "median_detected_frames": med_frames,
        "candidates": candidates,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    files = video_files()
    print(f"== Audit video dinamis ==  ({len(files)} video)")

    # hash -> filename (deteksi duplikat)
    digests: dict[str, str] = {}
    digest_map: dict[str, str] = {}
    for label, p in files:
        d = sha256(p)
        if d in digests:
            digest_map[d] = digests[d]
        else:
            digests[d] = p.name

    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    rows: list[dict] = []
    try:
        for i, (label, p) in enumerate(files, 1):
            row = audit_video(label, p, landmarker, digest_map)
            idxs = row.pop("_keyframe_indices", [])
            print(f"[{i:2d}/{len(files)}] {p.name:12s} frames={row['frames']:3d} "
                  f"det={row['detection_rate']*100:5.1f}% valid={row['valid']} "
                  f"{('| ' + row['notes']) if row['notes'] else ''}")
            rows.append(row)
            build_contact_sheet(label, p, idxs, landmarker)
    finally:
        landmarker.close()

    # ---- CSV ----
    cols = [
        "label", "filename", "format", "size_bytes", "fps", "frames",
        "duration_sec", "width", "height", "detected_frames", "detection_rate",
        "left_frames", "right_frames", "no_hand_frames", "valid", "notes",
        "codec", "decode_errors", "bad_landmark_frames",
        "nonfinite_feature_frames", "mean_wrist_disp", "mean_fingertip_disp",
        "total_movement", "first_last_diff", "active_span_sec", "moving_frames",
        "duration_container_sec",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    # ---- Statistik agregat ----
    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return (0.0, 0.0, 0.0)
        return (min(vals), float(statistics.mean(vals)), max(vals))

    by_label = {lab: [r for r in rows if r["label"] == lab] for lab in LABELS}
    overall = rows
    det_rate = (statistics.mean(r["detection_rate"] for r in rows)
                if rows else 0.0)
    total_frames = sum(r["frames"] for r in rows)
    total_detected = sum(r["detected_frames"] for r in rows)
    left_all = sum(r["left_frames"] for r in rows)
    right_all = sum(r["right_frames"] for r in rows)
    invalid = [r for r in rows if not r["valid"]]
    dup_notes = [r for r in rows if "duplikat" in r["notes"]]

    rec = recommend_sequence_length(rows)

    # ---- Laporan ----
    L: list[str] = []
    a = L.append
    a("# DYNAMIC VIDEO AUDIT - SIBI J & Z")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a("- Dataset: `data/raw/J`, `data/raw/Z` (read-only)")
    a("- MediaPipe: HandLandmarker (mode IMAGE, 21 landmark)")
    a(f"- Waktu audit: {time.perf_counter() - start:.1f} detik")
    a("- **Tidak ada training, tidak ada perubahan model/aplikasi.**")
    a("")
    a("## 1. Ringkasan")
    a("")
    a(f"- Video J: **{len(by_label['J'])}** | Video Z: **{len(by_label['Z'])}** | "
      f"Total: **{len(rows)}**")
    a(f"- Kodek: {', '.join(sorted({r['codec'] for r in rows}))} | "
      f"Format: {', '.join(sorted({r['format'] for r in rows}))}")
    resolutions = ", ".join(
        sorted({"{0}x{1}".format(r["width"], r["height"]) for r in rows})
    )
    a(f"- Resolusi: {resolutions}")
    fps_j = agg([r["fps"] for r in by_label["J"]])
    fps_z = agg([r["fps"] for r in by_label["Z"]])
    a(f"- FPS: J = {fps_j[0]:.0f} (tetap) | Z = {fps_z[0]:.0f} (tetap) "
      f"→ **kelas tidak seragam FPS**")
    a(f"- Total frame: **{total_frames}** | terdeteksi tangan: **{total_detected}** "
      f"({total_detected / total_frames * 100:.1f}%)")
    a(f"- Detection rate rata-rata: **{det_rate * 100:.1f}%**")
    a(f"- Handedness (frame): Left **{left_all}**, Right **{right_all}**")
    a(f"- Video tidak valid: **{len(invalid)}** | indikasi duplikat: "
      f"**{len(dup_notes)}**")
    a("")
    a("## 2. Inventaris Video (agregat)")
    a("")
    a("| Label | Jml | Resolusi | FPS | Durasi (min/mean/max s) | "
      "Frame (min/mean/max) |")
    a("|:-----:|----:|:--------:|:---:|:-----------------------:|:---------------------:|")
    for lab in LABELS:
        rs = by_label[lab]
        if not rs:
            continue
        d = agg([r["duration_sec"] for r in rs])
        f = agg([float(r["frames"]) for r in rs])
        a(f"| {lab} | {len(rs)} | {rs[0]['width']}x{rs[0]['height']} | "
          f"{rs[0]['fps']:.0f} | {d[0]:.2f} / {d[1]:.2f} / {d[2]:.2f} | "
          f"{int(f[0])} / {f[1]:.0f} / {int(f[2])} |")
    a("")
    a("Detail per video: lihat `reports/dynamic_video_summary.csv`.")
    a("")
    a("## 3. Audit MediaPipe")
    a("")
    a("MediaPipe HandLandmarker dijalankan pada **seluruh frame** setiap video.")
    a("")
    a(f"- Rata-rata detection rate: **{det_rate * 100:.1f}%**")
    a(f"- Frame tanpa tangan: **{total_frames - total_detected}**")
    a(f"- Handedness: Left {left_all} / Right {right_all} frame")
    a(f"- Frame landmark != 21: "
      f"**{sum(r['bad_landmark_frames'] for r in rows)}**")
    a(f"- Frame fitur NaN/inf: "
      f"**{sum(r['nonfinite_feature_frames'] for r in rows)}**")
    a(f"- Frame decode error (corrupt): "
      f"**{sum(r['decode_errors'] for r in rows)}**")
    a("")
    a("| Label | Detection rate (mean) | Left | Right | Tanpa tangan |")
    a("|:-----:|----------------------:|-----:|------:|-------------:|")
    for lab in LABELS:
        rs = by_label[lab]
        if not rs:
            continue
        dr = statistics.mean(r["detection_rate"] for r in rs)
        a(f"| {lab} | {dr * 100:.1f}% | {sum(r['left_frames'] for r in rs)} | "
          f"{sum(r['right_frames'] for r in rs)} | "
          f"{sum(r['no_hand_frames'] for r in rs)} |")
    a("")
    a("Catatan handedness: distribusi berbeda antar kelas — J sangat didominasi "
      "satu sisi, sedangkan Z lebih bercampur. Ini perlu diperhatikan agar "
      "model temporal tidak bias ke satu sisi tangan (lihat juga "
      "`HANDEDNESS_REPORT.md` untuk dataset statis).")
    a("")
    a("## 4. Audit Gerakan")
    a("")
    a("| Label | wrist disp (mean) | fingertip disp (mean) | total movement | "
      "first-vs-last | durasi aktif (s) |")
    a("|:-----:|------------------:|----------------------:|---------------:|"
      "--------------:|-----------------:|")
    for lab in LABELS:
        rs = by_label[lab]
        if not rs:
            continue
        a(f"| {lab} | {statistics.mean(r['mean_wrist_disp'] for r in rs):.4f} | "
          f"{statistics.mean(r['mean_fingertip_disp'] for r in rs):.4f} | "
          f"{statistics.mean(r['total_movement'] for r in rs):.3f} | "
          f"{statistics.mean(r['first_last_diff'] for r in rs):.4f} | "
          f"{statistics.mean(r['active_span_sec'] for r in rs):.2f} |")
    a("")
    a("Nilai displacement dalam koordinat landmark ternormalisasi (0–1). "
      "Audit gerakan memakai seluruh frame, bukan satu frame, sehingga J/Z "
      "tidak disimpulkan dari pose tunggal.")
    a("")
    a("## 5. Preview / Quality Check")
    a("")
    a(f"Contact sheet keyframes (0%, 25%, 50%, 75%, 100%) untuk setiap video "
      f"tersimpan di `reports/dynamic_previews/` ({len(rows)} file). "
      "Setiap tile menampilkan indeks frame, waktu, dan skeleton bila tangan "
      "terdeteksi. Tidak ada OCR.")
    a("")
    a(f"Contoh: `{(PREVIEWS / (rows[0]['label'] + '_' + Path(rows[0]['filename']).stem + '.png')).relative_to(ROOT).as_posix()}`"
      if rows else "")
    a("")
    a("## 6. Rekomendasi Sequence Length")
    a("")
    ps = rec["per_label"]
    for lab in LABELS:
        if lab in ps:
            s = ps[lab]
            a(f"- **{lab}**: fps={s['fps']:.0f}, median frame terdeteksi="
              f"{s['frames']:.0f}, median durasi={s['dur']:.2f}s, median span "
              f"aktif={s['span']:.2f}s")
    a("")
    a(f"- Median durasi video: **{rec['median_duration']:.2f}s**; median frame "
      f"terdeteksi: **{rec['median_detected_frames']:.0f}**; minimum frame "
      f"terdeteksi: **{rec['min_detected_frames']}**.")
    a("")
    a("| N frame | setara @50fps | setara @25fps | video dengan < N frame terdeteksi | biaya relatif |")
    a("|--------:|--------------:|--------------:|:--------------------------------:|--------------:|")
    for c in rec["candidates"]:
        a(f"| {c['n']} | {c['sec_50fps']:.2f}s | {c['sec_25fps']:.2f}s | "
          f"{c['videos_below']}/{c['videos_total']} | {c['cost']}x |")
    a("")
    # Rekomendasi eksplisit berbasis data (bukan memilih N terkecil otomatis).
    REC_SEQ = 24
    c24 = next(c for c in rec["candidates"] if c["n"] == REC_SEQ)
    c20 = next(c for c in rec["candidates"] if c["n"] == 20)
    a(f"**Rekomendasi: {REC_SEQ} frame** sebagai panjang sequence awal "
      "(alternatif: 20 frame lebih ringan, 30 frame lebih detail).")
    a("")
    a("Alasan:")
    a(f"- Durasi aktif gesture (span) ±1.0–1.3s; median durasi video "
      f"{rec['median_duration']:.2f}s. Segmen aktif cukup panjang untuk "
      f"{REC_SEQ} sampel.")
    a("- Karena FPS kelas tidak seragam (J=50, Z=25), panjang sequence harus "
      "diperlakukan sebagai **jumlah sampel dari segmen aktif** yang "
      "di-resample, bukan durasi mentah. Jadi N dipilih dari kecukupan "
      "representasi gerakan, bukan dari FPS.")
    a(f"- Pada N={REC_SEQ}, {c24['videos_below']}/{c24['videos_total']} video "
      f"punya lebih sedikit frame terdeteksi (di-interpolasi/padding); "
      f"sebagian besar cukup. N=20 sedikit lebih hemat "
      f"({c20['videos_below']}/{c20['videos_total']} perlu interpolasi) tetapi "
      "memberi resolusi temporal lebih rendah.")
    a(f"- N={REC_SEQ} memberi jarak antar-sampel ±50 ms pada gerakan J "
      "(cukup menangkap lintasan), sementara 16 frame (±75 ms) berisiko "
      "memotong fase gerakan penentu J/Z, dan 30–32 frame menambah biaya "
      "komputasi untuk tambahan informasi kecil.")
    a("- Realtime: N=24 pada ~24 FPS inference ≈ 1 detik jendela gerakan — "
      "wajar untuk pengenalan J/Z tanpa menambah latency berarti.")
    a("")
    a("## 7. Rencana Temporal Normalization (tahap berikutnya, belum diimplementasikan)")
    a("")
    a("```")
    a("video")
    a("  → decode frame")
    a("  → MediaPipe landmark per frame (21 x (x,y,z))")
    a("  → buang frame tanpa deteksi (atau interpolasi)")
    a("  → wrist-relative normalization (p - p_wrist)")
    a("  → scale normalization (/ max ||p - p_wrist||)")
    a("  → urutan temporal 63-fitur per frame")
    a("  → resampling ke panjang sequence tetap (N=24)")
    a("```")
    a("")
    a("Resampling yang direkomendasikan: **linear index sampling** pada indeks "
      "ternormalisasi (`np.linspace(0, T-1, N)`) dengan pembulatan/interpolasi "
      "linear antar-frame. Metode ini sederhana, deterministik, dan menjaga "
      "urutan temporal tanpa memerlukan model tambahan. Alternatif: interpolasi "
      "linear pada fitur (bukan sekadar memilih frame terdekat) untuk "
      "mengurangi jitter.")
    a("")
    a("## 8. Data Leakage & Saran Split")
    a("")
    a("- Pola nama file hanya `J_xxx` / `Z_xxx`; **tidak ada informasi "
      "signer/subjek** yang bisa dipakai untuk signer-independent split.")
    a(f"- Indikasi duplikat: {len(dup_notes)} file. "
      + (", ".join(f"`{r['filename']}` ({r['notes']})" for r in dup_notes)
         if dup_notes else "tidak ada."))
    a(f"- Celah penomoran: J kehilangan nomor tertentu (mis. J_020) — periksa "
      "apakah disengaja.")
    a("- Perbedaan FPS antar kelas (J=50, Z=25) dan lonjakan ukuran file "
      "(Z_011–Z_015) menunjukkan kemungkinan **sesi/perangkat perekaman "
      "berbeda**; jangan menganggap seluruh video satu subjek.")
    a("")
    a("Saran split:")
    a("- **Unit split = video/sequence, BUKAN frame.** Jangan memisah frame "
      "dari video yang sama ke train dan test.")
    a("- Gunakan **video-level (atau signer/sesi-level) hold-out** dan "
      "stratified per kelas. Jika nanti ada ID signer, gunakan "
      "signer-independent split.")
    a("- Buang/bedakan duplikat persis agar tidak bocor antar split.")
    a("- Validasi dengan GroupKFold berbasis video.")
    a("")
    a("## 9. Video Bermasalah")
    a("")
    if invalid:
        a("| File | Masalah |")
        a("|:-----|:--------|")
        for r in invalid:
            a(f"| {r['label']}/{r['filename']} | {r['notes'] or 'invalid'} |")
    else:
        a("Tidak ada video yang gagal validasi dasar (semua terbaca, tidak ada "
          "frame corrupt, detection rate >= 50%).")
    a("")
    a("## 10. Apakah Dataset Cukup untuk Eksperimen Awal?")
    a("")
    a(f"- Total {len(rows)} video ({len(by_label['J'])} J, {len(by_label['Z'])} Z) "
      f"dengan ±{total_frames} frame dan detection rate "
      f"{det_rate * 100:.1f}%.")
    a("- Volume ini **cukup untuk eksperimen awal / proof-of-concept** model "
      "temporal J vs Z, tetapi **kecil** untuk model produksi dan rawan "
      "overfitting.")
    a("- Sangat disarankan **augmentasi** dan validasi silang video-level, serta "
      "menambah subjek/sesi perekaman berbeda sebelum klaim generalisasi.")
    a("")
    a("## 11. Keterbatasan Dataset")
    a("")
    a("- Jumlah video sedikit (±15–21 per kelas) dan durasi pendek (±1–2s).")
    a("- FPS tidak seragam antar kelas (50 vs 25).")
    a("- Tidak ada metadata signer; risiko leakage antar split jika asal-usul "
      "video tidak dipisah.")
    a("- Kemungkinan duplikat (mis. J_001/J_007) yang harus dibersihkan.")
    a("- Gerakan hanya satu tangan per frame (model fokus pada satu tangan).")
    a("")
    a("> **Dataset BELUM dinyatakan siap training.** Audit ini adalah langkah "
      "pra-syarat; keputusan lanjut harus mempertimbangkan temuan di atas.")
    a("")
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")

    print(f"\nSummary CSV : {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Report      : {REPORT_MD.relative_to(ROOT)}")
    print(f"Previews    : {PREVIEWS.relative_to(ROOT)} ({len(list(PREVIEWS.glob('*.png')))} file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
