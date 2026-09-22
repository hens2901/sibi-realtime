"""Diagnostic live kelas statis sulit (K,N,R,U,V,X) - CURRENT vs V2.

Preprocessing landmark dilakukan SATU KALI per frame; fitur 63 yang sama
diberikan ke model CURRENT dan V2 (side-by-side).

User memilih TARGET aktual (bukan prediksi). Tekan S untuk menyimpan satu
landmark sample beserta hasil KEDUA model. Hanya landmark yang disimpan.

Kontrol:
    1..6        target: 1=K 2=N 3=R 4=U 5=V 6=X
    S / SPACE   simpan (wajib pilih target dulu)
    C           clear target
    Q / ESC     keluar

Output:
- data/validation/static_live/<target>/<timestamp>.npz
- reports/static_hard_live.csv

Jalankan:
    python scripts/diagnose_static_hard_classes.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402

HARD = ["K", "N", "R", "U", "V", "X"]
OUT_DIR = ROOT / "data" / "validation" / "static_live"
CSV_PATH = ROOT / "reports" / "static_hard_live.csv"
WINDOW = "SIBI Static Hard-Class: CURRENT vs V2"
MODELS = ROOT / "models"
CONF_TH, MARGIN_TH = 0.85, 0.20

CSV_COLS = ["timestamp", "target", "handedness", "hand_score", "mirror", "npz_path",
            "current_top1", "current_conf", "current_top2", "current_top2_prob",
            "current_margin", "current_accepted", "current_correct",
            "v2_top1", "v2_conf", "v2_top2", "v2_top2_prob",
            "v2_margin", "v2_accepted", "v2_correct"]


def load_models():
    cur = rt.load_artifacts("augmented")
    v2_model = joblib.load(MODELS / "sibi_mlp_augmented_v2.joblib")
    v2_scaler = joblib.load(MODELS / "scaler_augmented_v2.joblib")
    v2_enc = joblib.load(MODELS / "label_encoder_augmented_v2.joblib")
    v2 = (v2_model, v2_scaler, v2_enc, [str(x) for x in v2_enc.classes_])
    return {"current": cur, "v2": v2}


def predict(bundle, feats, target):
    model, scaler, enc, labels = bundle
    probs = rt.predict_proba(model, scaler, feats)
    t2 = rt.top2_from_probs(probs, list(labels))
    accepted = rt.passes_rejection(t2.top1_prob, t2.margin, CONF_TH, MARGIN_TH)
    return {
        "top1": t2.top1_label, "conf": float(t2.top1_prob),
        "top2": t2.top2_label, "top2_prob": float(t2.top2_prob),
        "margin": float(t2.margin), "accepted": bool(accepted),
        "correct": bool(target is not None and t2.top1_label == target),
    }


def open_camera(index, w, h):
    for backend in (cv2.CAP_ANY, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def draw(frame, target, cur, v2, handedness, mirror, counts, message, fps):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel = 230
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, panel), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    def ln(t, y, c=(255, 255, 255), s=0.5):
        cv2.putText(frame, t, (10, y), font, s, c, 1, cv2.LINE_AA)

    ln(f"TARGET: {target or '-'}   [1]K [2]N [3]R [4]U [5]V [6]X   "
       f"S=save  C=clear  Q=quit   FPS {fps:4.1f}", 20, (0, 255, 255), 0.5)
    ln(f"handedness={handedness}   mirror={'ON' if mirror else 'OFF'}", 40,
       (200, 200, 200), 0.46)
    cv2.putText(frame, "CURRENT", (10, 66), font, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "V2", (w // 2 + 10, 66), font, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    if cur and v2:
        for i, (label, r) in enumerate((("current", cur), ("v2", v2))):
            x = 10 if i == 0 else w // 2 + 10
            y0 = 92
            cv2.putText(frame, f"top1: {r['top1']} {r['conf']*100:.0f}%", (x, y0),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"top2: {r['top2']} {r['top2_prob']*100:.0f}%", (x, y0 + 22),
                        font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"margin: {r['margin']*100:.0f}%", (x, y0 + 44),
                        font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
            status = "ACCEPT" if r["accepted"] else "REJECT"
            color = (0, 255, 0) if r["accepted"] else (0, 100, 255)
            if target:
                status += f"  {'OK' if r['correct'] else 'WRONG'}"
                color = (0, 255, 0) if r["correct"] and r["accepted"] else color
            cv2.putText(frame, f"status: {status}", (x, y0 + 66), font, 0.5, color, 1, cv2.LINE_AA)
        # ringkasan R/U/V
        for i, cls in enumerate(("R", "U", "V")):
            c = counts.get(cls, 0)
            cv2.putText(frame, f"{cls}={c}", (10 + i * 60, 210), font, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)
    if message:
        cv2.putText(frame, message, (10, panel - 6), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--mirror", dest="mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    args = ap.parse_args(argv)

    print("== Static hard-class diagnostic: CURRENT vs V2 ==")
    print("Pilih target (1-6) SEBELUM menekan S. Q keluar.")
    bundles = load_models()
    landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print("ERROR: kamera tidak dapat dibuka.", file=sys.stderr)
        landmarker.close()
        return 1

    target = None
    counts = {c: 0 for c in HARD}
    message = ""
    prev_t = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))
            cur = v2 = None
            feats = None
            handedness, hand_score = "-", 0.0
            if res.hand_landmarks:
                land = res.hand_landmarks[0]
                feats = rt.normalize_landmarks(land)          # SATU kali
                cur = predict(bundles["current"], feats, target)
                v2 = predict(bundles["v2"], feats, target)
                if res.handedness and res.handedness[0]:
                    handedness = str(res.handedness[0][0].category_name)
                    hand_score = float(res.handedness[0][0].score)
                rt.draw_hand(frame, land, rt.compute_bbox(land, w, h),
                             cur["top1"], cur["accepted"])
            draw(frame, target, cur, v2, handedness, args.mirror, counts, message, fps)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                target = HARD[key - ord("1")]
                message = f"target = {target}"
            elif key == ord("c"):
                target = None
            elif key in (ord("s"), 32):
                if target is None:
                    message = "Pilih target aktual terlebih dahulu."
                    print(message)
                elif feats is None:
                    message = "Tidak ada tangan; tidak disimpan."
                    print(message)
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    d = OUT_DIR / target
                    d.mkdir(parents=True, exist_ok=True)
                    npz = d / f"{ts}.npz"
                    np.savez_compressed(
                        npz, features=feats.astype(np.float32), target=target,
                        handedness=handedness, handedness_score=np.float32(hand_score),
                        mirror=bool(args.mirror), timestamp=ts, fps=np.float32(fps),
                        current_top1=str(cur["top1"]), current_conf=np.float32(cur["conf"]),
                        current_top2=str(cur["top2"]), current_top2_prob=np.float32(cur["top2_prob"]),
                        current_margin=np.float32(cur["margin"]), current_accepted=bool(cur["accepted"]),
                        current_correct=bool(cur["correct"]),
                        v2_top1=str(v2["top1"]), v2_conf=np.float32(v2["conf"]),
                        v2_top2=str(v2["top2"]), v2_top2_prob=np.float32(v2["top2_prob"]),
                        v2_margin=np.float32(v2["margin"]), v2_accepted=bool(v2["accepted"]),
                        v2_correct=bool(v2["correct"]))
                    row = {
                        "timestamp": ts, "target": target, "handedness": handedness,
                        "hand_score": round(hand_score, 4), "mirror": args.mirror,
                        "npz_path": npz.relative_to(ROOT).as_posix(),
                        "current_top1": cur["top1"], "current_conf": round(cur["conf"], 6),
                        "current_top2": cur["top2"], "current_top2_prob": round(cur["top2_prob"], 6),
                        "current_margin": round(cur["margin"], 6),
                        "current_accepted": cur["accepted"], "current_correct": cur["correct"],
                        "v2_top1": v2["top1"], "v2_conf": round(v2["conf"], 6),
                        "v2_top2": v2["top2"], "v2_top2_prob": round(v2["top2_prob"], 6),
                        "v2_margin": round(v2["margin"], 6),
                        "v2_accepted": v2["accepted"], "v2_correct": v2["correct"],
                    }
                    new = not CSV_PATH.exists()
                    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
                        wtr = csv.DictWriter(fh, fieldnames=CSV_COLS)
                        if new:
                            wtr.writeheader()
                        wtr.writerow(row)
                    counts[target] += 1
                    message = f"saved {target} #{counts[target]}"
                    print(f"saved {target} -> {npz.relative_to(ROOT)}")
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print(f"Saved counts: {counts}")
    print(f"CSV: {CSV_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
