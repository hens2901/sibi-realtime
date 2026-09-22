"""Uji webcam untuk hybrid router STATIC (MLP) + DYNAMIC (GRU J/Z).

TERPISAH dari realtime.py dan app_streamlit.py (tidak diubah).

Fitur diagnostik:
- alasan rejection per-kriteria (CONF / MARGIN / MOTION / DIST / VARIATION / STEPS);
- log setiap dynamic event ke JSONL;
- opsi --save-dynamic-debug untuk menyimpan sequence (raw + resampled) + metadata.

Kontrol:
    Q = keluar
    R = reset state router

Jalankan:
    python hybrid_realtime_test.py
    python hybrid_realtime_test.py --save-dynamic-debug
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import realtime as rt  # noqa: E402
from hybrid_router import HybridRouter  # noqa: E402

WINDOW = "SIBI Hybrid Router Test (STATIC + J/Z)"
DEBUG_DIR = ROOT / "reports" / "realtime_dynamic_debug"


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


def draw_guide(frame):
    h, w = frame.shape[:2]
    gw, gh = int(w * 0.55), int(h * 0.70)
    x0, y0 = (w - gw) // 2, (h - gh) // 2
    x1, y1 = x0 + gw, y0 + gh
    arm = max(16, int(min(gw, gh) * 0.12))
    overlay = frame.copy()
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(overlay, (cx, cy), (cx + dx * arm, cy), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (cx, cy + dy * arm), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "Posisikan satu tangan di area ini", (x0, max(18, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _fmt_check(name, chk):
    if chk is None:
        return f"{name}: -"
    ok = "PASS" if chk.get("pass") else "FAIL"
    val = chk.get("value")
    if isinstance(val, dict):
        val = "/".join(f"{k}={v}" for k, v in val.items())
    elif isinstance(val, float):
        val = round(val, 3)
    if "range" in chk:
        thr = f"range {chk['range'][0]:.2f}-{chk['range'][1]:.2f}"
    elif "threshold" in chk:
        thr = f"thr {chk['threshold']}"
    else:
        thr = ""
    return f"{name}: {ok} ({val} | {thr})"


def draw_overlay(frame, out, fps, counters, last_event, profile="current"):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_h = 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def line(txt, y, color=(255, 255, 255), scale=0.46):
        cv2.putText(frame, txt, (8, y), font, scale, color, 1, cv2.LINE_AA)

    line(f"Profile: {profile.upper()}   State: {out.state}   Route: {out.route}   "
         f"Motion type: {out.motion_type}", 18, (0, 255, 255), 0.5)
    line(f"FPS={fps:4.1f}  hand={'Y' if out.hand_found else 'N'}  "
         f"inst_speed={out.instant_speed:.3f}   "
         f"STATIC: {out.static_label or '-'} {out.static_conf*100:.0f}%", 36, (0, 220, 0), 0.45)

    if last_event is not None:
        d = last_event.get("durations_ms", {})
        line(f"GRU: {last_event.get('predicted')}  "
             f"conf={last_event.get('confidence', 0)*100:.1f}%  "
             f"Recording={d.get('recording', 0):.0f}ms  "
             f"Pre-roll={d.get('pre_roll', 0):.0f}ms  "
             f"Post-roll={d.get('post_roll', 0):.0f}ms", 54, (255, 180, 0), 0.44)
        ch = last_event.get("checks", {})
        line(f"Distance={last_event.get('distance')}  "
             f"Movement={last_event.get('movement_magnitude', 0):.3f}  "
             f"Variation={last_event.get('variation', 0):.3f}", 72, (255, 180, 0), 0.44)
        line(f"  {_fmt_check('CONF', ch.get('confidence'))}   "
             f"{_fmt_check('DIST', ch.get('distance'))}", 90)
        line(f"  {_fmt_check('MOTION', ch.get('movement'))}   "
             f"{_fmt_check('VARIATION', ch.get('variation'))}", 108)
        if "trajectory" in ch:
            line(f"  {_fmt_check('TRAJECTORY', ch.get('trajectory'))}", 126)
        final = (f"ACCEPT {last_event.get('predicted')}"
                 if last_event.get("accepted") else "REJECT")
        color = (0, 255, 0) if last_event.get("accepted") else (0, 100, 255)
        line(f"  Final: {final}   near={last_event.get('near')} "
             f"bucket={last_event.get('bucket')}   reason: "
             f"{str(last_event.get('reason',''))[:42]}", 146, color)
    else:
        line("  (belum ada dynamic event)", 60, (170, 170, 170))

    line(f"Output: {out.final_output}", 168, (255, 255, 0), 0.48)
    line(f"J {counters['J_accept']}/{counters['J_attempt']}  "
         f"Z {counters['Z_accept']}/{counters['Z_attempt']}  "
         f"NEG {counters['negative_accept']}/{counters['negative_attempt']}  "
         f"stat->dyn={counters['static_to_dyn']}", 186, (200, 200, 200), 0.44)
    line("Q=quit  R=reset", h - 12, (180, 180, 180), 0.5)


def print_event(ev):
    print(f"\n[{ev.motion_type}] pred={ev.predicted} conf={ev.confidence:.3f} "
          f"margin={ev.margin:.3f}")
    for name in ("confidence", "margin", "movement", "distance", "variation",
                 "moving_steps", "trajectory"):
        chk = ev.checks.get(name)
        if chk:
            ok = "PASS" if chk["pass"] else "FAIL"
            val = chk["value"]
            if isinstance(val, float):
                val = round(val, 3)
            extra = (f"range {chk['range'][0]:.2f}-{chk['range'][1]:.2f}"
                     if "range" in chk else f"thr {chk.get('threshold')}")
            print(f"  {name.upper():12s} {ok}  value={val}  {extra}")
    print(f"  movement={ev.movement_magnitude:.3f} wrist={ev.wrist_movement:.3f} "
          f"fingertip={ev.fingertip_movement:.3f} vel={ev.mean_velocity:.3f} "
          f"maxvel={ev.max_velocity:.3f}")
    print(f"  distance={ev.distance} variation={ev.variation:.4f} "
          f"coverage={ev.coverage:.3f} dir_changes={ev.direction_changes} "
          f"frames={ev.n_frames} fps~{ev.fps_est:.1f}")
    print(f"  durations_ms={ev.durations_ms}")
    print(f"  FINAL: {'ACCEPT ' + str(ev.predicted) if ev.accepted else 'REJECT'}"
          f"  reason='{ev.reason}'")


def save_event(ev, counters, profile="current"):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag = f"{ts}_{profile}_{ev.predicted or 'NA'}_{'acc' if ev.accepted else 'rej'}"
    meta = {
        "timestamp": ts,
        "profile": profile,
        "predicted": ev.predicted,
        "accepted": ev.accepted,
        "reason": ev.reason,
        "confidence": ev.confidence,
        "margin": ev.margin,
        "movement_magnitude": ev.movement_magnitude,
        "wrist_movement": ev.wrist_movement,
        "fingertip_movement": ev.fingertip_movement,
        "mean_velocity": ev.mean_velocity,
        "max_velocity": ev.max_velocity,
        "path_length": ev.path_length,
        "temporal_consistency": ev.temporal_consistency,
        "distance": ev.distance,
        "variation": ev.variation,
        "moving_steps": ev.moving_steps,
        "motion_type": ev.motion_type,
        "coverage": ev.coverage,
        "direction_changes": ev.direction_changes,
        "net_progress": ev.net_progress,
        "n_frames": ev.n_frames,
        "fps_est": ev.fps_est,
        "durations_ms": ev.durations_ms,
        "checks": ev.checks,
    }
    (DEBUG_DIR / f"{tag}.json").write_text(json.dumps(meta, indent=2, default=str),
                                           encoding="utf-8")
    arrays = {}
    if ev.sequence is not None:
        arrays["sequence"] = np.asarray(ev.sequence, dtype=np.float32)
    if ev.sequence_raw is not None:
        arrays["sequence_raw"] = np.asarray(ev.sequence_raw, dtype=np.float32)
    if arrays:
        np.savez_compressed(DEBUG_DIR / f"{tag}.npz", **arrays)
    with (DEBUG_DIR / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, default=str) + "\n")
    return tag


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid router webcam test")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--save-dynamic-debug", action="store_true",
                    help="simpan sequence + metadata tiap dynamic event")
    ap.add_argument("--profile", choices=["current", "c2", "c2_zadaptive"],
                    default="current",
                    help="profil threshold: current (production, default), c2, atau c2_zadaptive")
    args = ap.parse_args(argv)

    print("== SIBI Hybrid Router - webcam test ==")
    print(f"profile={args.profile}  save-dynamic-debug={args.save_dynamic_debug}")
    try:
        router = HybridRouter(profile=args.profile)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR memuat router: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print("ERROR: kamera tidak dapat dibuka.", file=sys.stderr)
        landmarker.close()
        return 1

    counters = Counter({
        "J_attempt": 0, "J_accept": 0, "J_reject": 0,
        "Z_attempt": 0, "Z_accept": 0, "Z_reject": 0,
        "negative_attempt": 0, "negative_accept": 0, "negative_reject": 0,
        "static_to_dyn": 0,
    })
    reasons = Counter()
    last_event = None
    last_latency = 0.0
    prev_t = time.perf_counter()
    fps = 0.0
    prev_state = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: frame tidak terbaca.", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(now * 1000))
            landmarks = res.hand_landmarks[0] if res.hand_landmarks else None

            out = router.update(now, landmarks)

            if prev_state == "STATIC" and out.route == "DYNAMIC":
                counters["static_to_dyn"] += 1
            prev_state = out.state

            if out.dynamic_event is not None:
                ev = out.dynamic_event
                # Klasifikasi pseudo-GT: kelas prototipe terdekat; "negative"
                # bila jarak melampaui p100 training kelas terdekat.
                dJ = ev.dJ if ev.dJ is not None else 1e9
                dZ = ev.dZ if ev.dZ is not None else 1e9
                near = "J" if dJ < dZ else "Z"
                p100 = router.thr.get("train_dist_p100", {"J": 12.0, "Z": 12.0})
                is_negative = (near == "J" and dJ > p100.get("J", 12.0)) or \
                              (near == "Z" and dZ > p100.get("Z", 12.0))
                bucket = "negative" if is_negative else near
                counters[f"{bucket}_attempt"] += 1
                good_accept = ev.accepted and (ev.predicted == near) and not is_negative
                if good_accept:
                    counters[f"{bucket}_accept"] += 1
                    last_latency = out.dynamic_latency_ms
                else:
                    counters[f"{bucket}_reject"] += 1
                    for name, chk in ev.checks.items():
                        if not chk.get("pass"):
                            reasons[name] += 1

                last_event = {
                    "profile": router.profile_name, "predicted": ev.predicted,
                    "near": near, "bucket": bucket, "accepted": ev.accepted,
                    "good_accept": good_accept, "reason": ev.reason,
                    "confidence": ev.confidence, "margin": ev.margin,
                    "movement_magnitude": ev.movement_magnitude,
                    "distance": ev.distance, "dJ": ev.dJ, "dZ": ev.dZ,
                    "variation": ev.variation, "moving_steps": ev.moving_steps,
                    "motion_type": ev.motion_type, "coverage": ev.coverage,
                    "direction_changes": ev.direction_changes,
                    "net_progress": ev.net_progress,
                    "checks": ev.checks, "durations_ms": ev.durations_ms,
                    "n_frames": ev.n_frames, "fps_est": ev.fps_est,
                }
                print_event(ev)
                print(f"  profile={router.profile_name} near={near} bucket={bucket} "
                      f"good_accept={good_accept}")
                if args.save_dynamic_debug:
                    tag = save_event(ev, counters, router.profile_name)
                    print(f"  saved: reports/realtime_dynamic_debug/{tag}")

            if landmarks is not None:
                rt.draw_hand(frame, landmarks, rt.compute_bbox(landmarks, w, h),
                             out.static_label or "?", out.static_recognized)
            else:
                draw_guide(frame)
            draw_overlay(frame, out, fps, counters, last_event, router.profile_name)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                router.reset()
                counters.clear()
                counters.update({
                    "J_attempt": 0, "J_accept": 0, "J_reject": 0,
                    "Z_attempt": 0, "Z_accept": 0, "Z_reject": 0,
                    "negative_attempt": 0, "negative_accept": 0,
                    "negative_reject": 0, "static_to_dyn": 0})
                reasons.clear()
                last_event = None
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print(f"\nSummary profile={args.profile}: "
          f"J {counters['J_accept']}/{counters['J_attempt']} "
          f"(rej {counters['J_reject']}) | "
          f"Z {counters['Z_accept']}/{counters['Z_attempt']} "
          f"(rej {counters['Z_reject']}) | "
          f"NEG {counters['negative_accept']}/{counters['negative_attempt']} "
          f"(rej {counters['negative_reject']}) | "
          f"static->dynamic={counters['static_to_dyn']} "
          f"latency_terakhir={last_latency:.0f}ms")
    print(f"Top rejection reasons: {reasons.most_common()}")
    print("Catatan: hasil webcam nyata harus diuji manual; belum diverifikasi di sini.")
    print("Bucket J/Z = pseudo-GT (prototipe terdekat); NEG = jarak > p100 training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
