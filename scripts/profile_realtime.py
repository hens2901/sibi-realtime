"""Profiling pipeline SIBI realtime per tahap (tanpa webcam).

Mengukur rata-rata waktu (ms) untuk:
- receive/encode frame (av.VideoFrame)
- konversi ke ndarray
- mirror (cv2.flip)
- MediaPipe inference (detect_for_video)
- preprocessing landmark (wrist + scale)
- scaler + MLP prediction
- drawing (landmark/bbox/guide)
- output encode (av.VideoFrame.from_ndarray)
- update shared state

Juga mengukur recv() penuh dari SibiVideoProcessor dan CPU utilization.

Output: reports/perf_profile_<tag>.json  (tag default: current)

Jalankan:
    python scripts/profile_realtime.py --tag before
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av  # noqa: E402
import cv2  # noqa: E402

import mediapipe as mp  # noqa: E402
import realtime as rt  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

DATASET = ROOT / "data" / "raw" / "Mono_Background"
REPORTS = ROOT / "reports"


def pick_hand_image() -> np.ndarray:
    for cls in ("H", "A", "B", "C", "D"):
        for img in sorted((DATASET / cls).glob("*.jpg")):
            rgb = np.asarray(Image.open(img).convert("RGB"))
            return np.ascontiguousarray(rgb[:, :, ::-1])
    raise RuntimeError("tidak ada gambar dataset")


def to_frame_size(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


def micro_benchmark(bgr: np.ndarray, frames: int = 60) -> dict:
    """Ukur tiap tahap secara terpisah (primitif yang sama di semua versi)."""
    landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
    model, scaler, encoder, labels = rt.load_artifacts("augmented")
    from app_streamlit import LatestResult, draw_frame_overlay

    shared = LatestResult()
    acc = {k: 0.0 for k in (
        "receive_encode", "to_ndarray", "mirror", "mp_image", "mediapipe",
        "preprocess", "predict", "draw", "output_encode", "shared_update",
    )}
    h, w = bgr.shape[:2]

    try:
        for i in range(frames):
            t = time.perf_counter()
            vf = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            acc["receive_encode"] += time.perf_counter() - t

            t = time.perf_counter()
            nd = vf.to_ndarray(format="bgr24")
            acc["to_ndarray"] += time.perf_counter() - t

            t = time.perf_counter()
            img = cv2.flip(nd, 1)
            acc["mirror"] += time.perf_counter() - t

            t = time.perf_counter()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(img[:, :, ::-1]))
            acc["mp_image"] += time.perf_counter() - t

            t = time.perf_counter()
            res = landmarker.detect_for_video(mp_image, i * 33)
            acc["mediapipe"] += time.perf_counter() - t

            result = None
            if res.hand_landmarks:
                hand = res.hand_landmarks[0]

                t = time.perf_counter()
                feats = rt.normalize_landmarks(hand)
                acc["preprocess"] += time.perf_counter() - t

                t = time.perf_counter()
                rt.predict_proba(model, scaler, feats)
                acc["predict"] += time.perf_counter() - t

                class _R:
                    pass
                result = _R()
                result.landmarks = hand
                result.bbox = rt.compute_bbox(hand, w, h)
                result.label = "H"
                result.recognized = True
                result.hand_found = True
            else:
                result = None

            t = time.perf_counter()
            draw_frame_overlay(img, result, bool(res.hand_landmarks))
            acc["draw"] += time.perf_counter() - t

            t = time.perf_counter()
            out = av.VideoFrame.from_ndarray(img, format="bgr24")
            acc["output_encode"] += time.perf_counter() - t

            t = time.perf_counter()
            shared.set({"i": i, "label": "H"})
            acc["shared_update"] += time.perf_counter() - t
    finally:
        landmarker.close()

    return {k: (v / frames) * 1000.0 for k, v in acc.items()}


def recv_benchmark(bgr: np.ndarray, frames: int = 60) -> dict:
    """Ukur recv() penuh dari SibiVideoProcessor yang aktif di app_streamlit."""
    import app_streamlit as app

    bundles = {n: rt.load_artifacts(n) for n in ("baseline", "augmented")}
    settings = app.RuntimeSettings()
    settings.update(model_id="augmented", mirror=True)
    shared = app.LatestResult()
    proc = app.SibiVideoProcessor(bundles, settings, shared)

    vf = av.VideoFrame.from_ndarray(bgr, format="bgr24")
    # warmup
    for _ in range(5):
        proc.recv(vf)

    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    for _ in range(frames):
        proc.recv(vf)
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    proc.on_ended()

    return {
        "recv_ms_avg": (wall / frames) * 1000.0,
        "recv_fps": frames / wall if wall else 0.0,
        "cpu_utilization": cpu / wall if wall else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="current")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    src = pick_hand_image()

    report = {"tag": args.tag, "frames": args.frames, "resolutions": {}}
    for (w, h) in ((640, 480), (1280, 720)):
        bgr = to_frame_size(src, w, h)
        micro = micro_benchmark(bgr, args.frames)
        recv = recv_benchmark(bgr, max(20, args.frames // 2))
        total_stage = sum(micro.values())
        report["resolutions"][f"{w}x{h}"] = {
            "micro_ms": micro,
            "micro_total_ms": total_stage,
            "recv": recv,
        }
        print(f"\n== {w}x{h} ==")
        for k, v in micro.items():
            print(f"  {k:16s} {v:7.2f} ms")
        print(f"  {'STAGE TOTAL':16s} {total_stage:7.2f} ms")
        print(f"  {'recv() full':16s} {recv['recv_ms_avg']:7.2f} ms "
              f"({recv['recv_fps']:.1f} FPS, CPU {recv['cpu_utilization']*100:.0f}%)")

    out = REPORTS / f"perf_profile_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
