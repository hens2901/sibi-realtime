"""Bandingkan runtime Current vs V2 pada input landmark/frame yang sama.

Memvalidasi kedua bundle (load, 63 fitur, 24 kelas, predict_proba) DAN
menjalankan SibiVideoProcessor masing-masing model pada frame yang sama,
melaporkan payload / exception yang sebenarnya.

Jalankan:
    python scripts/compare_current_v2_runtime.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sklearn  # noqa: E402

import app_streamlit as app  # noqa: E402
import realtime as rt  # noqa: E402

HAND_IMG = ROOT / "data" / "raw" / "Mono_Background" / "H" / "H_0.jpg"


def get_frame_bgr():
    if HAND_IMG.exists():
        from PIL import Image
        import cv2
        rgb = np.asarray(Image.open(HAND_IMG).convert("RGB"))
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 480)), "dataset H"
    return np.full((480, 640, 3), 128, dtype=np.uint8), "synthetic"


def validate_bundle(name):
    print(f"\n=== validate bundle: {name} ===")
    try:
        model, scaler, enc, labels = app.load_static_bundle(name)
    except Exception as exc:  # noqa: BLE001
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print(f"  model={type(model).__name__} n_features_in_={getattr(model,'n_features_in_',None)}")
    print(f"  scaler={type(scaler).__name__} n_features_in_={getattr(scaler,'n_features_in_',None)} "
          f"has_names={hasattr(scaler,'feature_names_in_')}")
    print(f"  classes_={np.asarray(model.classes_).tolist()[:5]}... n={len(model.classes_)}")
    print(f"  labels n={len(labels)} sklearn={sklearn.__version__}")
    ok = (getattr(model, "n_features_in_", None) == 63
          and getattr(scaler, "n_features_in_", None) == 63
          and len(labels) == 24
          and np.array_equal(model.classes_, np.arange(24)))
    print(f"  structure OK: {ok}")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, size=63).astype(np.float64)
    try:
        p = rt.predict_proba(model, scaler, x)
        print(f"  predict_proba shape={p.shape} sum={p.sum():.6f} "
              f"finite={np.all(np.isfinite(p))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  predict_proba FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        ok = False
    return ok


def run_processor(name, frame_bgr, bundles):
    import av
    print(f"\n=== processor recv: {name} ===")
    settings = app.RuntimeSettings()
    settings.update(model_id=name, mirror=True)
    shared = app.LatestResult()
    try:
        proc = app.SibiVideoProcessor(bundles, settings, shared)
    except Exception as exc:  # noqa: BLE001
        print(f"  __init__ FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return
    try:
        out = proc.recv(av.VideoFrame.from_ndarray(frame_bgr, format="bgr24"))
        payload = shared.get() or {}
        print(f"  recv OK -> {type(out).__name__}")
        print(f"  payload: hand_found={payload.get('hand_found')} "
              f"top1={payload.get('top1_label')} p={payload.get('top1_prob')} "
              f"error={payload.get('error')}")
    except Exception as exc:  # noqa: BLE001
        print(f"  recv FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        proc.on_ended()


def main() -> int:
    print(f"Python {sys.version.split()[0]} | sklearn {sklearn.__version__}")
    frame, src = get_frame_bgr()
    print(f"frame source: {src} shape={frame.shape}")

    results = {n: validate_bundle(n) for n in ("current", "v2")}

    bundles = {n: app.load_static_bundle(n) for n in ("current", "v2")}
    for name in ("current", "v2"):
        run_processor(name, frame, bundles)

    print("\n=== SUMMARY ===")
    for n, ok in results.items():
        print(f"  bundle {n}: {'OK' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
