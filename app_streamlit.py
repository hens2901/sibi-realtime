"""SIBI Real-Time Sign Language Recognition - aplikasi Streamlit.

Mengenali abjad jari SIBI (24 gesture statis: A-I dan K-Y) dari webcam
menggunakan MediaPipe HandLandmarker + MLPClassifier yang sudah ada.

Prinsip:
- TIDAK melatih ulang model dan TIDAK mengubah model/scaler/encoder.
- Preprocessing IDENTIK dengan `realtime.py` (fungsi diimpor, tidak diduplikasi).
- Model statis default: V2 (Current tersedia sebagai fallback). Mirror default: ON.
  Threshold 0.85 / margin 0.20.

Jalankan:
    python -m streamlit run app_streamlit.py
"""

from __future__ import annotations

import html
import os
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import realtime as rt
import mediapipe as mp
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent

APP_TITLE = "SIBI Real-Time Sign Language Recognition"
APP_SUBTITLE = "Kenali abjad jari SIBI secara realtime melalui kamera"

DEFAULT_MODEL_LABEL = "V2"                 # default aplikasi (static model)
STATIC_MODEL_MAP = {"V2": "v2", "Current": "current"}
STATIC_ID_TO_LABEL = {v: k for k, v in STATIC_MODEL_MAP.items()}
MODEL_LABEL_TO_ID = STATIC_MODEL_MAP       # alias
MODEL_HELP = (
    "V2: Model hasil augmentasi lanjutan untuk meningkatkan robustness pada "
    "beberapa gesture sulit. Current: Model baseline sebelumnya."
)
MODELS_DIR = ROOT / "models"
STATIC_MODEL_FILES = {
    "v2": (MODELS_DIR / "sibi_mlp_augmented_v2.joblib",
           MODELS_DIR / "scaler_augmented_v2.joblib",
           MODELS_DIR / "label_encoder_augmented_v2.joblib"),
    "current": (MODELS_DIR / "sibi_mlp_augmented.joblib",
                MODELS_DIR / "scaler_augmented.joblib",
                MODELS_DIR / "label_encoder_augmented.joblib"),
}

DEFAULT_MIRROR = True
DEFAULT_THRESHOLD = 0.85
DEFAULT_MARGIN = 0.20
DEFAULT_SMOOTHING = rt.SMOOTHING_WINDOW
DEFAULT_STABLE_FRAMES = rt.STABLE_FRAMES

# Auto-refresh panel (A/B): SIBI_UI_REFRESH=off|1.0s|0.5s|0.2s (default 1.0s).
_ui_refresh = (os.environ.get("SIBI_UI_REFRESH", "1.0s") or "1.0s").strip().lower()
AUTOREFRESH = _ui_refresh not in ("off", "0", "false", "no", "")
FRAGMENT_INTERVAL = (_ui_refresh if _ui_refresh not in ("on", "1", "true", "yes")
                     else "1.0s") if AUTOREFRESH else None

# Logging diagnostik processor: SIBI_PROC_LOG=on -> log siklus + exception penuh.
PROC_LOG = (os.environ.get("SIBI_PROC_LOG", "off") or "off").strip().lower() \
    in ("on", "1", "true", "yes")


def plog(msg: str) -> None:
    """Log diagnostik processor (bukan per frame) ke stderr."""
    if PROC_LOG:
        print(f"[SIBI-PROC] {msg}", file=sys.stderr, flush=True)


# Diagnostik lifecycle (thread-safe).
_PROC_LOCK = threading.Lock()
_PROC_CREATED_COUNT = 0
_FIRST_FRAME_COUNT = 0
_PROC_CREATED_TS = 0.0
_FIRST_FRAME_TS = 0.0

# SIBI_ASYNC_PROCESSING=on/off (A/B test async_processing).
ASYNC_PROCESSING = (os.environ.get("SIBI_ASYNC_PROCESSING", "on") or "on").strip().lower() \
    not in ("off", "0", "false", "no")


RESULT_TTL_S = 1.5            # payload dianggap basi setelah ini (detik)
MIN_CONFIDENCE, MAX_CONFIDENCE = 0.50, 0.99
MIN_MARGIN, MAX_MARGIN = 0.00, 0.80

# Capture default ringan untuk realtime (bukan 720p/1080p).
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS_IDEAL = 24

# Mode performa: interval inference (proses tiap N frame) + ukuran inference.
PERF_MODES = {
    "Seimbang": {"interval": 2, "inference_size": None},
    "Kualitas": {"interval": 1, "inference_size": None},
    "Cepat": {"interval": 3, "inference_size": (480, 360)},
}
DEFAULT_PERF_MODE = "Seimbang"

SUPPORTED_LETTERS = list(rt.SUPPORTED_CLASSES)
UNSUPPORTED_LETTERS = list(rt.UNSUPPORTED_CLASSES)

# Akurasi offline dari eksperimen.
# v2: reports/static_v2_comparison.csv ; current: hasil augmented sebelumnya.
OFFLINE_ACC = {
    "v2": {"original": "96.1%", "macro_f1": "96.1%", "mirrored_proxy": "95.8%"},
    "current": {"original": "94.0%", "macro_f1": "94.1%", "mirrored": "90.5%"},
}

CAMERA_GUIDE_TEXT = (
    "Arahkan satu tangan ke kamera dan pastikan seluruh jari terlihat."
)

DISCLAIMER = (
    "Aplikasi ini merupakan prototype penelitian untuk pengenalan abjad jari "
    "SIBI dan bukan pengganti penerjemah bahasa isyarat profesional."
)


# --------------------------------------------------------------------------- #
# Shared state lintas-thread (video callback <-> UI)
# --------------------------------------------------------------------------- #

class LatestResult:
    """Menyimpan hasil inference terbaru secara thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict | None = None

    def set(self, payload: dict) -> None:
        with self._lock:
            self._data = payload

    def get(self) -> dict | None:
        with self._lock:
            return self._data

    def clear(self) -> None:
        with self._lock:
            self._data = None


class RuntimeSettings:
    """Pengaturan runtime yang dapat dibaca thread video (mirror, threshold...)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model_id = MODEL_LABEL_TO_ID[DEFAULT_MODEL_LABEL]
        self.mirror = DEFAULT_MIRROR
        self.threshold = DEFAULT_THRESHOLD
        self.margin = DEFAULT_MARGIN
        self.smoothing_window = DEFAULT_SMOOTHING
        self.stable_frames = DEFAULT_STABLE_FRAMES
        self.show_chart = False
        self.perf_mode = DEFAULT_PERF_MODE
        self.inference_interval = PERF_MODES[DEFAULT_PERF_MODE]["interval"]
        self.inference_size = PERF_MODES[DEFAULT_PERF_MODE]["inference_size"]

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model_id": self.model_id,
                "mirror": self.mirror,
                "threshold": self.threshold,
                "margin": self.margin,
                "smoothing_window": self.smoothing_window,
                "stable_frames": self.stable_frames,
                "show_chart": self.show_chart,
                "perf_mode": self.perf_mode,
                "inference_interval": int(self.inference_interval),
                "inference_size": self.inference_size,
            }


# --------------------------------------------------------------------------- #
# Model bundle (dimuat sekali, dipakai bersama semua sesi - read-only)
# --------------------------------------------------------------------------- #

def missing_runtime_assets() -> list[str]:
    """Daftar aset runtime yang hilang (path relatif ke project)."""
    files = []
    for name in ("v2", "current"):
        files.extend(list(STATIC_MODEL_FILES[name]))
    files.append(rt.LANDMARKER_PATH)
    return [p.relative_to(ROOT).as_posix() for p in files if not p.exists()]


@st.cache_resource(show_spinner=False)
def get_bundles() -> dict:
    """Muat model statis terpilih (V2 + Current) sekali. Read-only/thread-safe."""
    return {name: load_static_bundle(name) for name in ("v2", "current")}


def load_static_bundle(name: str):
    """Muat (model, scaler, encoder, labels) untuk nama static model."""
    import joblib
    model_path, scaler_path, enc_path = STATIC_MODEL_FILES[name]
    for p in (model_path, scaler_path, enc_path):
        if not p.exists():
            raise FileNotFoundError(f"Artefak model statis tidak ditemukan: {p}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(enc_path)
    labels = [str(x) for x in encoder.classes_]
    if getattr(model, "n_features_in_", None) != rt.NUM_FEATURES:
        raise ValueError(f"Model {name} bukan 63 fitur.")
    if getattr(scaler, "n_features_in_", None) != rt.NUM_FEATURES:
        raise ValueError(f"Scaler {name} bukan 63 fitur.")
    if len(labels) != len(rt.SUPPORTED_CLASSES):
        raise ValueError(f"Encoder {name} bukan 24 kelas.")
    if not np.array_equal(model.classes_, np.arange(len(labels))):
        raise ValueError(f"Urutan kelas model {name} tidak konsisten dengan encoder.")
    return model, scaler, encoder, labels


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def _draw_center_guide(frame: np.ndarray) -> None:
    """Guide minimal (corner brackets) saat tanpa tangan.

    Sengaja bukan kotak penuh agar tidak terlihat seperti face-detection box,
    cukup besar untuk satu tangan, dan tidak pernah menutupi landmark karena
    hanya muncul ketika belum ada tangan.
    """
    h, w = frame.shape[:2]
    gw, gh = int(w * 0.55), int(h * 0.70)
    x0, y0 = (w - gw) // 2, (h - gh) // 2
    x1, y1 = x0 + gw, y0 + gh
    arm = max(18, int(min(gw, gh) * 0.12))
    color = (255, 255, 255)
    thickness = 3

    overlay = frame.copy()
    corners = (
        (x0, y0, 1, 1), (x1, y0, -1, 1),
        (x0, y1, 1, -1), (x1, y1, -1, -1),
    )
    for cx, cy, dx, dy in corners:
        cv2.line(overlay, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    text = "Posisikan tangan di area ini"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
    ty = y1 + th + 16
    if ty > h - 8:
        ty = y0 - 12
    cv2.putText(frame, text, ((w - tw) // 2, ty), font, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_frame_overlay(frame: np.ndarray, result, hand_found: bool) -> None:
    """Overlay ringan: guide saat tanpa tangan, skeleton+bbox+label saat ada tangan."""
    if not hand_found:
        _draw_center_guide(frame)
        return
    if result is not None and getattr(result, "landmarks", None):
        rt.draw_hand(
            frame, result.landmarks, result.bbox,
            result.label or "?", bool(result.recognized),
        )


# --------------------------------------------------------------------------- #
# Fast inference helpers (numerically identik dengan realtime.predict_proba)
# --------------------------------------------------------------------------- #

def fast_predict_proba(model, scaler, features) -> np.ndarray:
    """Scaler + MLP tanpa membangun DataFrame per frame.

    Hasil angka IDENTIK dengan `realtime.predict_proba`; DataFrame hanya dipakai
    untuk menghindari warning nama fitur. Jauh lebih cepat di jalur panas.
    """
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != rt.NUM_FEATURES:
        raise ValueError(
            f"Input harus {rt.NUM_FEATURES} fitur, ditemukan {x.shape[1]}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Input fitur mengandung NaN/inf.")
    if hasattr(scaler, "feature_names_in_"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_scaled = scaler.transform(x)
    else:
        x_scaled = scaler.transform(x)
    return model.predict_proba(x_scaled)[0]


def _ema(prev: float, new: float, alpha: float = 0.1) -> float:
    return new if prev == 0.0 else (1.0 - alpha) * prev + alpha * new


# --------------------------------------------------------------------------- #
# Video processor
# --------------------------------------------------------------------------- #

class SibiVideoProcessor(VideoProcessorBase):
    """Pipeline inference SIBI dengan throttling + metrik performa.

    - Mirror diterapkan sebelum inference.
    - Inference (MediaPipe + ANN) dijalankan tiap N frame; frame lain memakai
      hasil terakhir agar video tetap lancar (tidak berkedip).
    - Temporal smoothing hanya di-update pada frame inference (logis).
    - Model/scaler/encoder/landmarker reusable (tidak dibuat per frame).
    - Semua angka identik dengan `realtime.py` (preprocessing & rejection sama).
    """

    def __init__(self, bundles: dict, settings: RuntimeSettings,
                 shared: LatestResult) -> None:
        # __init__ HARUS ringan: jangan buat MediaPipe/model di sini, karena
        # berjalan saat WebRTC establishment dan bisa menghambat media.
        global _PROC_CREATED_COUNT, _PROC_CREATED_TS
        self.bundles = bundles
        self.settings = settings
        self.shared = shared
        self.landmarker = None
        self.model = self.scaler = self.encoder = None
        self.labels: list[str] = []
        self.smoother = None
        self._engine_key: tuple | None = None
        self._last: "rt.FrameResult | None" = None
        self._first_frame_logged = False
        self._detect_logged = False
        self._init_lock = threading.Lock()
        self.initialized = False
        self.landmarker_ready = False
        self._created_ts = time.perf_counter()
        self._first_frame_ts = 0.0

        with _PROC_LOCK:
            _PROC_CREATED_COUNT += 1
            _PROC_CREATED_TS = self._created_ts
            created_count = _PROC_CREATED_COUNT
        snap0 = settings.snapshot()
        plog(f"PROCESSOR CREATED model_id={snap0.get('model_id')} "
             f"count={created_count} t={self._created_ts:.3f}")

        # metrik
        self._prev_t = time.perf_counter()
        self._prev_infer_t = 0.0
        self._frame_latency = 0.0
        self._infer_latency = 0.0
        self._mediapipe_ms = 0.0
        self._predict_ms = 0.0
        self._draw_ms = 0.0
        self._camera_fps = 0.0
        self._inference_fps = 0.0
        self._frame_idx = 0
        self._infer_count = 0
        self._skipped = 0

    def _ensure_initialized(self, snap: dict) -> None:
        """Lazy init (landmarker + engine) di first frame, thread-safe sekali."""
        if self.initialized:
            return
        with self._init_lock:
            if self.initialized:
                return
            plog("LAZY LANDMARKER INIT START")
            try:
                self.landmarker = rt.build_landmarker(vision.RunningMode.VIDEO)
            except Exception:  # noqa: BLE001
                plog("LAZY LANDMARKER INIT FAILED:\n" + traceback.format_exc())
                raise
            self.landmarker_ready = True
            plog("LAZY LANDMARKER INIT OK")
            self._sync_engine(snap)
            self.initialized = True
            plog("LAZY INIT DONE")

    def _sync_engine(self, snap: dict) -> None:
        key = (snap["model_id"], snap["smoothing_window"], snap["stable_frames"])
        if self.smoother is not None and key == self._engine_key:
            return
        plog(f"engine sync: model_id={snap['model_id']} (bundles has key: "
             f"{snap['model_id'] in self.bundles})")
        try:
            model, scaler, encoder, labels = self.bundles[snap["model_id"]]
        except Exception:  # noqa: BLE001
            plog("engine sync FAILED:\n" + traceback.format_exc())
            raise
        self.model, self.scaler, self.encoder = model, scaler, encoder
        self.labels = list(labels)
        self.smoother = rt.TemporalSmoother(
            self.labels, window=snap["smoothing_window"],
            stable_frames=snap["stable_frames"],
        )
        self._engine_key = key
        self._last = None
        plog(f"ENGINE SYNC OK model_id={snap['model_id']} "
             f"n_features={getattr(model,'n_features_in_',None)} labels={len(self.labels)}")

    def _empty_outcome(self) -> "rt.FrameResult":
        return rt.FrameResult(False, None, 0.0, False, False, None, None)

    def _infer(self, infer_img: np.ndarray, out_w: int, out_h: int,
               snap: dict, ts_ms: int) -> tuple["rt.FrameResult", float, float]:
        """Jalankan MediaPipe + ANN pada satu frame. Kembalikan (outcome, mp_ms, pred_ms)."""
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = self.landmarker.detect_for_video(mp_image, ts_ms)
        mp_ms = (time.perf_counter() - t0) * 1000.0
        if det.hand_landmarks and not getattr(self, "_detect_logged", False):
            plog(f"MEDIAPIPE DETECT OK hands={len(det.hand_landmarks)}")
            self._detect_logged = True

        if not det.hand_landmarks:
            self.smoother.update(None)
            return self._empty_outcome(), mp_ms, 0.0

        hand = det.hand_landmarks[0]
        t1 = time.perf_counter()
        try:
            feats = rt.normalize_landmarks(hand)
            plog(f"FEATURES shape={np.shape(feats)}")
            probs = fast_predict_proba(self.model, self.scaler, feats)
            plog(f"PREDICT OK model_id={self._engine_key[0] if self._engine_key else '?'} "
                 f"probs_shape={np.shape(probs)} sum={float(np.sum(probs)):.4f} "
                 f"finite={bool(np.all(np.isfinite(probs)))}")
        except ValueError:
            self.smoother.update(None)
            return self._empty_outcome(), mp_ms, (time.perf_counter() - t1) * 1000.0
        except Exception:  # noqa: BLE001 - log exception sebenarnya
            plog("PREDICT FAILED (unexpected):\n" + traceback.format_exc())
            raise
        pred_ms = (time.perf_counter() - t1) * 1000.0

        bbox = rt.compute_bbox(hand, out_w, out_h)
        top2 = rt.top2_from_probs(probs, self.labels)
        accepted = rt.passes_rejection(
            top2.top1_prob, top2.margin, snap["threshold"], snap["margin"]
        )

        if not accepted:
            self.smoother.update(None)
            outcome = rt.FrameResult(
                True, None, top2.top1_prob, False, False, hand, bbox,
                top1_label=top2.top1_label, top1_prob=top2.top1_prob,
                top2_label=top2.top2_label, top2_prob=top2.top2_prob,
                margin=top2.margin, rejected=True,
            )
        else:
            label, conf, stable = self.smoother.update(probs)
            outcome = rt.FrameResult(
                True, label, conf, True, stable, hand, bbox,
                top1_label=top2.top1_label, top1_prob=top2.top1_prob,
                top2_label=top2.top2_label, top2_prob=top2.top2_prob,
                margin=top2.margin, rejected=False,
            )
        return outcome, mp_ms, pred_ms

    def _payload(self, outcome: "rt.FrameResult", now: float, *,
                 top5: list | None = None, error: str | None = None,
                 error_detail: str | None = None) -> dict:
        return {
            "t": now,
            "hand_found": bool(outcome.hand_found),
            "label": outcome.label,
            "confidence": float(outcome.confidence),
            "recognized": bool(outcome.recognized),
            "stable": bool(outcome.stable),
            "rejected": bool(outcome.rejected),
            "top1_label": outcome.top1_label,
            "top1_prob": float(outcome.top1_prob),
            "top2_label": outcome.top2_label,
            "top2_prob": float(outcome.top2_prob),
            "margin": float(outcome.margin),
            "top5": top5 or [],
            "error": error,
            "error_detail": error_detail,
            # metrik performa
            "camera_fps": self._camera_fps,
            "inference_fps": self._inference_fps,
            "frame_latency_ms": self._frame_latency,
            "inference_latency_ms": self._infer_latency,
            "mediapipe_ms": self._mediapipe_ms,
            "predict_ms": self._predict_ms,
            "draw_ms": self._draw_ms,
            "infer_count": self._infer_count,
            "skipped_frames": self._skipped,
            # diagnostik lifecycle
            "model_id": self._engine_key[0] if self._engine_key else None,
            "initialized": bool(self.initialized),
            "processor_created_count": _PROC_CREATED_COUNT,
            "first_frame_count": _FIRST_FRAME_COUNT,
            "processor_created_ts": _PROC_CREATED_TS,
            "first_frame_ts": _FIRST_FRAME_TS,
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global _FIRST_FRAME_COUNT, _FIRST_FRAME_TS
        frame_start = time.perf_counter()
        img = frame.to_ndarray(format="bgr24")
        # Log RAW frame SEBELUM heavy init, agar bisa membedakan:
        #   - tidak ada frame WebRTC, atau
        #   - frame ada tetapi lazy init macet.
        if not self._first_frame_logged:
            self._first_frame_logged = True
            self._first_frame_ts = time.perf_counter()
            with _PROC_LOCK:
                _FIRST_FRAME_COUNT += 1
                _FIRST_FRAME_TS = self._first_frame_ts
                ff_count = _FIRST_FRAME_COUNT
            plog(f"FIRST FRAME RAW RECEIVED shape={img.shape} count={ff_count} "
                 f"t={self._first_frame_ts:.3f}")
        snap = self.settings.snapshot()
        h, w = img.shape[:2]
        try:
            self._ensure_initialized(snap)
            if snap["mirror"]:
                img = cv2.flip(img, 1)

            now = time.perf_counter()
            dt = now - self._prev_t
            self._prev_t = now
            if dt > 0:
                self._camera_fps = _ema(self._camera_fps, 1.0 / dt)

            self._frame_idx += 1
            interval = max(1, int(snap["inference_interval"]))
            do_infer = (self._last is None) or (self._frame_idx % interval == 0)

            if do_infer:
                infer_img = img
                size = snap.get("inference_size")
                if size and (w, h) != tuple(size):
                    infer_img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
                outcome, mp_ms, pred_ms = self._infer(infer_img, w, h, snap, int(now * 1000))
                self._last = outcome
                self._infer_count += 1
                if self._prev_infer_t:
                    idt = now - self._prev_infer_t
                    if idt > 0:
                        self._inference_fps = _ema(self._inference_fps, 1.0 / idt)
                self._prev_infer_t = now
                self._mediapipe_ms = _ema(self._mediapipe_ms, mp_ms)
                self._predict_ms = _ema(self._predict_ms, pred_ms)
                self._infer_latency = _ema(self._infer_latency, mp_ms + pred_ms)
            else:
                self._skipped += 1
                outcome = self._last

            # Chart opsional: hitung top-5 hanya bila diminta.
            top5: list[tuple[str, float]] = []
            if snap.get("show_chart") and outcome.hand_found and outcome.landmarks is not None:
                try:
                    probs = fast_predict_proba(
                        self.model, self.scaler,
                        rt.normalize_landmarks(outcome.landmarks),
                    )
                    order = np.argsort(probs)[::-1][:5]
                    top5 = [(self.labels[int(i)], float(probs[int(i)])) for i in order]
                except Exception:  # noqa: BLE001 - chart opsional
                    top5 = []

            self.shared.set(self._payload(outcome, now, top5=top5))

            t_draw = time.perf_counter()
            draw_frame_overlay(img, outcome, outcome.hand_found)
            self._draw_ms = _ema(self._draw_ms, (time.perf_counter() - t_draw) * 1000.0)

            self._frame_latency = _ema(
                self._frame_latency, (time.perf_counter() - frame_start) * 1000.0
            )
        except Exception as exc:  # noqa: BLE001 - jangan crash seluruh app
            tb = traceback.format_exc()
            plog("recv EXCEPTION:\n" + tb)
            outcome = self._last or self._empty_outcome()
            self.shared.set(self._payload(
                outcome, time.perf_counter(),
                error=f"{type(exc).__name__}: {exc}", error_detail=tb,
            ))
            try:
                draw_frame_overlay(img, outcome, outcome.hand_found)
            except Exception:  # noqa: BLE001
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self) -> None:
        # Reset prediksi aktif saat kamera berhenti agar UI tidak menampilkan
        # hasil lama. Hasil ejaan pengguna tidak disentuh.
        try:
            self.shared.clear()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.landmarker.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #

def apply_spelling_action(spelling: str, action: str, letter: str | None = None) -> str:
    """Logika murni untuk aksi hasil ejaan (mudah diuji tanpa runtime Streamlit)."""
    if action == "add":
        return spelling + (letter or "")
    if action == "space":
        return spelling + " "
    if action == "delete":
        return spelling[:-1]
    if action == "reset":
        return ""
    return spelling


def init_state() -> None:
    st.session_state.setdefault("spelling", "")
    st.session_state.setdefault("shared", LatestResult())
    st.session_state.setdefault("runtime_settings", RuntimeSettings())


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader(":material/tune: Pengaturan")
        st.caption("Pengaturan awal sudah optimal. Kamu bisa langsung memakai "
                   "aplikasi tanpa mengubah apa pun.")

        model_label = st.segmented_control(
            "Model statis", list(STATIC_MODEL_MAP.keys()),
            default=DEFAULT_MODEL_LABEL,
            key="model_label", help=MODEL_HELP,
        ) or DEFAULT_MODEL_LABEL

        mirror = st.toggle(
            "Mode cermin (mirror)", value=DEFAULT_MIRROR, key="opt_mirror",
            help="Membalik tampilan kamera seperti cermin. Direkomendasikan aktif.",
        )

        with st.expander("Pengaturan lanjutan", icon=":material/tune:"):
            st.caption("Untuk kebutuhan eksperimen/penelitian.")

            perf_mode = st.segmented_control(
                "Mode performa",
                list(PERF_MODES.keys()),
                default=DEFAULT_PERF_MODE,
                key="opt_perf_mode",
                help="Seimbang: inference tiap 2 frame (disarankan). "
                     "Kualitas: tiap frame. Cepat: tiap 3 frame + resolusi lebih kecil.",
            ) or DEFAULT_PERF_MODE

            threshold = st.slider(
                "Ambang keyakinan", MIN_CONFIDENCE, MAX_CONFIDENCE,
                DEFAULT_THRESHOLD, 0.01, key="opt_threshold",
                help="Semakin tinggi nilainya, semakin ketat sistem menerima gesture.",
            )
            margin = st.slider(
                "Ambang selisih (margin)", MIN_MARGIN, MAX_MARGIN,
                DEFAULT_MARGIN, 0.01, key="opt_margin",
                help="Perbedaan minimal antara prediksi pertama dan kedua.",
            )
            smoothing = st.slider(
                "Jendela smoothing", 1, 30, DEFAULT_SMOOTHING,
                key="opt_smoothing",
                help="Jumlah frame terakhir yang dirata-ratakan agar prediksi stabil.",
            )
            stable_frames = st.slider(
                "Minimum frame stabil", 1, 30, DEFAULT_STABLE_FRAMES,
                key="opt_stable_frames",
                help="Berapa lama gesture harus ditahan sebelum dianggap stabil.",
            )
            show_chart = st.toggle(
                "Tampilkan grafik probabilitas", value=False,
                key="opt_show_chart",
                help="Menampilkan top-5 kelas pada Detail prediksi.",
            )

        st.divider()
        st.caption("Model statis default: **V2** · Mirror: **ON**")

    perf = PERF_MODES.get(perf_mode, PERF_MODES[DEFAULT_PERF_MODE])
    st.session_state.runtime_settings.update(
        model_id=STATIC_MODEL_MAP.get(model_label, "v2"),
        mirror=mirror, threshold=threshold, margin=margin,
        smoothing_window=smoothing, stable_frames=stable_frames,
        show_chart=show_chart,
        perf_mode=perf_mode,
        inference_interval=perf["interval"],
        inference_size=perf["inference_size"],
    )


def is_fresh(payload: dict | None, ttl: float = RESULT_TTL_S) -> bool:
    """True bila payload inference masih baru (belum stale)."""
    return bool(payload) and (time.perf_counter() - payload.get("t", 0.0)) <= ttl


def active_prediction(payload: dict | None) -> dict | None:
    """Payload yang dianggap 'prediksi aktif'.

    None bila stale, tidak ada tangan, atau belum ada top-1. Dipakai agar
    Detail Prediksi tidak menampilkan probabilitas lama seolah masih berlaku.
    """
    if not is_fresh(payload):
        return None
    if not payload.get("hand_found"):
        return None
    if payload.get("top1_label") is None:
        return None
    return payload


def derive_state(payload: dict | None, settings: dict) -> dict:
    """Terjemahkan payload teknis menjadi state UX yang ramah pengguna."""
    fresh = is_fresh(payload)

    if not fresh:
        return {"kind": "idle", "letter": "—", "muted": True, "confidence": None,
                "status": "Aktifkan kamera untuk mulai",
                "hint": "Izinkan akses kamera pada browser, lalu tunggu beberapa saat.",
                "can_add": False, "technical": None}

    if payload.get("error"):
        return {"kind": "error", "letter": "—", "muted": True, "confidence": None,
                "status": "Terjadi gangguan saat memproses frame",
                "hint": "Coba lagi atau muat ulang halaman.",
                "can_add": False, "technical": payload.get("error")}

    if not payload["hand_found"]:
        return {"kind": "waiting_hand", "letter": "—", "muted": True,
                "confidence": None, "status": "Menunggu tangan...",
                "hint": "Posisikan tangan di dalam area kamera.",
                "can_add": False, "technical": None}

    if payload.get("rejected") or not payload.get("recognized"):
        if payload.get("recognized") is False and not payload.get("rejected"):
            return {"kind": "analyzing", "letter": "…", "muted": True,
                    "confidence": None, "status": "Menganalisis gesture...",
                    "hint": "Tahan posisi tangan sebentar.",
                    "can_add": False, "technical": None}
        return {"kind": "rejected", "letter": "?", "muted": True,
                "confidence": None,
                "status": "Gesture belum dikenali",
                "hint": "Coba sesuaikan posisi tangan, pencahayaan, atau jarak ke kamera.",
                "can_add": False, "technical": None}

    # recognized
    letter = payload.get("label") or "?"
    confidence = payload.get("confidence", 0.0)
    if payload.get("stable"):
        return {"kind": "recognized", "letter": letter, "muted": False,
                "confidence": confidence, "status": "Gesture dikenali",
                "hint": "Tekan “Tambahkan huruf” untuk menyusun kata.",
                "can_add": True, "technical": None}
    return {"kind": "hold", "letter": letter, "muted": False,
            "confidence": confidence, "status": "Tahan posisi tangan sebentar",
            "hint": "Pertahankan bentuk tangan sampai sistem stabil.",
            "can_add": False, "technical": None}


def render_hero(letter: str, confidence: float | None, muted: bool) -> None:
    color = "var(--text-color)" if muted else "var(--primary-color)"
    opacity = "0.35" if muted else "1"
    conf_html = ""
    if confidence is not None:
        conf_html = (
            f'<div style="font-size:1.4rem;color:var(--text-color);'
            f'opacity:.75;margin-top:2px;">{confidence * 100:.0f}%</div>'
        )
    st.html(
        f'<div style="text-align:center;padding:6px 0 2px 0;">'
        f'<div style="font-size:clamp(3.5rem,9vw,7rem);line-height:1.05;'
        f'font-weight:800;color:{color};opacity:{opacity};">{html.escape(letter)}</div>'
        f"{conf_html}</div>"
    )


def _status_line(state: dict) -> None:
    kind = state["kind"]
    if kind == "recognized":
        st.success(state["status"], icon=":material/check_circle:")
    elif kind == "rejected":
        st.warning(state["status"], icon=":material/error:")
    elif kind == "error":
        st.error(state["status"], icon=":material/error:")
    elif kind == "waiting_hand":
        st.info(state["status"], icon=":material/pan_tool:")
    elif kind == "idle":
        st.info(state["status"], icon=":material/videocam_off:")
    else:  # hold / analyzing
        st.info(state["status"], icon=":material/hourglass_top:")
    if state.get("hint"):
        st.caption(state["hint"])


def _render_prediction_body(reset_when_idle: bool = True) -> None:
    st.subheader("Prediksi")
    payload = st.session_state.shared.get()
    settings = st.session_state.runtime_settings.snapshot()
    state = derive_state(payload, settings)

    # Reset prediksi aktif saat kamera mati / data sudah stale.
    if reset_when_idle and state["kind"] == "idle":
        st.session_state.shared.clear()

    with st.container(border=True):
        render_hero(state["letter"], state["confidence"], state["muted"])
        _status_line(state)

    add_clicked = st.button(
        "+ Tambahkan huruf", type="primary", icon=":material/add:",
        width="stretch", disabled=not state["can_add"],
        help="Aktif setelah gesture dikenali dan stabil.",
    )
    if add_clicked:
        payload = st.session_state.shared.get()
        if payload and payload.get("label"):
            st.session_state.spelling = apply_spelling_action(
                st.session_state.spelling, "add", payload["label"]
            )
            st.session_state.last_added = payload["label"]
        st.rerun()  # full rerun agar hasil ejaan ikut terbarui

    if not state["can_add"]:
        st.caption("Tombol aktif saat gesture sudah dikenali dan stabil.")


@st.fragment(run_every=FRAGMENT_INTERVAL)
def prediction_fragment() -> None:
    """Auto-refresh HANYA dirender saat WebRTC playing (lihat main)."""
    _render_prediction_body(reset_when_idle=True)


def render_prediction_static() -> None:
    """Panel prediksi tanpa auto-refresh (dipakai saat WebRTC connecting)."""
    _render_prediction_body(reset_when_idle=False)


def render_spelling() -> None:
    st.subheader("Hasil ejaan")
    spelling = st.session_state.get("spelling", "")

    with st.container(border=True):
        if spelling:
            st.html(
                '<div style="font-size:clamp(1.5rem,3.2vw,2.1rem);'
                'font-weight:600;color:var(--text-color);'
                'word-break:break-word;line-height:1.4;">'
                f"{html.escape(spelling)}</div>"
            )
        else:
            st.caption("Belum ada huruf. Bentuk gesture, lalu tekan "
                       "**Tambahkan huruf**.")

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        with st.container(horizontal=True):
            if st.button("Spasi", icon=":material/space_bar:"):
                st.session_state.spelling = apply_spelling_action(
                    st.session_state.spelling, "space"
                )
                st.rerun()
            if st.button("Hapus", icon=":material/backspace:"):
                st.session_state.spelling = apply_spelling_action(
                    st.session_state.spelling, "delete"
                )
                st.rerun()
    with right:
        if st.button("Reset", icon=":material/restart_alt:",
                     help="Menghapus seluruh hasil ejaan."):
            confirm_reset_dialog()


@st.dialog("Reset hasil ejaan?")
def confirm_reset_dialog() -> None:
    st.write("Seluruh huruf yang sudah disusun akan dihapus.")
    col_yes, col_no = st.columns(2)
    if col_yes.button("Ya, reset", type="primary", width="stretch"):
        st.session_state.spelling = ""
        st.rerun()
    if col_no.button("Batal", width="stretch"):
        st.rerun()


_STATUS_TECH = {
    "recognized": "Diterima (stabil)",
    "hold": "Diterima (belum stabil)",
    "rejected": "Ditolak",
    "analyzing": "Menganalisis",
    "waiting_hand": "Tidak ada tangan",
    "idle": "Kamera belum aktif",
    "error": "Error",
}


def _tech_table(payload: dict, settings: dict, state: dict) -> None:
    rows = [
        ("Status", _STATUS_TECH.get(state["kind"], "-")),
        ("top-1", f"{payload.get('top1_label') or '-'} "
                  f"({payload.get('top1_prob', 0.0):.3f})"),
        ("top-2", f"{payload.get('top2_label') or '-'} "
                  f"({payload.get('top2_prob', 0.0):.3f})"),
        ("Margin", f"{payload.get('margin', 0.0):.3f}"),
        ("Camera FPS", f"{payload.get('camera_fps', 0.0):.1f}"),
        ("Inference FPS", f"{payload.get('inference_fps', 0.0):.1f}"),
        ("Latensi frame", f"{payload.get('frame_latency_ms', 0.0):.1f} ms"),
        ("Latensi inference", f"{payload.get('inference_latency_ms', 0.0):.1f} ms"),
        ("MediaPipe", f"{payload.get('mediapipe_ms', 0.0):.1f} ms"),
        ("Prediksi (scaler+MLP)", f"{payload.get('predict_ms', 0.0):.1f} ms"),
        ("Drawing", f"{payload.get('draw_ms', 0.0):.1f} ms"),
        ("Frame di-skip (throttle)", f"{payload.get('skipped_frames', 0)}"),
        ("Static model", STATIC_ID_TO_LABEL.get(settings["model_id"],
                                                  settings["model_id"])),
        ("Mode performa", settings.get("perf_mode", DEFAULT_PERF_MODE)),
        ("Mirror", "ON" if settings["mirror"] else "OFF"),
        ("Ambang keyakinan", f"{settings['threshold']:.2f}"),
        ("Ambang margin", f"{settings['margin']:.2f}"),
        ("Stable frame", f"{settings['stable_frames']}"),
        ("Error terakhir", payload.get("error") or "-"),
        ("Processor dibuat (count)", payload.get("processor_created_count")),
        ("First frame (count)", payload.get("first_frame_count")),
        ("Initialized", payload.get("initialized")),
        ("WebRTC playing", st.session_state.get("stream_playing")),
        ("Processor created t", f"{payload.get('processor_created_ts', 0.0):.3f}"),
        ("First frame t", f"{payload.get('first_frame_ts', 0.0):.3f}"),
    ]
    st.markdown(
        "| Item | Nilai |\n|:--|:--|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in rows)
    )


def render_details() -> None:
    exp = st.expander("Detail prediksi", icon=":material/monitoring:",
                      on_change="rerun")
    if exp.open:
        payload = st.session_state.shared.get()
        settings = st.session_state.runtime_settings.snapshot()

        # Tampilkan exception sebenarnya (jangan disembunyikan sebagai masalah kamera)
        if payload and payload.get("error"):
            st.error(f"Error inference: {payload['error']}", icon=":material/error:")
            if payload.get("error_detail"):
                st.code(payload["error_detail"], language="text")

        active = active_prediction(payload)

        if active is None:
            st.caption("Tidak ada prediksi aktif.")
            return

        state = derive_state(payload, settings)
        _tech_table(active, settings, state)

        if settings.get("show_chart") and active.get("top5"):
            st.caption("Grafik memakai snapshot saat panel dibuka. "
                       "Kamera tetap menjadi prioritas performa.")
            top5 = active["top5"]
            chart_df = pd.DataFrame(
                {"Kelas": [c for c, _ in top5],
                 "Probabilitas": [p for _, p in top5]}
            )
            st.bar_chart(chart_df, x="Kelas", y="Probabilitas", height=220)


def render_reference() -> None:
    with st.expander("Cara menggunakan", icon=":material/help:"):
        st.markdown(
            "1. Izinkan akses kamera.\n"
            "2. Arahkan satu tangan ke kamera.\n"
            "3. Bentuk salah satu abjad SIBI.\n"
            "4. Tahan posisi sampai gesture dikenali.\n"
            "5. Tekan **Tambahkan huruf** untuk menyusun kata."
        )

    with st.expander("Huruf yang didukung", icon=":material/abc:"):
        st.markdown(
            " ".join(f"**{c}**" for c in SUPPORTED_LETTERS)
        )
        st.markdown(
            f"- **J** — belum didukung\n"
            f"- **Z** — belum didukung"
        )
        st.caption(
            "J dan Z menggunakan gesture dinamis, sedangkan versi model saat ini "
            "dirancang untuk gesture statis. Sistem ini belum mendukung penuh A–Z."
        )

    with st.expander("Tentang sistem", icon=":material/info:"):
        st.markdown("**Model statis**")
        st.markdown(
            "- Jenis: `MLPClassifier` (scikit-learn)\n"
            "- Input: 63 fitur landmark tangan\n"
            "- Landmark: 21 titik × (x, y, z)\n"
            "- Jumlah kelas: 24 gesture SIBI statis\n"
            "- Pilihan: **V2** (default) atau **Current** (baseline)"
        )
        st.markdown("**Akurasi uji offline**")
        st.markdown(
            f"- V2 — original test: {OFFLINE_ACC['v2']['original']}, "
            f"macro F1: {OFFLINE_ACC['v2']['macro_f1']}\n"
            f"- Current — original test: {OFFLINE_ACC['current']['original']}, "
            f"macro F1: {OFFLINE_ACC['current']['macro_f1']}"
        )
        st.caption(
            "Performa webcam nyata dapat berbeda dari hasil pengujian dataset. "
            "V2 tidak diklaim lebih baik untuk semua huruf."
        )
        st.caption(
            "Hasil pengujian webcam nyata dapat berbeda dari hasil pengujian "
            "dataset. Angka di atas bukan jaminan akurasi untuk semua pengguna."
        )
        st.divider()
        st.caption(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def render_header() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)


def render_camera(bundles: dict, settings: RuntimeSettings,
                  shared: LatestResult) -> None:
    st.caption(CAMERA_GUIDE_TEXT)
    try:
        ctx = webrtc_streamer(
            key="sibi_camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: SibiVideoProcessor(
                bundles, settings, shared
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": CAPTURE_WIDTH},
                    "height": {"ideal": CAPTURE_HEIGHT},
                    "frameRate": {"ideal": CAPTURE_FPS_IDEAL, "max": 30},
                },
                "audio": False,
            },
            rtc_configuration=None,
            # async_processing: SIBI_ASYNC_PROCESSING=on/off (A/B test).
            async_processing=ASYNC_PROCESSING,
        )
        playing = bool(ctx.state.playing)
        prev = st.session_state.get("_webrtc_playing_prev")
        if playing and prev is not True:
            plog(f"WEBRTC PLAYING t={time.perf_counter():.3f} "
                 f"processor_created={_PROC_CREATED_COUNT}")
        elif (not playing) and prev is True:
            plog("WEBRTC STOPPED")
        st.session_state["_webrtc_playing_prev"] = playing
        # Catat status saja; JANGAN st.rerun() dari perubahan state kamera.
        st.session_state.stream_playing = playing
        st.caption(
            "Gunakan tombol **START**/**STOP** pada kamera untuk menyalakan "
            "atau menghentikan, dan **SELECT DEVICE** untuk memilih kamera. "
            "Izinkan akses kamera jika browser memintanya."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Kamera tidak dapat diakses.\n\n"
            "Pastikan browser telah diberikan izin kamera.",
            icon=":material/videocam_off:",
        )
        st.caption(f"Detail teknis: {type(exc).__name__}")


def main() -> None:
    st.set_page_config(
        page_title="SIBI Real-Time",
        page_icon="🤟",
        layout="wide",
    )
    init_state()
    render_sidebar()
    render_header()

    missing = missing_runtime_assets()
    if missing:
        st.error(
            "Model aplikasi tidak dapat dimuat. Periksa konfigurasi aplikasi.",
            icon=":material/error:",
        )
        st.caption("Aset runtime tidak ditemukan: " + ", ".join(missing))
        return

    try:
        bundles = get_bundles()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Model aplikasi tidak dapat dimuat. Periksa konfigurasi aplikasi.",
            icon=":material/error:",
        )
        st.caption(f"Detail teknis (developer): {type(exc).__name__}: {exc}")
        return

    settings = st.session_state.runtime_settings
    shared = st.session_state.shared

    col_cam, col_pred = st.columns([1.5, 1], gap="large")
    with col_cam:
        render_camera(bundles, settings, shared)
    with col_pred:
        # TEST E: auto-refresh HANYA saat WebRTC benar-benar playing. Saat
        # connecting, jangan ada rerun berkala (hindari race establishment).
        if st.session_state.get("stream_playing"):
            prediction_fragment()
        else:
            render_prediction_static()

    st.divider()
    render_spelling()

    st.divider()
    render_details()
    render_reference()


if __name__ == "__main__":
    main()
