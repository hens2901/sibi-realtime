"""Hybrid router STATIC (MLP 24 huruf) vs DYNAMIC (GRU J/Z) + dynamic rejection.

Tidak mengintegrasikan ke aplikasi. Dipakai oleh:
- scripts/test_hybrid_router.py (offline)
- hybrid_realtime_test.py (webcam)

Prinsip:
- motion features dihitung dari landmark RAW (sama dengan training J/Z);
- sequence 24x63 memakai preprocessing yang sama (wrist-relative + scale);
- routing berbasis state machine dengan hysteresis;
- dynamic rejection: confidence + margin + rentang movement + jarak prototipe
  + variasi temporal + cukup perubahan.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import realtime as rt  # noqa: E402

SEQ_LEN = 24
FEATURE_COUNT = 63
FINGERTIPS = (4, 8, 12, 16, 20)
THRESHOLDS_JSON = ROOT / "reports" / "dynamic_router_thresholds.json"
PROTO_NPZ = ROOT / "models" / "dynamic_jz_prototypes.npz"

# Kontrol state machine
INSTANT_WINDOW_S = 0.30      # jendela untuk kecepatan sesaat (onset/settle)
ONSET_CONFIRM = 3            # frame berturut di atas threshold -> mulai rekam
SETTLE_CONFIRM = 5           # frame berturut di bawah threshold -> selesai
MAX_RECORD_S = 2.5           # batas durasi perekaman
COOLDOWN_S = 1.2             # jeda setelah J/Z diterima
MISS_GRACE = 3               # frame tanpa tangan sebelum reset
MIN_RECORD_FRAMES = 5        # minimal frame untuk membentuk sequence
PRE_ROLL_S = 0.15            # sertakan frame sebelum onset (posisi awal)
POST_ROLL_S = 0.08           # sertakan frame akhir sebelum classify
MOVE_STEP_EPS = 1e-3

# Profile threshold eksperimental. "current" memakai dynamic_router_thresholds.json
# (production). "c2" = C2 balanced dari reports/LIVE_DYNAMIC_CALIBRATION.md.
# "c2_zadaptive" = C2 + capture Z-adaptive (tiered onset, pre/post-roll lebih
# panjang, trajectory completeness). Default tetap "current".
_C2_GATES = {
    "distance_max_cls": {"J": 9.83, "Z": 7.50},
    "movement_range": {"J": (0.20, 1.60), "Z": (0.05, 1.40)},
    "variation_min_cls": {"J": 0.02, "Z": 0.02},
    "dynamic_conf_min": 0.95,
    "dynamic_margin_min": 0.20,
}
PROFILES = {
    "current": {},
    "c2": dict(_C2_GATES),
    "c2_zadaptive": {
        # Gate J = C2 (sesuai spesifikasi bagian 1).
        "distance_max_cls": {"J": 9.83, "Z": 9.50},
        "movement_range": {"J": (0.20, 1.60), "Z": (0.05, 1.50)},
        "variation_min_cls": {"J": 0.02, "Z": 0.02},
        "dynamic_conf_min": 0.95,
        "dynamic_margin_min": 0.20,
        "tiered_onset": True,
        "pre_roll_s": 0.25,
        "post_roll_s": 0.25,
        "trajectory": {"min_coverage": 0.35, "min_path": 0.02,
                       "min_direction_changes": 0},
    },
}
DEFAULT_PROFILE = "current"
VALID_PROFILES = tuple(PROFILES.keys())


@dataclass
class DynamicEvent:
    """Detail lengkap satu dynamic event (untuk debug & diagnostik)."""
    t_onset_first: float = 0.0
    t_onset_confirmed: float = 0.0
    t_settle_first: float = 0.0
    t_classify: float = 0.0
    rec_start: float = 0.0
    rec_end: float = 0.0
    durations_ms: dict = field(default_factory=dict)
    predicted: str | None = None
    confidence: float = 0.0
    margin: float = 0.0
    movement_magnitude: float = 0.0
    wrist_movement: float = 0.0
    fingertip_movement: float = 0.0
    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    path_length: float = 0.0
    temporal_consistency: float = 0.0
    distance: float | None = None
    variation: float = 0.0
    moving_steps: int = 0
    dJ: float | None = None
    dZ: float | None = None
    motion_type: str = "STATIC"
    coverage: float = 0.0
    direction_changes: int = 0
    net_progress: float = 0.0
    checks: dict = field(default_factory=dict)   # name -> {pass, value, threshold}
    accepted: bool = False
    reason: str = ""
    n_frames: int = 0
    fps_est: float = 0.0
    sequence_raw: np.ndarray | None = None       # (T,63) normalized
    sequence: np.ndarray | None = None           # (24,63) resampled


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class MotionStats:
    movement_magnitude: float = 0.0
    wrist_movement: float = 0.0
    fingertip_movement: float = 0.0
    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    path_length: float = 0.0
    temporal_consistency: float = 0.0
    duration_sec: float = 0.0
    n_frames: int = 0
    moving_steps: int = 0


@dataclass
class RouterOutput:
    state: str = "IDLE"
    hand_found: bool = False
    instant_speed: float = 0.0
    route: str = "-"
    motion_type: str = "STATIC"
    motion: MotionStats | None = None
    static_label: str | None = None
    static_conf: float = 0.0
    static_recognized: bool = False
    static_stable: bool = False
    dynamic_label: str | None = None
    dynamic_conf: float = 0.0
    dynamic_margin: float = 0.0
    dynamic_distance: float = 0.0
    dynamic_accepted: bool = False
    dynamic_reason: str = ""
    dynamic_latency_ms: float = 0.0
    dynamic_event: "DynamicEvent | None" = None
    event: str | None = None          # mis. "J" bila diterima
    final_output: str = "-"


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def as_frames(raw: np.ndarray | None):
    return None if raw is None else np.asarray(raw, dtype=np.float64)


def resample_raw(times, raw, n=SEQ_LEN) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    target = np.linspace(times[0], times[-1], n)
    flat = raw.reshape(len(times), -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(target, times, flat[:, j])
    return out.reshape((n,) + raw.shape[1:])


def resample_features(times, feats, n=SEQ_LEN) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float64)
    target = np.linspace(times[0], times[-1], n)
    out = np.empty((n, feats.shape[1]), dtype=np.float64)
    for j in range(feats.shape[1]):
        out[:, j] = np.interp(target, times, feats[:, j])
    return out


def motion_from_frames(times, raw) -> MotionStats:
    if raw is None or len(times) < 2:
        return MotionStats(n_frames=0)
    raw_seq = resample_raw(times, raw, SEQ_LEN)
    d_wrist = np.linalg.norm(np.diff(raw_seq[:, 0, :], axis=0), axis=1)
    d_tips = np.linalg.norm(np.diff(raw_seq[:, FINGERTIPS, :], axis=0), axis=2)
    d_all = np.linalg.norm(np.diff(raw_seq, axis=0), axis=2)     # (23,21)
    step = np.mean(d_all, axis=1)                                # (23,)
    duration = float(times[-1] - times[0])
    movement = float(step.sum())
    wrist = float(d_wrist.sum())
    tip = float(d_tips.sum())
    path = wrist
    net = float(np.linalg.norm(raw_seq[-1, 0, :] - raw_seq[0, 0, :]))
    consistency = float(net / path) if path > 1e-9 else 0.0
    dt = duration / max(1, len(step))
    max_vel = float(step.max() / dt) if dt > 0 else 0.0
    return MotionStats(
        movement_magnitude=movement,
        wrist_movement=wrist,
        fingertip_movement=tip,
        mean_velocity=movement / duration if duration > 0 else 0.0,
        max_velocity=max_vel,
        path_length=path,
        temporal_consistency=consistency,
        duration_sec=duration,
        n_frames=len(times),
        moving_steps=int((step > MOVE_STEP_EPS).sum()),
    )


# --------------------------------------------------------------------------- #
# Static pipeline (reuse realtime primitives)
# --------------------------------------------------------------------------- #

class StaticPipeline:
    def __init__(self):
        model, scaler, encoder, labels = rt.load_artifacts("augmented")
        self.model, self.scaler, self.encoder = model, scaler, encoder
        self.labels = list(labels)
        self.smoother = rt.TemporalSmoother(self.labels)

    def reset(self):
        self.smoother.reset()

    def predict(self, features: np.ndarray):
        probs = rt.predict_proba(self.model, self.scaler, features)
        top2 = rt.top2_from_probs(probs, self.labels)
        accepted = rt.passes_rejection(
            top2.top1_prob, top2.margin,
            rt.CONFIDENCE_THRESHOLD, rt.MARGIN_THRESHOLD)
        if accepted:
            label, conf, stable = self.smoother.update(probs)
        else:
            self.smoother.update(None)
            label, conf, stable = None, 0.0, False
        return {
            "label": label, "conf": float(conf), "recognized": bool(accepted),
            "stable": bool(stable), "top1": top2.top1_label,
            "top1_prob": top2.top1_prob, "top2": top2.top2_label,
            "top2_prob": top2.top2_prob, "margin": top2.margin,
        }


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class TemporalBuffer:
    def __init__(self, max_seconds: float = 3.0):
        self.max_seconds = max_seconds
        self.frames: deque = deque()

    def add(self, t, features, raw):
        self.frames.append((float(t), np.asarray(features, dtype=np.float64),
                            np.asarray(raw, dtype=np.float64)))
        while self.frames and (t - self.frames[0][0]) > self.max_seconds:
            self.frames.popleft()

    def clear(self):
        self.frames.clear()

    def window(self, t0=None, t1=None):
        out = []
        for t, f, r in self.frames:
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            out.append((t, f, r))
        return out

    def last_span(self, t, seconds):
        return self.window(t0=t - seconds)


class HybridRouter:
    IDLE, STATIC, MOTION_START, RECORDING, COOLDOWN = (
        "IDLE", "STATIC", "MOTION_START", "DYNAMIC_RECORDING", "COOLDOWN")

    def __init__(self, static: StaticPipeline | None = None,
                 dynamic=None, config: dict | None = None,
                 profile: str = DEFAULT_PROFILE):
        self.profile_name = profile if profile in PROFILES else DEFAULT_PROFILE
        cfg = config or {}
        self.thr = self._load_thresholds()

        # Override profile (eksperimental). Kunci capture di-pop dulu, sisanya
        # menjadi override threshold.
        over = dict(PROFILES.get(self.profile_name, {}))
        self.tiered_onset = bool(over.pop("tiered_onset", False))
        self.pre_roll_s = float(over.pop("pre_roll_s", PRE_ROLL_S))
        self.post_roll_s = float(over.pop("post_roll_s", POST_ROLL_S))
        self.trajectory_params = over.pop("trajectory", None)
        self._onset_subtle = over.pop("onset_subtle", None)
        self._onset_strong = over.pop("onset_strong", None)
        self._settle_override = over.pop("settle_speed", None)
        self.thr.update(over)
        if cfg:
            self.thr.update(cfg)
        self.static = static or StaticPipeline()
        if dynamic is None:
            from dynamic_jz_inference import DynamicJZPredictor
            dynamic = DynamicJZPredictor()
        self.dynamic = dynamic
        self.proto = self._load_prototypes()
        self.buffer = TemporalBuffer()
        self.reset()

    @staticmethod
    def _load_thresholds() -> dict:
        if THRESHOLDS_JSON.exists():
            d = json.loads(THRESHOLDS_JSON.read_text(encoding="utf-8"))
            chosen = d.get("chosen", {})
            # Onset/settle diturunkan dari distribusi velocity Z (kelas dengan
            # gerakan lebih halus) bila tidak tersedia eksplisit.
            zvel = d.get("movement", {}).get("raw_velocity", {}).get("Z", {})
            jvel = d.get("movement", {}).get("raw_velocity", {}).get("J", {})
            onset = chosen.get("onset_speed")
            if onset is None and zvel.get("p10"):
                onset = float(zvel["p10"]) * 0.8
            settle = chosen.get("settle_speed")
            if settle is None and zvel.get("p05"):
                settle = float(zvel["p05"]) * 0.5

            def pget(section: dict, key: str, default: float) -> float:
                try:
                    return float(section.get(key, default))
                except (TypeError, ValueError):
                    return default

            move = d.get("movement", {}).get("raw_movement_magnitude", {})
            dist = d.get("distance", {})
            var = d.get("temporal_variation", {})
            movement_range = {}
            for cl in ("J", "Z"):
                mv = move.get(cl, {})
                lo = mv.get("p10", None)
                lo = float(lo) if lo is not None else pget(mv, "p0", 0.05) * 0.8
                movement_range[cl] = (lo, pget(mv, "p100", 2.0))
            distance_max_cls = {
                cl: pget(dist.get(cl, {}), "p95", 12.0) * 1.1 for cl in ("J", "Z")
            }
            variation_min_cls = {
                cl: pget(var.get(cl, {}), "p0", 0.03) * 0.5 for cl in ("J", "Z")
            }
            train_dist_p100 = {
                cl: pget(dist.get(cl, {}), "p100", 12.0) for cl in ("J", "Z")
            }
            return {
                "static_motion_max": chosen.get("static_motion_max", 0.05),
                "dynamic_motion_min": chosen.get("dynamic_motion_min", 0.10),
                "dynamic_motion_max": chosen.get("dynamic_motion_max", 2.0),
                "dynamic_conf_min": chosen.get("dynamic_conf_min", 0.90),
                "dynamic_margin_min": chosen.get("dynamic_margin_min", 0.20),
                "distance_max": chosen.get("distance_max", 12.0),
                "variation_min": chosen.get("variation_min", 0.05),
                "onset_speed": onset,
                "settle_speed": settle,
                "onset_subtle_default": pget(zvel, "p05", 0.12) * 0.8,
                "onset_strong_default": pget(jvel, "p05", 0.30) * 0.8,
                "movement_range": movement_range,
                "distance_max_cls": distance_max_cls,
                "variation_min_cls": variation_min_cls,
                "train_dist_p100": train_dist_p100,
            }
        return {"static_motion_max": 0.05, "dynamic_motion_min": 0.10,
                "dynamic_motion_max": 2.0, "dynamic_conf_min": 0.90,
                "dynamic_margin_min": 0.20, "distance_max": 12.0,
                "variation_min": 0.05, "onset_speed": None, "settle_speed": None,
                "onset_subtle_default": 0.10, "onset_strong_default": 0.25,
                "train_dist_p100": {"J": 12.0, "Z": 12.0}}

    @staticmethod
    def _load_prototypes():
        if not PROTO_NPZ.exists():
            return None
        d = np.load(PROTO_NPZ, allow_pickle=True)
        return {"J": d["J"].astype(np.float64), "Z": d["Z"].astype(np.float64)}

    # ---------------------------------------------------------------- state
    def reset(self):
        self.state = self.IDLE
        self.buffer.clear()
        self.static.reset()
        self.onset_count = 0
        self.settle_count = 0
        self.miss_count = 0
        self.onset_t = None
        self.settle_t = None
        self._onset_first = None
        self._settle_first = None
        self._onset_confirmed_t = None
        self._motion_type = "STATIC"
        self.cooldown_until = 0.0
        self.last_event = None
        self.last_output = RouterOutput()

    def _onset_speed(self) -> float:
        # Kecepatan minimum untuk memulai dynamic recording.
        if self.tiered_onset:
            v = self._onset_subtle
            if v is None:
                v = self.thr.get("onset_subtle_default", 0.10)
            return float(v)
        v = self.thr.get("onset_speed")
        if v:
            return float(v)
        return float(self.thr["dynamic_motion_min"]) * 2.0

    def _onset_strong_speed(self) -> float:
        v = self._onset_strong
        if v is None:
            v = self.thr.get("onset_strong_default", 0.25)
        return float(v)

    def motion_tier(self, inst_speed: float) -> str:
        """STATIC / SUBTLE / STRONG. Tanpa tiered onset -> STRONG/STATIC."""
        if inst_speed >= self._onset_strong_speed():
            return "STRONG"
        if inst_speed >= self._onset_speed():
            return "SUBTLE" if self.tiered_onset else "STRONG"
        return "STATIC"

    def _settle_speed(self) -> float:
        if self._settle_override is not None:
            return float(self._settle_override)
        v = self.thr.get("settle_speed")
        if v:
            return float(v)
        return float(self.thr["static_motion_max"])

    # ---------------------------------------------------------------- update
    def update(self, t: float, landmarks) -> RouterOutput:
        out = RouterOutput(state=self.state)
        hand = landmarks is not None

        if hand:
            self.miss_count = 0
            feats = rt.normalize_landmarks(landmarks)
            raw = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
            self.buffer.add(t, feats, raw)
        else:
            self.miss_count += 1
            if self.miss_count > MISS_GRACE:
                self.buffer.clear()
                if self.state in (self.MOTION_START, self.RECORDING):
                    self.state = self.IDLE
                elif self.state != self.COOLDOWN:
                    self.state = self.IDLE
            out.state = self.state
            out.hand_found = False
            out.final_output = "Tidak ada tangan"
            self.last_output = out
            return out

        # kecepatan sesaat dari jendela pendek
        span = self.buffer.last_span(t, INSTANT_WINDOW_S)
        inst = motion_from_frames([f[0] for f in span], np.stack([f[2] for f in span])) \
            if len(span) >= 2 else MotionStats()
        inst_speed = inst.mean_velocity
        out.instant_speed = inst_speed
        out.motion_type = self.motion_tier(inst_speed)
        self._motion_type = out.motion_type
        out.hand_found = True

        # static prediction (route default)
        static = self.static.predict(feats)
        out.static_label = static["label"]
        out.static_conf = static["conf"]
        out.static_recognized = static["recognized"]
        out.static_stable = static["stable"]

        # ---------------- state machine ----------------
        if self.state == self.COOLDOWN:
            if t >= self.cooldown_until:
                self.state = self.STATIC
            out.route = "STATIC"
            out.state = self.state
            out.final_output = self.last_event or "-"
            self.last_output = out
            return out

        if self.state in (self.IDLE, self.STATIC):
            if inst_speed >= self._onset_speed():
                if self.onset_count == 0:
                    self._onset_first = t
                self.onset_count += 1
            else:
                self.onset_count = 0
            if self.onset_count >= ONSET_CONFIRM:
                self.state = self.MOTION_START
                self.onset_t = self._onset_first if self._onset_first is not None else t
                self._onset_confirmed_t = t
                self.settle_count = 0
                self._settle_first = None
            else:
                self.state = self.STATIC
                out.route = "STATIC"
                out.state = self.state
                out.final_output = (f"{static['label']}"
                                    if static["recognized"] else "Tidak dikenali (statis)")
                self.last_output = out
                return out

        if self.state == self.MOTION_START:
            self.state = self.RECORDING
            self.settle_count = 0
            self._settle_first = None

        if self.state == self.RECORDING:
            # deteksi berhenti bergerak
            if inst_speed <= self._settle_speed():
                if self.settle_count == 0:
                    self._settle_first = t
                self.settle_count += 1
            else:
                self.settle_count = 0
                self._settle_first = None
            too_long = (self.onset_t is not None) and (t - self.onset_t) > MAX_RECORD_S
            if self.settle_count >= SETTLE_CONFIRM or too_long:
                self.settle_t = self._settle_first if self._settle_first is not None else t
                self._classify(out, t)
                self.state = self.COOLDOWN
                self.cooldown_until = t + COOLDOWN_S
                self.onset_count = 0
            else:
                out.route = "DYNAMIC"
                out.state = self.state
                out.final_output = "Merekam gerakan..."
            self.last_output = out
            return out

        self.last_output = out
        return out

    # ------------------------------------------------------------ classify
    def _classify(self, out: RouterOutput, t: float):
        ev = DynamicEvent()
        ev.motion_type = getattr(self, "_motion_type", "STATIC")
        ev.t_onset_first = self._onset_first or t
        ev.t_onset_confirmed = self._onset_confirmed_t or t
        ev.t_settle_first = self._settle_first or t
        ev.t_classify = t
        ev.rec_start = ev.t_onset_first - self.pre_roll_s
        ev.rec_end = min(t, ev.t_settle_first + self.post_roll_s)
        ev.durations_ms = {
            "onset_confirm": (ev.t_onset_confirmed - ev.t_onset_first) * 1000.0,
            "recording": (ev.t_settle_first - ev.t_onset_confirmed) * 1000.0,
            "pre_roll": self.pre_roll_s * 1000.0,
            "post_roll": self.post_roll_s * 1000.0,
            "motion_to_classify": (t - ev.t_onset_first) * 1000.0,
            "classify": 0.0,
            "capture_total": (ev.rec_end - ev.rec_start) * 1000.0,
        }

        frames = self.buffer.window(t0=ev.rec_start, t1=ev.rec_end)
        out.route = "DYNAMIC"
        ev.n_frames = len(frames)
        if len(frames) >= 2:
            ev.fps_est = (len(frames) - 1) / max(1e-6, frames[-1][0] - frames[0][0])

        if len(frames) < MIN_RECORD_FRAMES:
            out.dynamic_accepted = False
            out.dynamic_reason = "sequence terlalu pendek"
            out.final_output = "Tidak dikenali"
            ev.reason = out.dynamic_reason
            out.dynamic_event = ev
            out.dynamic_latency_ms = (t - ev.t_onset_first) * 1000.0
            return

        times = [f[0] for f in frames]
        raw = np.stack([f[2] for f in frames])
        feats = np.stack([f[1] for f in frames])
        motion = motion_from_frames(times, raw)
        out.motion = motion
        ev.movement_magnitude = motion.movement_magnitude
        ev.wrist_movement = motion.wrist_movement
        ev.fingertip_movement = motion.fingertip_movement
        ev.mean_velocity = motion.mean_velocity
        ev.max_velocity = motion.max_velocity
        ev.path_length = motion.path_length
        ev.temporal_consistency = motion.temporal_consistency
        ev.moving_steps = motion.moving_steps
        out.dynamic_latency_ms = (t - ev.t_onset_first) * 1000.0

        seq = resample_features(times, feats, SEQ_LEN).astype(np.float32)
        ev.sequence_raw = feats.astype(np.float32)
        ev.sequence = seq
        if not np.all(np.isfinite(seq)):
            out.dynamic_accepted = False
            out.dynamic_reason = "sequence NaN/inf"
            out.final_output = "Tidak dikenali"
            ev.reason = out.dynamic_reason
            out.dynamic_event = ev
            return

        proba = self.dynamic.predict(seq)[0]  # (2,)
        order = np.argsort(proba)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        conf = float(proba[top1])
        margin = float(proba[top1] - proba[top2])
        label = self.dynamic.classes[top1]
        out.dynamic_label = label
        out.dynamic_conf = conf
        out.dynamic_margin = margin
        ev.predicted = label
        ev.confidence = conf
        ev.margin = margin

        # jarak ke prototipe (scaled space)
        dist = None
        if self.proto is not None:
            scaled = self.dynamic.scaler.transform(
                seq.reshape(-1, FEATURE_COUNT)).reshape(SEQ_LEN, FEATURE_COUNT)
            d = {cl: float(np.mean(np.linalg.norm(
                scaled - self.proto[cl], axis=1))) for cl in ("J", "Z")}
            dist = min(d.values())
            out.dynamic_distance = dist
            ev.dJ = d["J"]
            ev.dZ = d["Z"]
        ev.distance = dist

        # variasi temporal
        dsteps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        variation = float(np.std(dsteps))
        ev.variation = variation

        # trajectory completeness (lintasan wrist pada raw resampled)
        raw_seq = resample_raw(times, raw, SEQ_LEN)
        wrist_path = raw_seq[:, 0, :]
        wsteps = np.diff(wrist_path, axis=0)
        step_mag = np.linalg.norm(wsteps, axis=1)
        coverage = float(np.mean(step_mag > 0.002)) if len(step_mag) else 0.0
        cross = wsteps[:-1, 0] * wsteps[1:, 1] - wsteps[:-1, 1] * wsteps[1:, 0]
        signs = np.sign(cross)
        dir_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        path_len = float(step_mag.sum())
        net = float(np.linalg.norm(wrist_path[-1] - wrist_path[0]))
        net_progress = net / path_len if path_len > 1e-9 else 0.0
        ev.coverage = coverage
        ev.direction_changes = dir_changes
        ev.net_progress = net_progress

        mr = self.thr.get("movement_range", {}).get(label)
        if mr is None:
            mr = (self.thr["dynamic_motion_min"], self.thr["dynamic_motion_max"])
        dmax = self.thr.get("distance_max_cls", {}).get(label)
        vmin = self.thr.get("variation_min_cls", {}).get(label, self.thr["variation_min"])

        checks = {
            "confidence": {
                "pass": conf >= self.thr["dynamic_conf_min"],
                "value": conf, "threshold": self.thr["dynamic_conf_min"]},
            "margin": {
                "pass": margin >= self.thr["dynamic_margin_min"],
                "value": margin, "threshold": self.thr["dynamic_margin_min"]},
            "movement": {
                "pass": mr[0] <= motion.movement_magnitude <= mr[1] * 1.2,
                "value": motion.movement_magnitude,
                "range": [mr[0], mr[1] * 1.2]},
            "distance": {
                "pass": (dist is None) or (dmax is None) or (dist <= dmax),
                "value": dist, "threshold": dmax},
            "variation": {
                "pass": variation >= vmin, "value": variation, "threshold": vmin},
            "moving_steps": {
                "pass": motion.moving_steps >= 3, "value": motion.moving_steps,
                "threshold": 3},
        }
        if self.trajectory_params:
            tp = self.trajectory_params
            traj_pass = (coverage >= tp.get("min_coverage", 0.0)
                         and path_len >= tp.get("min_path", 0.0)
                         and dir_changes >= tp.get("min_direction_changes", 0))
            checks["trajectory"] = {
                "pass": bool(traj_pass),
                "value": {"coverage": round(coverage, 3),
                          "path": round(path_len, 3), "dir": dir_changes,
                          "net_progress": round(net_progress, 3)},
                "threshold": tp,
            }
        ev.checks = checks

        reasons = []
        if not checks["confidence"]["pass"]:
            reasons.append(f"CONF {conf:.2f} < {checks['confidence']['threshold']:.2f}")
        if not checks["margin"]["pass"]:
            reasons.append(f"MARGIN {margin:.2f} < {checks['margin']['threshold']:.2f}")
        if not checks["movement"]["pass"]:
            reasons.append(f"MOTION {motion.movement_magnitude:.3f} luar "
                           f"[{mr[0]:.2f},{mr[1]*1.2:.2f}]")
        if not checks["distance"]["pass"]:
            reasons.append(f"DIST {dist:.2f} > {dmax:.2f}")
        if not checks["variation"]["pass"]:
            reasons.append(f"VARIATION {variation:.3f} < {vmin:.3f}")
        if not checks["moving_steps"]["pass"]:
            reasons.append("STEPS < 3")
        if "trajectory" in checks and not checks["trajectory"]["pass"]:
            v = checks["trajectory"]["value"]
            reasons.append(f"TRAJECTORY cov={v['coverage']} path={v['path']} "
                           f"dir={v['dir']}")

        if reasons:
            out.dynamic_accepted = False
            out.dynamic_reason = "; ".join(reasons)
            out.final_output = "Tidak dikenali"
            ev.accepted = False
            ev.reason = out.dynamic_reason
        else:
            out.dynamic_accepted = True
            out.dynamic_reason = "diterima"
            out.event = label
            self.last_event = label
            out.final_output = f"{label} (dinamis)"
            ev.accepted = True
            ev.reason = "ACCEPT"
        out.dynamic_event = ev

