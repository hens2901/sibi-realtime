"""Test harness offline untuk hybrid router STATIC vs DYNAMIC + rejection.

Skenario:
- sequence J (asli, disimulasikan sebagai frame)
- sequence Z (asli)
- static sequence (fitur statis dari dataset, jitter kecil)
- small movement
- random trajectory
- incomplete J, incomplete Z
- transition-like (J lalu Z)

Synthetic negative dipakai HANYA untuk unit test router, bukan bukti akurasi
penelitian.

Output: reports/hybrid_router_test.txt

Jalankan:
    python scripts/test_hybrid_router.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_router import (HybridRouter, StaticPipeline, motion_from_frames,  # noqa: E402
                           FEATURE_COUNT, SEQ_LEN)

_SEQ_DIR = ROOT / "data" / "processed" / "dynamic_sequences"
_CACHE = {}
_ACTIVE_PROFILE = "current"


def make_router(profile: str | None = None) -> HybridRouter:
    profile = profile or _ACTIVE_PROFILE
    if "static" not in _CACHE:
        _CACHE["static"] = StaticPipeline()
        from dynamic_jz_inference import DynamicJZPredictor
        _CACHE["dynamic"] = DynamicJZPredictor()
    return HybridRouter(static=_CACHE["static"], dynamic=_CACHE["dynamic"],
                        profile=profile)

SEQ_DIR = ROOT / "data" / "processed" / "dynamic_sequences"
MANIFEST = ROOT / "data" / "processed" / "dynamic_manifest.csv"
STATIC_CSV = ROOT / "data" / "processed" / "sibi_landmarks.csv"
OUT = ROOT / "reports" / "hybrid_router_test.txt"


class Result:
    def __init__(self):
        self.items = []

    def add(self, name, passed, detail=""):
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def lm(x, y, z):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def load_manifest():
    d = {}
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[r["source_video"]] = r
    return d


def load_npz(label, stem):
    with np.load(SEQ_DIR / label / f"{stem}.npz", allow_pickle=True) as f:
        seq = np.asarray(f["sequence"], dtype=np.float64)
        meta = {k: f[k] for k in f.files if k != "sequence"}
    return seq, meta


def seq_to_raw_frames(seq, movement_total, wrist_total, offset=(0.5, 0.5)):
    """Ubah sequence (T,63) -> array raw (T,21,2) + z (T,21).

    Koordinat fitur sudah dinormalisasi skala tangan; untuk mensimulasikan
    koordinat kamera, skala relatif dikalibrasi agar total movement hasil
    simulasi mendekati nilai training (`movement_total`).
    """
    T = seq.shape[0]
    base = seq.reshape(T, 21, 3)
    xy_full = base[:, :, :2] + np.asarray(offset)
    z_full = base[:, :, 2]
    shape_norm = motion_from_frames(np.arange(T) / 30.0, xy_full).movement_magnitude
    shape_target = max(0.0, movement_total - wrist_total)
    s = shape_target / shape_norm if shape_norm > 1e-9 else 0.0
    s = float(np.clip(s, 0.0, 1.0))
    # Skala x, y, DAN z agar normalisasi ulang mengembalikan fitur asli.
    xy = base[:, :, :2] * s + np.asarray(offset)
    z = base[:, :, 2] * s

    rng = np.random.default_rng(0)
    steps = rng.normal(size=(T, 2))
    steps[0] = 0.0
    mag = np.linalg.norm(steps, axis=1)
    if mag.sum() > 0:
        steps = steps / mag.sum() * wrist_total
    wrist = np.cumsum(steps, axis=0)
    xy = xy + wrist[:, None, :]
    return xy, z


def frames_from_raw(xy, z, fps, t0=0.0):
    T = len(xy)
    out = []
    for i in range(T):
        t = t0 + i / fps
        ls = [lm(xy[i, k, 0], xy[i, k, 1], z[i, k]) for k in range(21)]
        out.append((t, ls))
    return out


def still_frames(reference_xy, reference_z, fps, n, t0):
    out = []
    for i in range(n):
        t = t0 + i / fps
        ls = [lm(reference_xy[k, 0], reference_xy[k, 1], reference_z[k])
              for k in range(21)]
        out.append((t, ls))
    return out


def run_frames(router, frames):
    events = []
    states = []
    last = None
    for t, ls in frames:
        out = router.update(t, ls)
        states.append(out.state)
        if out.event:
            events.append((out.event, out.dynamic_conf, out.state))
        last = out
    return events, states, last


def main() -> int:
    global _ACTIVE_PROFILE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["current", "c2", "c2_zadaptive"],
                    default="current")
    args = ap.parse_args()
    _ACTIVE_PROFILE = args.profile

    OUT.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    res = Result()
    print(f"== Test hybrid router (profile={args.profile}) ==")

    man = load_manifest()

    # ---------- J dan Z lengkap ----------
    for label in ("J", "Z"):
        stem = sorted((SEQ_DIR / label).glob("*.npz"))[0].stem
        seq, meta = load_npz(label, stem)
        wrist_total = float(meta["wrist_movement"])
        move_total = float(meta["movement_magnitude"])
        fps = float(meta["original_fps"]) if "original_fps" in meta else 25.0
        xy, z = seq_to_raw_frames(seq, move_total, wrist_total)
        router = make_router()
        pre = still_frames(xy[0], z[0], 30.0, 12, 0.0)
        gest = frames_from_raw(xy, z, fps, t0=12 / 30.0)
        post = still_frames(xy[-1], z[-1], 30.0, 14, gest[-1][0] + 1 / 30.0)
        frames = pre + gest + post
        events, states, last = run_frames(router, frames)
        got = [e[0] for e in events]
        res.add(f"Sequence {label} lengkap -> diterima sebagai {label}",
                label in got, f"events={got}")
        print(f"    {label}: route akhir={last.route} motion="
              f"{None if not last.motion else round(last.motion.movement_magnitude,3)} "
              f"conf={last.dynamic_conf:.3f} margin={last.dynamic_margin:.3f} "
              f"dist={last.dynamic_distance:.2f} reason='{last.dynamic_reason}'")

    # ---------- static gesture ----------
    with STATIC_CSV.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    feats = np.array([float(row[c]) for c in row if c != "label"])
    base = feats.reshape(21, 3) + np.array([0.5, 0.5, 0.0])
    rng = np.random.default_rng(1)
    router = make_router()
    frames = []
    t = 0.0
    for i in range(40):
        jitter = rng.normal(0, 0.0008, size=base.shape)
        p = base + jitter
        frames.append((t, [lm(p[k, 0], p[k, 1], p[k, 2]) for k in range(21)]))
        t += 1 / 30.0
    events, states, last = run_frames(router, frames)
    res.add("Static gesture -> route STATIC, tidak ada event dinamis",
            last.route == "STATIC" and not events,
            f"route={last.route} state={last.state} events={[e[0] for e in events]}")

    # ---------- small movement ----------
    router = make_router()
    frames = []
    t = 0.0
    p = base.copy()
    for i in range(40):
        p = base + np.array([0.002 * np.sin(i / 5), 0.002 * np.cos(i / 5), 0])
        frames.append((t, [lm(p[k, 0], p[k, 1], p[k, 2]) for k in range(21)]))
        t += 1 / 30.0
    events, states, last = run_frames(router, frames)
    res.add("Small movement -> tidak memicu dynamic",
            not events and last.route == "STATIC",
            f"route={last.route} events={[e[0] for e in events]}")

    # ---------- random trajectory ----------
    router = make_router()
    rng = np.random.default_rng(7)
    p = base.copy()
    frames = []
    t = 0.0
    for i in range(30):
        p = p + rng.normal(0, 0.02, size=base.shape)
        frames.append((t, [lm(p[k, 0], p[k, 1], p[k, 2]) for k in range(21)]))
        t += 1 / 25.0
    for i in range(15):
        frames.append((t, [lm(p[k, 0], p[k, 1], p[k, 2]) for k in range(21)]))
        t += 1 / 25.0
    events, states, last = run_frames(router, frames)
    res.add("Random trajectory -> dynamic ditolak",
            not events, f"events={[e[0] for e in events]} reason='{last.dynamic_reason}'")

    # ---------- incomplete J / Z ----------
    for label, frac in (("J", 0.25), ("Z", 0.25)):
        stem = sorted((SEQ_DIR / label).glob("*.npz"))[0].stem
        seq, meta = load_npz(label, stem)
        n = max(6, int(SEQ_LEN * frac))
        seq_part = seq[:n]
        wrist_total = float(meta["wrist_movement"]) * frac
        move_total = float(meta["movement_magnitude"]) * frac
        fps = float(meta["original_fps"]) if "original_fps" in meta else 25.0
        xy, z = seq_to_raw_frames(seq_part, move_total, wrist_total)
        router = make_router()
        pre = still_frames(xy[0], z[0], 30.0, 12, 0.0)
        gest = frames_from_raw(xy, z, fps, t0=12 / 30.0)
        post = still_frames(xy[-1], z[-1], 30.0, 14, gest[-1][0] + 1 / 30.0)
        events, states, last = run_frames(router, pre + gest + post)
        res.add(f"Incomplete {label} -> tidak diterima",
                len(events) == 0, f"events={[e[0] for e in events]} reason='{last.dynamic_reason}'")

    # ---------- transition-like (J lalu Z) ----------
    jseq, jmeta = load_npz("J", sorted((SEQ_DIR / "J").glob("*.npz"))[0].stem)
    zseq, zmeta = load_npz("Z", sorted((SEQ_DIR / "Z").glob("*.npz"))[0].stem)
    trans = np.concatenate([jseq[:12], zseq[-12:]], axis=0)
    wt = (float(jmeta["wrist_movement"]) + float(zmeta["wrist_movement"])) / 2
    mt = (float(jmeta["movement_magnitude"]) + float(zmeta["movement_magnitude"])) / 2
    xy, z = seq_to_raw_frames(trans, mt, wt)
    router = make_router()
    pre = still_frames(xy[0], z[0], 30.0, 12, 0.0)
    gest = frames_from_raw(xy, z, 30.0, t0=12 / 30.0)
    post = still_frames(xy[-1], z[-1], 30.0, 14, gest[-1][0] + 1 / 30.0)
    events, states, last = run_frames(router, pre + gest + post)
    res.add("Transition-like (J->Z) ditolak atau hanya satu event",
            len(events) <= 1, f"events={[e[0] for e in events]} reason='{last.dynamic_reason}'")

    # ---------- cooldown: gesture diulang cepat tidak menghasilkan JJ ----------
    seq, meta = load_npz("J", sorted((SEQ_DIR / "J").glob("*.npz"))[0].stem)
    xy, z = seq_to_raw_frames(seq, float(meta["movement_magnitude"]),
                              float(meta["wrist_movement"]))
    router = make_router()
    pre = still_frames(xy[0], z[0], 30.0, 12, 0.0)
    gest = frames_from_raw(xy, z, float(meta["original_fps"]), t0=12 / 30.0)
    gap = still_frames(xy[-1], z[-1], 30.0, 2, gest[-1][0] + 1 / 30.0)
    gest2 = frames_from_raw(xy, z, float(meta["original_fps"]), t0=gap[-1][0] + 0.05)
    events, states, last = run_frames(router, pre + gest + gap + gest2)
    res.add("Cooldown mencegah J berulang instan (<=1 event)",
            len(events) <= 1, f"events={[e[0] for e in events]}")

    elapsed = time.perf_counter() - start
    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    lines = [
        "TEST HARNESS - HYBRID ROUTER (offline)",
        f"Profile: {args.profile}",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Waktu: {elapsed:.2f} detik",
        f"Status: {'LULUS' if passed == total else 'ADA GAGAL'} ({passed}/{total})",
        "Catatan: synthetic negative hanya untuk unit test router.",
        "",
    ]
    for i, (name, p, detail) in enumerate(res.items, 1):
        lines.append(f"{i:2d}. [{'PASS' if p else 'FAIL'}] {name}"
                     + (f" - {detail}" if detail else ""))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nHasil: {passed}/{total} lulus")
    print(f"Rincian: {OUT.relative_to(ROOT)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
