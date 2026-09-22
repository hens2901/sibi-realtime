"""Evaluasi profil threshold (current / c2 / c2_zadaptive) pada event tersimpan.

Menggunakan event live yang sudah tersimpan (reports/realtime_dynamic_debug/)
dan synthetic negatives (unit-test fixture). Karena event tersimpan diambil
dengan profil `current`, replay ini mengevaluasi **gate logic** (distance,
movement, variation, trajectory) — bukan efek capture (pre/post-roll/onset).

GT default = pseudo-GT (prototipe terdekat). Jika tersedia
reports/realtime_dynamic_debug/gt_labels.json (timestamp -> "J"/"Z"/"neg"),
dipakai sebagai ground truth manual.

Output: reports/zadaptive_validation_metrics.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_router import HybridRouter, PROFILES  # noqa: E402
import calibrate_live_dynamic as CAL  # noqa: E402

OUT = ROOT / "reports" / "zadaptive_validation_metrics.csv"
GT_FILE = ROOT / "reports" / "realtime_dynamic_debug" / "gt_labels.json"


def trajectory_from_seq(seq):
    steps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    coverage = float(np.mean(steps > 0.01)) if len(steps) else 0.0
    # arah: proyeksi step ke sumbu dominan
    disp = np.diff(seq.reshape(len(seq), 21, 3)[:, 0, :2], axis=0)
    cross = disp[:-1, 0] * disp[1:, 1] - disp[:-1, 1] * disp[1:, 0]
    signs = np.sign(cross)
    dir_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    path = float(steps.sum())
    return coverage, dir_changes, path


def load_events(predictor, scaler, protos):
    gt = {}
    if GT_FILE.exists():
        gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    rows = []
    for r in CAL.add_distances(CAL.load_live(predictor), scaler, protos):
        seq = r["sequence"]
        cov, dc, path = trajectory_from_seq(seq)
        r["coverage"], r["dir_changes"], r["path"] = cov, dc, path
        key = r["file"].replace(".json", "")
        r["gt"] = gt.get(key) or gt.get(r["file"]) or None
        rows.append(r)
    return rows


def gate(row, thr, traj_params):
    pred = row["predicted"]
    d_own = row["dJ"] if pred == "J" else row["dZ"]
    mr = thr["movement_range"][pred]
    dmax = thr["distance_max_cls"][pred]
    vmin = thr["variation_min_cls"][pred]
    checks = {
        "confidence": row["conf"] >= thr["dynamic_conf_min"],
        "margin": row.get("margin", 1.0) >= thr["dynamic_margin_min"],
        "movement": mr[0] <= row["movement"] <= mr[1] * 1.2,
        "distance": d_own <= dmax,
        "variation": row["variation"] >= vmin,
        "moving_steps": row.get("moving_steps", 3) >= 3,
    }
    if traj_params:
        checks["trajectory"] = (row["coverage"] >= traj_params.get("min_coverage", 0)
                                and row["path"] >= traj_params.get("min_path", 0)
                                and row["dir_changes"] >= traj_params.get("min_direction_changes", 0))
    return all(checks.values()), checks


def label_of(row):
    """GT: manual bila ada, else pseudo-GT (prototipe terdekat)."""
    if row.get("gt"):
        return row["gt"]
    return row["near"]


def main() -> int:
    from dynamic_jz_inference import DynamicJZPredictor
    predictor = DynamicJZPredictor()
    scaler, protos = CAL.load_proto_scaler()

    live = load_events(predictor, scaler, protos)
    negs = CAL.gen_negatives(predictor, scaler, protos)
    for n in negs:
        n.setdefault("predicted", "J" if n["dJ"] < n["dZ"] else "Z")
        n["gt"] = "neg"

    profiles = {}
    for name in ("current", "c2", "c2_zadaptive"):
        r = HybridRouter(profile=name)
        # Trajectory gate dihitung dari RAW wrist path di router; tidak dapat
        # di-replay dari sequence tersimpan (wrist = origin setelah normalisasi).
        # Jadi replay offline mengevaluasi gate distance/movement/variation/
        # confidence/margin saja. Efek trajectory & capture butuh sesi live.
        profiles[name] = (r.thr, None)

    rows_out = []
    print(f"live={len(live)} negatives={len(negs)} "
          f"(GT manual={'ada' if GT_FILE.exists() else 'tidak, pakai pseudo-GT'})")
    for name, (thr, traj) in profiles.items():
        # genuine per kelas
        stats = defaultdict(int)
        for row in live:
            lab = label_of(row)
            if lab not in ("J", "Z"):
                continue
            ok, _ = gate(row, thr, traj)
            correct = ok and row["predicted"] == lab
            stats[f"{lab}_attempt"] += 1
            if correct:
                stats[f"{lab}_accept"] += 1
        neg_acc = 0
        neg_fj = neg_fz = 0
        for n in negs:
            ok, _ = gate(n, thr, traj)
            if ok:
                neg_acc += 1
                if n["predicted"] == "J":
                    neg_fj += 1
                else:
                    neg_fz += 1
        nJ, nZ = stats["J_attempt"], stats["Z_attempt"]
        line = {
            "profile": name,
            "J_true_accept": round(stats["J_accept"] / nJ, 3) if nJ else 0.0,
            "Z_true_accept": round(stats["Z_accept"] / nZ, 3) if nZ else 0.0,
            "false_J_rate": round(neg_fj / len(negs), 3) if negs else 0.0,
            "false_Z_rate": round(neg_fz / len(negs), 3) if negs else 0.0,
            "negative_rejection": round(1 - neg_acc / len(negs), 3) if negs else 0.0,
            "J_attempt": nJ, "Z_attempt": nZ, "neg_attempt": len(negs),
        }
        # latency dari event live (durasi capture/klasifikasi)
        def _cap(r):
            d = r.get("durations_ms", {}) or {}
            return d.get("capture_total") or (d.get("motion_to_classify", 0) + 0)
        caps = [_cap(r) for r in live]
        m2c = [(r.get("durations_ms", {}) or {}).get("motion_to_classify", 0) for r in live]
        line["avg_capture_ms"] = round(float(np.mean(caps)), 1) if caps else 0.0
        line["avg_motion_to_classify_ms"] = round(float(np.mean(m2c)), 1) if m2c else 0.0
        rows_out.append(line)
        print(f"{name:14s} J_accept={line['J_true_accept']:.2f} "
              f"Z_accept={line['Z_true_accept']:.2f} "
              f"falseJ={line['false_J_rate']:.2f} falseZ={line['false_Z_rate']:.2f} "
              f"neg_reject={line['negative_rejection']:.2f} "
              f"cap={line['avg_capture_ms']:.0f}ms")

    cols = list(rows_out[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    print(f"CSV: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
