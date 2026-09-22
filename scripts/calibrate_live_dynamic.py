"""Kalibrasi live dynamic rejection J/Z berbasis data webcam nyata.

Input:
- reports/realtime_dynamic_debug/*.json + *.npz (45 dynamic event live)
- training sequences (data/processed/dynamic_sequences) untuk distribusi acuan
- synthetic negatives (unit-test fixture) untuk uji safety

Output:
- reports/live_dynamic_threshold_sweep.csv
- (ringkasan untuk reports/LIVE_DYNAMIC_CALIBRATION.md)

Catatan: pseudo-ground-truth live = label prototipe terdekat (karena tidak ada
label manual). Ini proksi, bukan GT manual.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEBUG = ROOT / "reports" / "realtime_dynamic_debug"
SEQ_DIR = ROOT / "data" / "processed" / "dynamic_sequences"
SCALER = ROOT / "models" / "dynamic_jz_scaler.joblib"
PROTO = ROOT / "models" / "dynamic_jz_prototypes.npz"
OUT_CSV = ROOT / "reports" / "live_dynamic_threshold_sweep.csv"

CURRENT = {
    "conf_min": 0.95, "margin_min": 0.20,
    "dist_max": {"J": 9.83, "Z": 5.04},
    "move_lo": {"J": 0.543, "Z": 0.120},
    "move_hi": {"J": 1.585, "Z": 0.661},
    "var_min": {"J": 0.102, "Z": 0.045},
}


def load_proto_scaler():
    scaler = joblib.load(SCALER)
    p = np.load(PROTO, allow_pickle=True)
    return scaler, {"J": p["J"].astype(np.float64), "Z": p["Z"].astype(np.float64)}


def distances(scaler, protos, seq):
    scaled = scaler.transform(seq.reshape(-1, 63)).reshape(24, 63)
    return {cl: float(np.mean(np.linalg.norm(scaled - protos[cl], axis=1)))
            for cl in ("J", "Z")}


def load_live(predictor):
    rows = []
    for jf in sorted(DEBUG.glob("*.json")):
        if jf.name == "events.jsonl":
            continue
        m = json.loads(jf.read_text(encoding="utf-8"))
        npz = jf.with_suffix(".npz")
        if not npz.exists():
            continue
        seq = np.load(npz)["sequence"].astype(np.float64)
        proba = predictor.predict(seq)[0]
        rows.append({
            "file": jf.name, "predicted": m["predicted"],
            "accepted_current": bool(m["accepted"]), "reason": m["reason"],
            "pJ": float(proba[0]), "pZ": float(proba[1]),
            "conf": float(m["confidence"]), "margin": float(m["margin"]),
            "movement": float(m["movement_magnitude"]),
            "variation": float(m["variation"]),
            "moving_steps": int(m["moving_steps"]),
            "durations_ms": m.get("durations_ms", {}),
            "profile": m.get("profile", ""),
            "motion_type": m.get("motion_type"),
            "sequence": seq,
        })
    return rows


def add_distances(rows, scaler, protos):
    for r in rows:
        d = distances(scaler, protos, r["sequence"])
        r["dJ"], r["dZ"] = d["J"], d["Z"]
        r["near"] = "J" if d["J"] < d["Z"] else "Z"
    return rows


# --------------------------------------------------------------------------- #
# Synthetic negatives (unit-test fixture) -> pseudo features
# --------------------------------------------------------------------------- #

def gen_negatives(predictor, scaler, protos):
    import hybrid_router as HR
    import test_hybrid_router as TH

    static = HR.StaticPipeline()
    from dynamic_jz_inference import DynamicJZPredictor
    dyn = DynamicJZPredictor()
    rng = np.random.default_rng(123)

    neg_records: list[dict] = []

    def run_frames(frames, name):
        r = HR.HybridRouter(static=static, dynamic=dyn)
        for t, ls in frames:
            out = r.update(t, ls)
            ev = out.dynamic_event
            if ev is not None and ev.sequence is not None:
                seq = np.asarray(ev.sequence, dtype=np.float64)
                d = distances(scaler, protos, seq)
                steps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
                cov = float(np.mean(steps > 0.01)) if len(steps) else 0.0
                disp = np.diff(seq.reshape(len(seq), 21, 3)[:, 0, :2], axis=0)
                cross = disp[:-1, 0] * disp[1:, 1] - disp[:-1, 1] * disp[1:, 0]
                sg = np.sign(cross)
                dc = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
                neg_records.append({
                    "name": name, "dJ": d["J"], "dZ": d["Z"],
                    "movement": float(ev.movement_magnitude),
                    "variation": float(ev.variation),
                    "conf": float(ev.confidence),
                    "margin": float(ev.margin),
                    "moving_steps": int(ev.moving_steps),
                    "predicted": ev.predicted,
                    "coverage": cov, "dir_changes": dc,
                    "path": float(steps.sum()),
                })

    def with_settle(xy, z, fps, name, pre=8, post=12):
        fr = TH.still_frames(xy[0], z[0], 30.0, pre, 0.0)
        fr += TH.frames_from_raw(xy, z, fps, t0=pre / 30.0)
        fr += TH.still_frames(xy[-1], z[-1], 30.0, post, fr[-1][0] + 1 / 30.0)
        run_frames(fr, name)

    base = np.zeros((21, 2))
    # random walk besar
    for k in range(4):
        p = base.copy(); xy = []
        for i in range(28):
            p = p + rng.normal(0, 0.02, size=(21, 2)); xy.append(p.copy())
        xy = np.array(xy); z = np.zeros((28, 21))
        with_settle(xy, z, 25.0, f"random{k}")

    # wave (sinusoidal)
    for k in range(3):
        xy = []
        for i in range(28):
            off = np.array([0.12 * np.sin(i / 3), 0.05 * np.cos(i / 4)])
            xy.append(base + off)
        xy = np.array(xy); z = np.zeros((28, 21))
        with_settle(xy, z, 25.0, f"wave{k}")

    # incomplete J/Z
    for lab, frac in (("J", 0.5), ("Z", 0.5), ("J", 0.25), ("Z", 0.25)):
        stem = sorted((SEQ_DIR / lab).glob("*.npz"))[0].stem
        seq, meta = TH.load_npz(lab, stem)
        n = max(6, int(24 * frac))
        sub = seq[:n]
        xy, z = TH.seq_to_raw_frames(sub, float(meta["movement_magnitude"]) * frac,
                                     float(meta["wrist_movement"]) * frac)
        with_settle(xy, z, float(meta["original_fps"]), f"incomplete_{lab}{int(frac*100)}")

    # transition J->Z
    jseq, jm = TH.load_npz("J", sorted((SEQ_DIR / "J").glob("*.npz"))[0].stem)
    zseq, zm = TH.load_npz("Z", sorted((SEQ_DIR / "Z").glob("*.npz"))[0].stem)
    trans = np.concatenate([jseq[:12], zseq[-12:]], axis=0)
    xy, z = TH.seq_to_raw_frames(trans, (jm["movement_magnitude"] + zm["movement_magnitude"]) / 2,
                                 (jm["wrist_movement"] + zm["wrist_movement"]) / 2)
    with_settle(xy, z, 30.0, "transition")

    return neg_records


# --------------------------------------------------------------------------- #
# Acceptance + sweep
# --------------------------------------------------------------------------- #

def accepts(row, params) -> bool:
    pred = row["predicted"]
    d_own = row["dJ"] if pred == "J" else row["dZ"]
    if row["conf"] < params["conf_min"]:
        return False
    if row.get("margin", 1.0) < params["margin_min"]:
        return False
    if row.get("moving_steps", 3) < 3:
        return False
    if not (params["move_lo"][pred] <= row["movement"] <= params["move_hi"][pred]):
        return False
    if d_own > params["dist_max"][pred]:
        return False
    if row["variation"] < params["var_min"][pred]:
        return False
    return True


def genuine_metrics(genuine, params):
    if not genuine:
        return 0.0, 0
    ok = sum(1 for r in genuine if accepts(r, params))
    return ok / len(genuine), ok


def neg_metrics(negs, params):
    if not negs:
        return 0.0, 0
    ok = sum(1 for r in negs if accepts(r, params))
    return ok / len(negs), ok


def main() -> int:
    print("== Kalibrasi live dynamic rejection ==")
    from dynamic_jz_inference import DynamicJZPredictor
    predictor = DynamicJZPredictor()
    scaler, protos = load_proto_scaler()

    live = add_distances(load_live(predictor), scaler, protos)
    print(f"live events: {len(live)}  predicted={Counter(r['predicted'] for r in live)} "
          f"accepted_current={sum(r['accepted_current'] for r in live)}")
    genuine = [r for r in live if r["near"] == r["predicted"]]  # GRU benar & dekat prototipe
    mismatch = [r for r in live if r["near"] != r["predicted"]]
    print(f"genuine (pred==near): {len(genuine)} | mismatch: {len(mismatch)}")

    # ---------- 1 & 2. Analisis per kelas ----------
    for lab in ("J", "Z"):
        g = [r for r in genuine if r["near"] == lab]
        print(f"\n--- {lab} (pseudo-GT {lab}, n={len(g)}) ---")
        for r in g:
            print(f"  pred={r['predicted']} pJ={r['pJ']:.2f} pZ={r['pZ']:.2f} "
                  f"dJ={r['dJ']:.2f} dZ={r['dZ']:.2f} move={r['movement']:.3f} "
                  f"var={r['variation']:.3f} acc={r['accepted_current']} | {r['reason']}")

    z_g = [r for r in genuine if r["near"] == "Z"]
    z_pred_j = [r for r in z_g if r["predicted"] == "J"]
    z_pred_z_rej = [r for r in z_g if r["predicted"] == "Z"]
    print(f"\nZ genuine: {len(z_g)} | GRU predicted J (A): {len(z_pred_j)} | "
          f"predicted Z but rejected (B): {len(z_pred_z_rej)}")

    # ---------- negatives ----------
    negs = gen_negatives(predictor, scaler, protos)
    print(f"\nsynthetic negatives (dynamic events): {len(negs)}")
    for n in negs:
        n.setdefault("predicted", "J" if n["dJ"] < n["dZ"] else "Z")
    for n in negs:
        pred = n["predicted"]
        d_own = n["dJ"] if pred == "J" else n["dZ"]
        print(f"  {n['name']:18s} pred={pred} move={n['movement']:.3f} "
              f"var={n['variation']:.3f} dJ={n['dJ']:.2f} dZ={n['dZ']:.2f} "
              f"conf={n['conf']:.2f} cur_acc={accepts(n, CURRENT)}")

    # ---------- 3. Distance sweep ----------
    sweep_rows = []
    d_grid = [5.04, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 9.83, 11.0, 12.0, 14.0]
    for dz in d_grid:
        for dj in (9.83, 12.0):
            p = json.loads(json.dumps(CURRENT))
            p["dist_max"] = {"J": dj, "Z": dz}
            ga, gok = genuine_metrics(genuine, p)
            nf, nok = neg_metrics(negs, p)
            sweep_rows.append({"kind": "distance", "param": f"dist_J={dj},dist_Z={dz}",
                               "genuine_accept": round(ga, 3), "genuine_n": f"{gok}/{len(genuine)}",
                               "neg_false_accept": round(nf, 3), "neg_n": f"{nok}/{len(negs)}"})

    # print distance sweep
    print("\n--- distance sweep (J fixed 9.83/12, Z varies) ---")
    for row in sweep_rows:
        print(f"  {row['param']:32s} genuine={row['genuine_accept']:.2f} ({row['genuine_n']}) "
              f"neg_false={row['neg_false_accept']:.2f} ({row['neg_n']})")

    # ---------- 4/5. movement & variation class-aware sweep ----------
    # candidate live-derived ranges
    def pct(vals, q):
        return float(np.percentile(vals, q)) if vals else 0.0
    for lab in ("J", "Z"):
        g = [r for r in genuine if r["near"] == lab]
        mv = [r["movement"] for r in g]
        vr = [r["variation"] for r in g]
        print(f"{lab} live genuine movement p5/p50/p95={pct(mv,5):.3f}/{pct(mv,50):.3f}/{pct(mv,95):.3f} "
              f"variation p5/p50/p95={pct(vr,5):.3f}/{pct(vr,50):.3f}/{pct(vr,95):.3f}")

    # ---------- combined recommended candidate ----------
    rec = json.loads(json.dumps(CURRENT))
    # class-specific from live: use p5..p95 movement, p5 variation, distance p95(+margin)
    for lab in ("J", "Z"):
        g = [r for r in genuine if r["near"] == lab]
        mv = [r["movement"] for r in g]
        vr = [r["variation"] for r in g]
        d = [r["dJ"] if lab == "J" else r["dZ"] for r in g]
        rec["move_lo"][lab] = max(0.05, pct(mv, 5))
        rec["move_hi"][lab] = pct(mv, 100) * 1.05
        rec["var_min"][lab] = max(0.01, pct(vr, 5) * 0.5)
        rec["dist_max"][lab] = pct(d, 100) * 1.05
    ga, gok = genuine_metrics(genuine, rec)
    nf, nok = neg_metrics(negs, rec)
    print("\n--- recommended candidate (derived from live genuine) ---")
    print(f"  move_lo={rec['move_lo']}")
    print(f"  move_hi={rec['move_hi']}")
    print(f"  var_min={rec['var_min']}")
    print(f"  dist_max={rec['dist_max']}")
    print(f"  genuine_accept={ga:.2f} ({gok}/{len(genuine)})  neg_false={nf:.2f} ({nok}/{len(negs)})")

    sweep_rows.append({"kind": "recommended", "param": json.dumps(rec),
                       "genuine_accept": round(ga, 3), "genuine_n": f"{gok}/{len(genuine)}",
                       "neg_false_accept": round(nf, 3), "neg_n": f"{nok}/{len(negs)}"})

    # ---------- candidate operating points ----------
    import copy
    def mk(move_j, move_z, var, dj, dz):
        p = copy.deepcopy(CURRENT)
        p["move_lo"] = {"J": move_j[0], "Z": move_z[0]}
        p["move_hi"] = {"J": move_j[1], "Z": move_z[1]}
        p["var_min"] = {"J": var, "Z": var}
        p["dist_max"] = {"J": dj, "Z": dz}
        return p

    candidates = {
        "C0_current": copy.deepcopy(CURRENT),
        "C1_conservative": mk((0.20, 1.60), (0.05, 1.40), 0.02, 9.83, 6.5),
        "C2_balanced": mk((0.20, 1.60), (0.05, 1.40), 0.02, 9.83, 7.5),
        "C3_extended": mk((0.20, 1.60), (0.05, 1.40), 0.02, 9.83, 9.5),
        "C4_live_derived": rec,
    }
    ood_neg = [n for n in negs if n["name"].startswith(("random", "wave"))]
    partial_neg = [n for n in negs if not n["name"].startswith(("random", "wave"))]

    print("\n--- candidate operating points ---")
    print(f"{'candidate':16s} {'genuine':>8s} {'neg_false':>10s} {'ood_false':>10s} {'partial_false':>14s}")
    for name, p in candidates.items():
        gacc, gn = genuine_metrics(genuine, p)
        nf, nn = neg_metrics(negs, p)
        of, on = neg_metrics(ood_neg, p)
        pf, pn = neg_metrics(partial_neg, p)
        print(f"{name:16s} {gacc:8.2f} {nf:10.2f} {of:10.2f} {pf:14.2f}  "
              f"(g {gn}/{len(genuine)}, neg {nn}/{len(negs)})")
        sweep_rows.append({"kind": "candidate", "param": name,
                           "genuine_accept": round(gacc, 3), "genuine_n": f"{gn}/{len(genuine)}",
                           "neg_false_accept": round(nf, 3), "neg_n": f"{nn}/{len(negs)}"})
    # also per-class genuine for best candidates
    for name in ("C2_balanced", "C3_extended"):
        p = candidates[name]
        for lab in ("J", "Z"):
            g = [r for r in genuine if r["near"] == lab]
            a, k = genuine_metrics(g, p)
            print(f"   {name} {lab}: genuine_accept={a:.2f} ({k}/{len(g)})")

    # ---------- 7. calibration vs validation split (event-level) ----------
    calib, val = [], []
    for lab in ("J", "Z"):
        items = [r for r in genuine if r["near"] == lab]
        for i, r in enumerate(items):
            (calib if i % 2 == 0 else val).append(r)
    neg_calib = [n for i, n in enumerate(negs) if i % 2 == 0]
    neg_val = [n for i, n in enumerate(negs) if i % 2 == 1]

    rec2 = json.loads(json.dumps(CURRENT))
    for lab in ("J", "Z"):
        g = [r for r in calib if r["near"] == lab]
        if not g:
            continue
        mv = [r["movement"] for r in g]
        vr = [r["variation"] for r in g]
        d = [r["dJ"] if lab == "J" else r["dZ"] for r in g]
        rec2["move_lo"][lab] = max(0.05, pct(mv, 5))
        rec2["move_hi"][lab] = pct(mv, 100) * 1.05
        rec2["var_min"][lab] = max(0.01, pct(vr, 5) * 0.5)
        rec2["dist_max"][lab] = pct(d, 100) * 1.05

    ga_c, gok_c = genuine_metrics(calib, rec2)
    ga_v, gok_v = genuine_metrics(val, rec2)
    nf_c, nok_c = neg_metrics(neg_calib, rec2)
    nf_v, nok_v = neg_metrics(neg_val, rec2)
    print("\n--- calibration/validation split (event-level, stratified) ---")
    print(f"  calib genuine={len(calib)} neg={len(neg_calib)} | val genuine={len(val)} neg={len(neg_val)}")
    print(f"  CALIB  genuine_accept={ga_c:.2f} ({gok_c}/{len(calib)})  neg_false={nf_c:.2f} ({nok_c}/{len(neg_calib)})")
    print(f"  VALID  genuine_accept={ga_v:.2f} ({gok_v}/{len(val)})  neg_false={nf_v:.2f} ({nok_v}/{len(neg_val)})")
    print(f"  rec2 params: move_lo={rec2['move_lo']} move_hi={rec2['move_hi']} "
          f"var_min={rec2['var_min']} dist_max={rec2['dist_max']}")
    sweep_rows.append({"kind": "recommended_cv_calib", "param": json.dumps(rec2),
                       "genuine_accept": round(ga_c, 3), "genuine_n": f"{gok_c}/{len(calib)}",
                       "neg_false_accept": round(nf_c, 3), "neg_n": f"{nok_c}/{len(neg_calib)}"})
    sweep_rows.append({"kind": "recommended_cv_valid", "param": json.dumps(rec2),
                       "genuine_accept": round(ga_v, 3), "genuine_n": f"{gok_v}/{len(val)}",
                       "neg_false_accept": round(nf_v, 3), "neg_n": f"{nok_v}/{len(neg_val)}"})

    # ---------- write CSV ----------
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "param", "genuine_accept",
                                           "genuine_n", "neg_false_accept", "neg_n"])
        w.writeheader()
        w.writerows(sweep_rows)

    # ---------- write live detail CSV for report ----------
    detail = ROOT / "reports" / "live_dynamic_events_detail.csv"
    with detail.open("w", newline="", encoding="utf-8") as fh:
        cols = ["file", "predicted", "near", "accepted_current", "pJ", "pZ", "conf",
                "margin", "dJ", "dZ", "movement", "variation", "moving_steps", "reason"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(live, key=lambda x: (x["near"], x["predicted"])):
            w.writerow({k: r.get(k) for k in cols})

    print(f"\nCSV sweep  : {OUT_CSV.relative_to(ROOT)}")
    print(f"CSV detail : {detail.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
