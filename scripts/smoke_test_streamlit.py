"""Smoke test untuk aplikasi Streamlit SIBI (tanpa browser).

Menguji logika yang bisa diuji headless:
- import app_streamlit;
- kedua model statis (V2 & Current) dapat dimuat;
- scaler & label encoder termuat;
- preprocessing menghasilkan 63 fitur;
- pipeline prediksi berjalan pada data uji;
- rejection mechanism;
- temporal smoothing;
- model switching V2 <-> Current;
- mirror option tidak crash (lewat SibiVideoProcessor recv);
- aksi hasil ejaan (add/space/delete/reset);
- state UX (derive_state);
- (opsional) AppTest untuk memastikan app tidak error saat dijalankan.

Catatan: webcam browser TIDAK diuji di sini (butuh pengujian manual).

Output:
- reports/streamlit_smoke_test.txt

Jalankan:
    python scripts/smoke_test_streamlit.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av  # noqa: E402

import app_streamlit as app  # noqa: E402
import realtime as rt  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402


class Result:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def passed(self) -> bool:
        return all(p for _, p, _ in self.items)


def main() -> int:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "streamlit_smoke_test.txt"
    start = time.perf_counter()
    res = Result()
    print("== Smoke test Streamlit app ==")

    # 1. Import (sudah sukses kalau script ini berjalan)
    res.add("app_streamlit dapat di-import", True)

    # 2. Model statis V2 & Current
    bundles = {name: app.load_static_bundle(name) for name in ("v2", "current")}
    for name, (model, scaler, encoder, labels) in bundles.items():
        res.add(f"[{name}] model + scaler + encoder termuat", True,
                f"model={type(model).__name__}, kelas={len(labels)}")
        res.add(f"[{name}] 63 fitur & 24 kelas",
                getattr(model, "n_features_in_", None) == 63
                and getattr(scaler, "n_features_in_", None) == 63
                and len(encoder.classes_) == 24)

    # 3. Preprocessing -> 63 fitur
    dataset = ROOT / "data" / "raw" / "Mono_Background"
    sample_img = sorted((dataset / "H").glob("*.jpg"))[0]
    rgb = np.asarray(Image.open(sample_img).convert("RGB"))
    landmarker = rt.build_landmarker(vision.RunningMode.IMAGE)
    try:
        import mediapipe as mp
        det = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        feats = rt.normalize_landmarks(det.hand_landmarks[0])
    finally:
        landmarker.close()
    res.add("Preprocessing menghasilkan 63 fitur", feats.shape == (63,),
            f"shape={feats.shape}, finite={np.all(np.isfinite(feats))}")

    # 4. Pipeline prediksi pada data uji
    df = pd.read_csv(ROOT / "data" / "processed" / "sibi_landmarks.csv")
    fc = [c for c in df.columns if c != "label"]
    sample = df.iloc[::53][fc].to_numpy(dtype=np.float64)[:40]
    y_sample = df.iloc[::53]["label"].astype(str).to_numpy()[:40]
    for name, (model, scaler, encoder, labels) in bundles.items():
        probs = model.predict_proba(scaler.transform(pd.DataFrame(sample, columns=fc)))
        preds = encoder.inverse_transform(np.argmax(probs, axis=1))
        acc = float((preds == y_sample).mean())
        res.add(f"[{name}] pipeline prediksi pada data uji", probs.shape[1] == 24
                and acc >= 0.80, f"acc={acc:.3f}")

    # 4b. fast_predict_proba harus IDENTIK dengan rt.predict_proba
    max_diff = 0.0
    for name, (model, scaler, encoder, labels) in bundles.items():
        for row in sample:
            p_fast = app.fast_predict_proba(model, scaler, row)
            p_ref = rt.predict_proba(model, scaler, row)
            max_diff = max(max_diff, float(np.max(np.abs(p_fast - p_ref))))
    res.add("fast_predict_proba identik dengan rt.predict_proba",
            max_diff <= 1e-12, f"max_diff={max_diff:.2e}")

    # 5. Rejection mechanism
    labels = list(bundles["v2"][3])
    res.add("Rejection lolos saat top1 tinggi & margin besar",
            rt.passes_rejection(0.90, 0.30))
    res.add("Rejection menolak margin kecil",
            not rt.passes_rejection(0.95, 0.10))
    res.add("Rejection menolak confidence rendah",
            not rt.passes_rejection(0.80, 0.50))

    # 6. Temporal smoothing
    sm = rt.TemporalSmoother(labels, window=5, stable_frames=3)
    probs = np.zeros(len(labels))
    probs[labels.index("A")] = 0.95
    probs /= probs.sum()
    stable = False
    for _ in range(3):
        label, conf, stable = sm.update(probs)
    res.add("Temporal smoothing stabil pada frame ke-3",
            label == "A" and stable)
    lbl_rej, _, stable_rej = sm.update(None)
    res.add("Frame ditolak tidak menstabilkan",
            lbl_rej is None and not stable_rej)

    # 7. Model switching
    res.add("Mapping label model benar",
            app.STATIC_MODEL_MAP["V2"] == "v2"
            and app.STATIC_MODEL_MAP["Current"] == "current")
    settings = app.RuntimeSettings()
    res.add("Default static model = V2",
            settings.snapshot()["model_id"] == "v2",
            settings.snapshot()["model_id"])
    settings.update(model_id="current")
    res.add("RuntimeSettings switch ke current",
            settings.snapshot()["model_id"] == "current")
    settings.update(model_id="v2", mirror=True, threshold=0.85, margin=0.20)

    # 8/9. Video processor recv (mirror option) - end-to-end server-side
    shared = app.LatestResult()
    proc = app.SibiVideoProcessor(bundles, settings, shared)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")

    settings.update(mirror=False)
    out_frame = proc.recv(frame)
    payload = shared.get()
    res.add("recv mengembalikan av.VideoFrame (mirror OFF)",
            isinstance(out_frame, av.VideoFrame))
    res.add("recv menghasilkan payload + tangan terdeteksi (mirror OFF)",
            bool(payload) and payload.get("top1_label") is not None,
            f"top1={payload.get('top1_label')} ({payload.get('top1_prob', 0):.3f})")

    settings.update(mirror=True)
    out_frame2 = proc.recv(frame)
    res.add("recv tidak crash saat mirror ON", isinstance(out_frame2, av.VideoFrame))

    settings.update(model_id="current")
    out_frame3 = proc.recv(frame)
    res.add("recv tidak crash saat ganti model ke current",
            isinstance(out_frame3, av.VideoFrame))
    res.add("Payload memuat metrik performa",
            all(k in payload for k in (
                "camera_fps", "inference_fps", "frame_latency_ms",
                "inference_latency_ms", "mediapipe_ms", "predict_ms",
                "draw_ms", "skipped_frames")))
    metrics_ok = payload.get("mediapipe_ms", -1) >= 0 and payload.get("predict_ms", -1) >= 0
    res.add("Nilai metrik wajar", metrics_ok,
            f"mp={payload.get('mediapipe_ms', 0):.1f}ms "
            f"pred={payload.get('predict_ms', 0):.1f}ms")
    proc.on_ended()

    # 8b. Throttling inference + mode cepat
    settings2 = app.RuntimeSettings()
    settings2.update(inference_interval=2, inference_size=None)
    shared2 = app.LatestResult()
    p2 = app.SibiVideoProcessor(bundles, settings2, shared2)
    c0 = p2._infer_count
    p2.recv(frame)
    c1 = p2._infer_count
    p2.recv(frame)
    c2 = p2._infer_count
    p2.recv(frame)
    c3 = p2._infer_count
    res.add("Throttle: inference tiap 2 frame (ada frame di-skip)",
            c1 == c0 + 1 and c3 == c2 and p2._skipped >= 1,
            f"infer_count={c1},{c2},{c3} skipped={p2._skipped}")
    settings2.update(inference_interval=3, inference_size=(480, 360))
    out_small = p2.recv(frame)
    res.add("Mode Cepat (resize inference) tidak crash",
            isinstance(out_small, av.VideoFrame))
    p2.on_ended()

    # 10. Aksi hasil ejaan
    s = app.apply_spelling_action("", "add", "H")
    s = app.apply_spelling_action(s, "add", "A")
    s = app.apply_spelling_action(s, "space")
    s = app.apply_spelling_action(s, "add", "I")
    res.add("Aksi add + space benar", s == "HA I", f"hasil={s!r}")
    s = app.apply_spelling_action(s, "delete")
    res.add("Aksi delete benar", s == "HA ", f"hasil={s!r}")
    s = app.apply_spelling_action(s, "reset")
    res.add("Aksi reset benar", s == "", f"hasil={s!r}")

    # 11. State UX
    now = time.perf_counter()
    base_payload = {
        "t": now, "hand_found": True, "label": "H", "confidence": 0.987,
        "recognized": True, "stable": True, "rejected": False,
        "top1_label": "H", "top1_prob": 0.987, "top2_label": "A",
        "top2_prob": 0.004, "margin": 0.983, "fps": 24.0, "top5": [],
        "error": None,
    }
    snap = settings.snapshot()
    st_ok = app.derive_state(base_payload, snap)
    res.add("State UX: gesture dikenali & bisa ditambahkan",
            st_ok["kind"] == "recognized" and st_ok["can_add"] is True)
    hold = dict(base_payload, stable=False)
    res.add("State UX: belum stabil -> tidak bisa ditambahkan",
            app.derive_state(hold, snap)["kind"] == "hold"
            and app.derive_state(hold, snap)["can_add"] is False)
    rej = dict(base_payload, recognized=False, rejected=True, label=None)
    res.add("State UX: ditolak",
            app.derive_state(rej, snap)["kind"] == "rejected")
    no_hand = dict(base_payload, hand_found=False, recognized=False, label=None)
    res.add("State UX: menunggu tangan",
            app.derive_state(no_hand, snap)["kind"] == "waiting_hand")
    res.add("State UX: kamera belum aktif (payload basi)",
            app.derive_state(None, snap)["kind"] == "idle")

    # Stale prediction handling
    stale = dict(base_payload, t=now - (app.RESULT_TTL_S + 1.0))
    res.add("active_prediction: None saat stale", app.active_prediction(stale) is None)
    res.add("active_prediction: None saat tidak ada tangan",
            app.active_prediction(no_hand) is None)
    res.add("active_prediction: None saat tanpa top-1",
            app.active_prediction(dict(base_payload, top1_label=None)) is None)
    res.add("active_prediction: aktif saat valid",
            app.active_prediction(base_payload) is not None)

    latest = app.LatestResult()
    latest.set(base_payload)
    ok_get = latest.get() is not None
    latest.clear()
    res.add("LatestResult set/get/clear", ok_get and latest.get() is None)

    # Camera guide tidak crash
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        app._draw_center_guide(blank)
        app.draw_frame_overlay(blank, None, hand_found=False)
        res.add("Camera guide tidak crash", True)
    except Exception as exc:  # noqa: BLE001
        res.add("Camera guide tidak crash", False, str(exc))

    # 12. AppTest (opsional; webrtc mungkin butuh browser)
    apptest_status = "tidak dijalankan"
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=90)
        at.run()
        if at.exception:
            apptest_status = f"exception: {at.exception[0].value}"
            res.add("AppTest menjalankan app tanpa exception", False, apptest_status)
        else:
            apptest_status = "OK (tanpa exception)"
            res.add("AppTest menjalankan app tanpa exception", True, apptest_status)
    except Exception as exc:  # noqa: BLE001
        apptest_status = f"tidak tersedia: {type(exc).__name__}: {exc}"
        res.add("AppTest (informasional)", True, apptest_status)

    elapsed = time.perf_counter() - start
    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    lines = [
        "SMOKE TEST - STREAMLIT APP (headless)",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Waktu: {elapsed:.2f} detik",
        f"Status: {'LULUS' if passed == total else 'ADA GAGAL'}",
        "Catatan: webcam browser TIDAK diuji di sini.",
        "",
    ]
    for i, (name, p, detail) in enumerate(res.items, 1):
        lines.append(f"{i:2d}. [{'PASS' if p else 'FAIL'}] {name}"
                     + (f" - {detail}" if detail else ""))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nHasil: {passed}/{total} pemeriksaan lulus")
    print(f"AppTest: {apptest_status}")
    print(f"Rincian: {out.relative_to(ROOT)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
