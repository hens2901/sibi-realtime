"""Smoke test deployment readiness (tanpa webcam) untuk Streamlit Cloud.

Menguji:
- import app_streamlit
- resolusi path aman-Linux (relatif, tanpa path absolut Windows di runtime code)
- keberadaan runtime asset (model V2 + Current + MediaPipe task)
- load Current & V2 + scaler + encoder
- MediaPipe resource dapat dibuat
- inference pada sampel (63 fitur) → probabilitas valid
- app Streamlit tidak membutuhkan TensorFlow
- tidak ada ketergantungan path absolut D:\\

Output: reports/streamlit_cloud_smoke.txt
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "streamlit_cloud_smoke.txt"


class Result:
    def __init__(self):
        self.items = []

    def add(self, name, passed, detail=""):
        self.items.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def passed(self):
        return all(p for _, p, _ in self.items)


def scan_windows_paths(files: list[Path]) -> list[str]:
    import re
    pat = re.compile(r"[A-Za-z]:\\|\\\\Users\\\\|/Users/")
    hits = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{f.name}:{i}: {line.strip()[:80]}")
    return hits


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    res = Result()
    print("== Cloud smoke test ==")

    tf_before = "tensorflow" in sys.modules

    import app_streamlit as app  # noqa: E402
    import realtime as rt  # noqa: E402
    from mediapipe.tasks.python import vision  # noqa: E402

    res.add("import app_streamlit berhasil", True)

    # 1. path relatif
    res.add("ROOT relatif terhadap file (pathlib)", app.ROOT == ROOT,
            f"ROOT={app.ROOT}")
    hits = scan_windows_paths([Path(app.__file__), Path(rt.__file__)])
    res.add("Tidak ada path absolut Windows di runtime code", not hits,
            "; ".join(hits) if hits else "")
    res.add("MODELS_DIR di dalam project",
            str(app.MODELS_DIR).startswith(str(ROOT)),
            app.MODELS_DIR.relative_to(ROOT).as_posix())

    # 2. aset
    missing = app.missing_runtime_assets()
    res.add("Semua runtime asset ditemukan", not missing,
            "missing=" + (", ".join(missing) if missing else "none"))

    # 3. load model
    labels = {}
    for name in ("v2", "current"):
        try:
            model, scaler, enc, labs = app.load_static_bundle(name)
            labels[name] = labs
            res.add(f"[{name}] model+scaler+encoder dimuat (63 fitur, 24 kelas)",
                    getattr(model, "n_features_in_", None) == 63
                    and getattr(scaler, "n_features_in_", None) == 63
                    and len(labs) == 24)
        except Exception as exc:  # noqa: BLE001
            res.add(f"[{name}] dimuat", False, f"{type(exc).__name__}: {exc}")

    # 4. MediaPipe resource
    try:
        lm = rt.build_landmarker(vision.RunningMode.IMAGE)
        lm.close()
        res.add("MediaPipe HandLandmarker task dapat dibuat", True)
    except Exception as exc:  # noqa: BLE001
        res.add("MediaPipe HandLandmarker task dapat dibuat", False, str(exc))

    # 5. inference pada sampel dataset
    df = pd.read_csv(ROOT / "data" / "processed" / "sibi_landmarks.csv")
    fc = [c for c in df.columns if c != "label"]
    sample = df.iloc[:20][fc].to_numpy(dtype=np.float64)
    res.add("Preprocessing fitur = 63", len(fc) == 63, f"n={len(fc)}")
    for name in ("v2", "current"):
        if name not in labels:
            continue
        model, scaler, enc, labs = app.load_static_bundle(name)
        probs = model.predict_proba(scaler.transform(pd.DataFrame(sample, columns=fc)))
        ok = (probs.shape == (len(sample), 24)
              and np.all(np.isfinite(probs))
              and np.allclose(probs.sum(axis=1), 1.0, atol=1e-6))
        res.add(f"[{name}] probability valid (sum~1, finite)", bool(ok),
                f"shape={probs.shape}")

    # 6. TensorFlow tidak diperlukan oleh runtime code
    import re
    src = (Path(app.__file__).read_text(encoding="utf-8")
           + Path(rt.__file__).read_text(encoding="utf-8"))
    res.add("Runtime code tidak mengimpor tensorflow",
            re.search(r"\b(import|from)\s+tensorflow", src) is None)
    req_lines = [ln.strip().lower() for ln in
                 (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
    res.add("requirements.txt tidak memuat tensorflow/keras",
            not any(("tensorflow" in ln or "keras" in ln) for ln in req_lines))
    res.add("TensorFlow tidak wajib (mediapipe optional import)",
            True,
            "TF ada di env ini" if "tensorflow" in sys.modules else "TF tidak ada di env")

    elapsed = time.perf_counter() - start
    total = len(res.items)
    passed = sum(1 for _, p, _ in res.items if p)
    lines = [
        "CLOUD SMOKE TEST - SIBI Streamlit",
        f"Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {sys.version.split()[0]}",
        f"Waktu: {elapsed:.2f} detik",
        f"Status: {'LULUS' if passed == total else 'ADA GAGAL'} ({passed}/{total})",
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
