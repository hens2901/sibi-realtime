"""Scanner SIBI real-time: webcam + MediaPipe Hand Landmarker + ANN (MLPClassifier).

Model yang digunakan (TIDAK diubah dan TIDAK ditraining ulang):
- models/sibi_mlp.joblib
- models/scaler.joblib
- models/label_encoder.joblib
- models/hand_landmarker.task

MEKANISME PENOLAKAN (REJECTION)
-------------------------------
Model bersifat closed-set (selalu memilih salah satu dari 24 kelas), sehingga pose
tangan non-SIBI bisa dipaksa masuk ke salah satu huruf. Untuk itu ditambahkan
rejection mechanism berbasis confidence + margin:

1. Hitung top-1 dan top-2 probability.
2. margin = p(top-1) - p(top-2).
3. Prediksi diterima HANYA jika:
   - p(top-1) >= CONFIDENCE_THRESHOLD (default 0.85), DAN
   - margin     >= MARGIN_THRESHOLD     (default 0.20).
4. Jika salah satu tidak terpenuhi -> tampil "Tidak dikenali" dan frame tersebut
   TIDAK ikut masuk ke buffer temporal smoothing.
5. Temporal smoothing hanya menstabilkan prediksi yang lolos rejection, sehingga
   pose transisi tidak langsung ditambahkan ke hasil ejaan.

BATASAN PENTING
---------------
Model hanya mendukung 24 gesture SIBI statis: A-I dan K-Y.
Huruf J dan Z TIDAK didukung karena tidak ada di dataset (gestur dinamis)
dan secara sengaja TIDAK dibuatkan dukungan palsu/sintetis.

Preprocessing realtime dibuat identik dengan `scripts/extract_landmarks.py`:
- 21 landmark tangan (x, y, z) -> 63 fitur;
- translasi relatif terhadap wrist;
- normalisasi skala berdasarkan ukuran tangan
  (max jarak landmark ke wrist).

Jalankan:
    python realtime.py

Kontrol:
    Q          keluar
    R          reset teks ejaan
    SPASI      tambahkan huruf stabil ke hasil ejaan
    BACKSPACE  hapus karakter terakhir
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------- #
# Path & konstanta
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

MLP_PATH = MODELS_DIR / "sibi_mlp.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

# Model alternatif (hasil eksperimen mirror augmentation). Dipilih via --model.
MLP_PATH_AUGMENTED = MODELS_DIR / "sibi_mlp_augmented.joblib"
SCALER_PATH_AUGMENTED = MODELS_DIR / "scaler_augmented.joblib"
ENCODER_PATH_AUGMENTED = MODELS_DIR / "label_encoder_augmented.joblib"

# name -> (model, scaler, encoder). "baseline" tetap default.
MODEL_ARTIFACTS = {
    "baseline": (MLP_PATH, SCALER_PATH, ENCODER_PATH),
    "augmented": (MLP_PATH_AUGMENTED, SCALER_PATH_AUGMENTED, ENCODER_PATH_AUGMENTED),
}
MODEL_CHOICES = tuple(MODEL_ARTIFACTS.keys())
DEFAULT_MODEL = "baseline"


NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # 63

CONFIDENCE_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.20
SMOOTHING_WINDOW = 10      # jumlah frame terakhir untuk rata-rata probabilitas
STABLE_FRAMES = 8          # jumlah frame konsisten agar dianggap "stabil"
MISS_GRACE_FRAMES = 3      # toleransi frame tanpa tangan sebelum buffer direset

SPACE_COOLDOWN_S = 0.35
BACKSPACE_COOLDOWN_S = 0.12

WINDOW_NAME = "SIBI Realtime Scanner"

# Kelas yang didukung model (24 statis). J dan Z sengaja tidak ada.
SUPPORTED_CLASSES = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y",
)
UNSUPPORTED_CLASSES = ("J", "Z")

# Koneksi antar landmark untuk menggambar kerangka tangan (indeks MediaPipe).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # jempol
    (0, 5), (5, 6), (6, 7), (7, 8),          # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),     # tengah
    (9, 13), (13, 14), (14, 15), (15, 16),   # manis
    (13, 17), (17, 18), (18, 19), (19, 20),  # kelingking
    (0, 17),                                  # telapak
)


# --------------------------------------------------------------------------- #
# Loading artefak
# --------------------------------------------------------------------------- #

def load_artifacts(model_name: str = DEFAULT_MODEL):
    """Muat model, scaler, dan label encoder. Tidak mengubah file model.

    model_name: "baseline" (default) atau "augmented".
    """
    model_name = (model_name or DEFAULT_MODEL).lower()
    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(
            f"Model tidak dikenal: {model_name!r}. Pilihan: {MODEL_CHOICES}."
        )
    mlp_path, scaler_path, encoder_path = MODEL_ARTIFACTS[model_name]

    for path in (mlp_path, scaler_path, encoder_path, LANDMARKER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Artefak tidak ditemukan: {path}")

    model = joblib.load(mlp_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    n_in = getattr(model, "n_features_in_", None)
    if n_in != NUM_FEATURES:
        raise ValueError(
            f"Model mengharapkan {n_in} fitur, program ini menyiapkan {NUM_FEATURES}."
        )

    if getattr(scaler, "n_features_in_", NUM_FEATURES) != NUM_FEATURES:
        raise ValueError("Scaler tidak cocok: jumlah fitur bukan 63.")

    labels = [str(x) for x in encoder.classes_]
    if len(labels) != len(SUPPORTED_CLASSES):
        raise ValueError(
            f"Label encoder berisi {len(labels)} kelas, diharapkan "
            f"{len(SUPPORTED_CLASSES)} (A-I, K-Y)."
        )

    # Pastikan model & encoder konsisten terhadap urutan kelas.
    if not np.array_equal(model.classes_, np.arange(len(labels))):
        raise ValueError("Urutan kelas model tidak konsisten dengan label encoder.")

    return model, scaler, encoder, labels


def build_landmarker(running_mode) -> "vision.HandLandmarker":
    base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# Preprocessing (IDENTIK dengan scripts/extract_landmarks.py)
# --------------------------------------------------------------------------- #

def normalize_landmarks(landmarks) -> np.ndarray:
    """Translasi relatif wrist lalu skala berdasarkan ukuran tangan.

    Ini adalah salinan logika `normalize_landmarks` pada
    `scripts/extract_landmarks.py` agar fitur realtime setara dengan fitur
    training. Output: array 1D berisi tepat 63 nilai float.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Jumlah landmark harus {NUM_LANDMARKS}, ditemukan {pts.shape}."
        )

    wrist = pts[0]
    translated = pts - wrist

    scale = float(np.max(np.linalg.norm(translated, axis=1)))
    if not math.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    normalized = translated / scale
    return normalized.reshape(-1)


def predict_proba(model, scaler, features) -> np.ndarray:
    """Terapkan scaler lalu jalankan ANN; kembalikan vektor probabilitas."""
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != NUM_FEATURES:
        raise ValueError(
            f"Input harus {NUM_FEATURES} fitur, ditemukan {x.shape[1]}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Input fitur mengandung NaN/inf.")

    # Scaler dilatih dengan DataFrame (punya nama fitur f1..f63). Bungkus ulang
    # agar tidak memunculkan warning dan urutan fitur tetap sesuai training.
    if hasattr(scaler, "feature_names_in_"):
        x = pd.DataFrame(x, columns=list(scaler.feature_names_in_))

    x_scaled = scaler.transform(x)
    return model.predict_proba(x_scaled)[0]


def class_labels(model, encoder) -> list[str]:
    """Nama kelas sesuai urutan kolom probabilitas model."""
    return [str(x) for x in encoder.inverse_transform(model.classes_)]


@dataclass
class Top2:
    """Hasil peringkat dua probabilitas teratas untuk satu frame."""
    top1_label: str
    top1_prob: float
    top2_label: str
    top2_prob: float
    margin: float


def top2_from_probs(probs: np.ndarray, labels: list[str]) -> Top2:
    """Ambil top-1 dan top-2 dari vektor probabilitas, hitung margin."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(labels):
        raise ValueError(
            f"Panjang probabilitas {probs.shape[0]} != jumlah kelas {len(labels)}."
        )
    order = np.argsort(probs)[::-1]
    i1 = int(order[0])
    i2 = int(order[1]) if probs.shape[0] > 1 else int(order[0])
    p1 = float(probs[i1])
    p2 = float(probs[i2])
    return Top2(labels[i1], p1, labels[i2], p2, p1 - p2)


def passes_rejection(
    top1_prob: float,
    margin: float,
    threshold: float = CONFIDENCE_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> bool:
    """True jika prediksi lolos confidence threshold DAN margin threshold."""
    return top1_prob >= threshold and margin >= margin_threshold



# --------------------------------------------------------------------------- #
# Temporal smoothing
# --------------------------------------------------------------------------- #

class TemporalSmoother:
    """Rata-ratakan probabilitas beberapa frame terakhir agar stabil.

    PENTING: buffer hanya boleh diisi oleh frame yang LOLOS rejection. Frame
    yang ditolak harus dikirim sebagai ``None`` agar tidak menstabilkan pose
    non-SIBI/transisi. Pengaturan ini dilakukan oleh ``SibiRecognizer``.
    """

    def __init__(
        self,
        labels: list[str],
        window: int = SMOOTHING_WINDOW,
        stable_frames: int = STABLE_FRAMES,
        miss_grace: int = MISS_GRACE_FRAMES,
    ) -> None:
        self.labels = labels
        self.window = max(1, int(window))
        self.stable_frames = max(1, int(stable_frames))
        self.miss_grace = max(0, int(miss_grace))
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._stable_label: str | None = None
        self._stable_count = 0
        self._miss_count = 0

    def reset(self) -> None:
        self._buf.clear()
        self._stable_label = None
        self._stable_count = 0
        self._miss_count = 0

    def update(self, probs: np.ndarray | None) -> tuple[str | None, float, bool]:
        """Kembalikan (label_smoothed, confidence, stabil)."""
        if probs is None:
            self._miss_count += 1
            if self._miss_count > self.miss_grace:
                self.reset()
            return None, 0.0, False

        self._miss_count = 0
        self._buf.append(np.asarray(probs, dtype=np.float64))

        avg = np.mean(np.stack(self._buf), axis=0)
        idx = int(np.argmax(avg))
        label = self.labels[idx]
        confidence = float(avg[idx])

        if label == self._stable_label:
            self._stable_count += 1
        else:
            self._stable_label = label
            self._stable_count = 1

        return label, confidence, self._stable_count >= self.stable_frames


# --------------------------------------------------------------------------- #
# Recognizer (pipeline per-frame)
# --------------------------------------------------------------------------- #

@dataclass
class FrameResult:
    hand_found: bool
    label: str | None          # label setelah smoothing
    confidence: float          # confidence setelah smoothing
    recognized: bool           # lolos rejection (top1 >= thr dan margin >= margin_thr)
    stable: bool               # sudah stabil selama N frame
    landmarks: list | None     # landmark MediaPipe (21 titik)
    bbox: tuple[int, int, int, int] | None
    # --- info debugging rejection (nilai per-frame, sebelum smoothing) ---
    top1_label: str | None = None
    top1_prob: float = 0.0
    top2_label: str | None = None
    top2_prob: float = 0.0
    margin: float = 0.0
    rejected: bool = False     # ada tangan tetapi tidak lolos rejection


class SibiRecognizer:
    """Pipeline deteksi -> landmark -> preprocess -> scaler -> ANN -> decode."""

    def __init__(
        self,
        model,
        scaler,
        encoder,
        landmarker,
        threshold: float = CONFIDENCE_THRESHOLD,
        margin_threshold: float = MARGIN_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
        stable_frames: int = STABLE_FRAMES,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.encoder = encoder
        self.landmarker = landmarker
        self.threshold = float(threshold)
        self.margin_threshold = float(margin_threshold)
        self.labels = class_labels(model, encoder)
        self.smoother = TemporalSmoother(
            self.labels, window=smoothing_window, stable_frames=stable_frames
        )
        self.mode = "VIDEO"

    def set_mode(self, mode: str) -> None:
        self.mode = mode.upper()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> FrameResult:
        """Proses satu frame BGR. Mengembalikan FrameResult."""
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.mode == "VIDEO":
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.landmarker.detect(mp_image)

        h, w = frame_bgr.shape[:2]

        if not result.hand_landmarks:
            self.smoother.update(None)
            return FrameResult(False, None, 0.0, False, False, None, None)

        landmarks = result.hand_landmarks[0]
        try:
            features = normalize_landmarks(landmarks)
            probs = predict_proba(self.model, self.scaler, features)
        except ValueError:
            self.smoother.update(None)
            return FrameResult(False, None, 0.0, False, False, None, None)

        bbox = compute_bbox(landmarks, w, h)

        # --- Rejection mechanism (per-frame, sebelum smoothing) ---
        top2 = top2_from_probs(probs, self.labels)
        accepted = passes_rejection(
            top2.top1_prob, top2.margin, self.threshold, self.margin_threshold
        )

        if not accepted:
            # Frame ditolak: TIDAK dimasukkan ke buffer smoothing supaya pose
            # transisi / non-SIBI tidak menstabilkan prediksi apa pun.
            self.smoother.update(None)
            return FrameResult(
                hand_found=True,
                label=None,
                confidence=top2.top1_prob,
                recognized=False,
                stable=False,
                landmarks=landmarks,
                bbox=bbox,
                top1_label=top2.top1_label,
                top1_prob=top2.top1_prob,
                top2_label=top2.top2_label,
                top2_prob=top2.top2_prob,
                margin=top2.margin,
                rejected=True,
            )

        # Lolos rejection -> baru boleh masuk temporal smoothing.
        label, confidence, stable = self.smoother.update(probs)

        return FrameResult(
            hand_found=True,
            label=label,
            confidence=confidence,
            recognized=True,
            stable=stable,
            landmarks=landmarks,
            bbox=bbox,
            top1_label=top2.top1_label,
            top1_prob=top2.top1_prob,
            top2_label=top2.top2_label,
            top2_prob=top2.top2_prob,
            margin=top2.margin,
            rejected=False,
        )


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def compute_bbox(landmarks, width: int, height: int, pad: int = 20):
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(width - 1, int(max(xs)) + pad)
    y1 = min(height - 1, int(max(ys)) + pad)
    return x0, y0, x1, y1


def draw_hand(frame: np.ndarray, landmarks, bbox, label: str, recognized: bool) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        color = (0, 0, 255) if i == 0 else (255, 160, 0)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        color = (0, 255, 0) if recognized else (0, 140, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        text = label if recognized else "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        ty = y0 - 12 if y0 - 12 - th > 0 else y1 + th + 12
        tx = min(max(0, x0), max(0, w - tw - 4))
        cv2.putText(frame, text, (tx, ty), font, 1.0, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    result: FrameResult,
    spelling: str,
    fps: float,
    threshold: float,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    banner_h = min(190, max(140, h // 4))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Baris 1: hasil ejaan
    shown = spelling if spelling else "(kosong)"
    if len(shown) > 42:
        shown = "..." + shown[-39:]
    cv2.putText(frame, f"Ejaan: {shown}", (14, 32), font, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)

    fps_txt = f"FPS: {fps:4.1f}"
    (fw, _), _ = cv2.getTextSize(fps_txt, font, 0.6, 2)
    cv2.putText(frame, fps_txt, (w - fw - 14, 32), font, 0.6,
                (200, 200, 200), 2, cv2.LINE_AA)

    # Posisi baris dari bawah agar rapi pada banner yang lebih tinggi.
    hint_y = banner_h - 10
    debug2_y = hint_y - 24
    debug1_y = debug2_y - 24
    status_y = debug1_y - 30

    # Baris 2: status prediksi
    if not result.hand_found:
        status, color = "Tidak ada tangan terdeteksi", (0, 200, 255)
    elif result.recognized:
        status, color = f"{result.label}  ({result.confidence * 100:.1f}%)", (0, 255, 0)
    else:
        status = "Tidak dikenali"
        color = (0, 100, 255)
    cv2.putText(frame, status, (14, status_y), font, 0.85, color, 2, cv2.LINE_AA)

    if result.stable and result.recognized:
        tag = "STABIL"
        (tw, _), _ = cv2.getTextSize(tag, font, 0.6, 2)
        cv2.putText(frame, tag, (w - tw - 14, status_y), font, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)

    # Baris debugging rejection (top-1, top-2, margin).
    if result.hand_found and result.top1_label is not None:
        t1 = f"top1: {result.top1_label} {result.top1_prob * 100:.1f}%"
        t2 = f"top2: {result.top2_label} {result.top2_prob * 100:.1f}%"
        mg = f"margin: {result.margin * 100:.1f}%"
        cv2.putText(frame, f"{t1}   {t2}   {mg}", (14, debug1_y), font, 0.58,
                    (235, 235, 235), 1, cv2.LINE_AA)
        verdict = "LOLOS" if result.recognized else "DITOLAK"
        vcolor = (0, 255, 0) if result.recognized else (0, 140, 255)
        info = (
            f"threshold: {threshold * 100:.0f}% | "
            f"margin_min: {margin_threshold * 100:.0f}% | status: {verdict}"
        )
        cv2.putText(frame, info, (14, debug2_y), font, 0.58, vcolor, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "top1: -   top2: -   margin: -", (14, debug1_y),
                    font, 0.58, (160, 160, 160), 1, cv2.LINE_AA)
        info = (
            f"threshold: {threshold * 100:.0f}% | "
            f"margin_min: {margin_threshold * 100:.0f}% | status: -"
        )
        cv2.putText(frame, info, (14, debug2_y), font, 0.58,
                    (160, 160, 160), 1, cv2.LINE_AA)

    hint = "Q: keluar | R: reset | SPASI: simpan huruf | BACKSPACE: hapus"
    cv2.putText(frame, hint, (14, hint_y), font, 0.5,
                (170, 170, 170), 1, cv2.LINE_AA)

    # Bar confidence
    bar_x, bar_w, bar_h = 14, max(80, w - 28), 10
    bar_y = banner_h + 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    if result.hand_found:
        ratio = max(0.0, min(1.0, result.confidence))
        filled = int(bar_w * ratio)
        bar_color = (0, 255, 0) if result.recognized else (0, 100, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      bar_color, -1)
    thr_x = bar_x + int(bar_w * threshold)
    cv2.line(frame, (thr_x, bar_y - 3), (thr_x, bar_y + bar_h + 3),
             (255, 255, 255), 2)


# --------------------------------------------------------------------------- #
# Kamera & loop utama
# --------------------------------------------------------------------------- #

def open_camera(index: int, width: int, height: int):
    backends = [cv2.CAP_ANY]
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def run_camera(args) -> int:
    model_name = getattr(args, "model", DEFAULT_MODEL)
    model, scaler, encoder, labels = load_artifacts(model_name)
    landmarker = build_landmarker(vision.RunningMode.VIDEO)
    recognizer = SibiRecognizer(
        model, scaler, encoder, landmarker,
        threshold=args.threshold,
        margin_threshold=args.margin,
        smoothing_window=args.smoothing,
        stable_frames=args.stable_frames,
    )
    recognizer.set_mode("VIDEO")

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print(
            f"ERROR: tidak dapat membuka webcam (index {args.camera}).\n"
            "Coba tutup aplikasi lain yang memakai kamera, atau ganti index "
            "dengan --camera 1.",
            file=sys.stderr,
        )
        landmarker.close()
        return 1

    print(f"Model aktif: {model_name}")
    print(f"Model siap. Kelas didukung ({len(labels)}): {', '.join(labels)}")
    print("J/Z tidak didukung (tidak ada di dataset).")
    print("Tekan Q untuk keluar.")

    spelling = ""
    last_space = 0.0
    last_backspace = 0.0
    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: frame kamera tidak terbaca; keluar.", file=sys.stderr)
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0 else (0.9 * fps + 0.1 * inst_fps)

            result = recognizer.process(frame, int(now * 1000))

            # Gambar overlay lalu tampilkan frame.
            draw_overlay(frame, result, spelling, fps, args.threshold, args.margin)
            if result.hand_found and result.landmarks:
                draw_hand(frame, result.landmarks, result.bbox,
                          result.label or "?", result.recognized)

            cv2.imshow(WINDOW_NAME, frame)

            # Tangani input keyboard.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                spelling = ""
                recognizer.smoother.reset()
            elif key in (8, 127):  # backspace
                if now - last_backspace >= BACKSPACE_COOLDOWN_S:
                    spelling = spelling[:-1]
                    last_backspace = now
            elif key == 32:  # space
                # Hanya prediksi yang LOLOS rejection DAN stabil yang boleh
                # ditambahkan. Pose transisi (rejected / belum stabil) diabaikan.
                if (
                    result.hand_found
                    and result.recognized
                    and result.stable
                    and now - last_space >= SPACE_COOLDOWN_S
                ):
                    spelling += result.label
                    last_space = now
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    return 0


# --------------------------------------------------------------------------- #
# Check offline (tanpa webcam)
# --------------------------------------------------------------------------- #

def run_check(model_name: str = DEFAULT_MODEL) -> int:
    """Validasi ringan: artefak dapat dimuat dan pipeline menghasilkan 63 fitur."""
    print(f"== SIBI Realtime self-check (model={model_name}) ==")
    model, scaler, encoder, labels = load_artifacts(model_name)
    print(f"[OK] model/scaler/encoder dimuat. Kelas: {len(labels)}")

    missing = [c for c in UNSUPPORTED_CLASSES if c in labels]
    if missing:
        print(f"[FAIL] Kelas tidak didukung ditemukan di encoder: {missing}")
        return 1
    print("[OK] J dan Z tidak ada di encoder (sesuai dataset).")

    # Vektor nol tetap harus 63 fitur dan menghasilkan label valid.
    feats = np.zeros(NUM_FEATURES, dtype=np.float64)
    probs = predict_proba(model, scaler, feats)
    if probs.shape[0] != len(labels):
        print("[FAIL] Panjang probabilitas tidak sesuai jumlah kelas.")
        return 1
    pred = labels[int(np.argmax(probs))]
    print(f"[OK] Prediksi vektor nol: {pred} ({float(np.max(probs)):.4f})")

    # Input dengan jumlah fitur salah harus ditolak.
    try:
        predict_proba(model, scaler, np.zeros(10))
    except ValueError:
        print("[OK] Input != 63 fitur ditolak.")
    else:
        print("[FAIL] Input != 63 fitur tidak ditolak.")
        return 1

    # --- Pemeriksaan rejection mechanism ---
    print(f"[INFO] confidence threshold = {CONFIDENCE_THRESHOLD:.2f}, "
          f"margin threshold = {MARGIN_THRESHOLD:.2f}")

    t2 = top2_from_probs(probs, labels)
    print(f"[OK] top1={t2.top1_label} ({t2.top1_prob:.4f}), "
          f"top2={t2.top2_label} ({t2.top2_prob:.4f}), margin={t2.margin:.4f}")

    cases = [
        ("top1 tinggi + margin besar -> lolos",
         np.array([0.90, 0.05, 0.05] + [0.0] * (len(labels) - 3)), True),
        ("top1 tinggi + margin kecil -> ditolak",
         np.array([0.50, 0.48, 0.02] + [0.0] * (len(labels) - 3)), False),
        ("top1 rendah -> ditolak",
         np.array([0.40, 0.30, 0.30] + [0.0] * (len(labels) - 3)), False),
    ]
    for name, vec, expect in cases:
        c = top2_from_probs(vec, labels)
        got = passes_rejection(c.top1_prob, c.margin)
        if got != expect:
            print(f"[FAIL] {name}: got={got}")
            return 1
        print(f"[OK] {name} (margin={c.margin:.2f})")

    print("Semua pemeriksaan dasar lulus.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scanner SIBI real-time (24 gesture statis: A-I, K-Y).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="index webcam")
    parser.add_argument("--width", type=int, default=1280, help="lebar capture")
    parser.add_argument("--height", type=int, default=720, help="tinggi capture")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="confidence threshold top-1")
    parser.add_argument("--margin", type=float, default=MARGIN_THRESHOLD,
                        help="margin threshold (top1 - top2)")
    parser.add_argument("--smoothing", type=int, default=SMOOTHING_WINDOW,
                        help="jumlah frame untuk temporal smoothing")
    parser.add_argument("--stable-frames", type=int, default=STABLE_FRAMES,
                        help="jumlah frame konsisten agar stabil")
    parser.add_argument("--mirror", dest="mirror", action="store_true",
                        default=True, help="mode cermin (default aktif)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="matikan mode cermin (coba jika akurasi buruk)")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
                        help="pilih model: baseline (default) atau augmented")
    parser.add_argument("--check", action="store_true",
                        help="jalankan validasi offline tanpa webcam")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.check:
        return run_check(args.model)
    return run_camera(args)


if __name__ == "__main__":
    raise SystemExit(main())
