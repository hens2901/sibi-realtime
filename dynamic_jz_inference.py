"""Helper inference untuk GRU dinamis J/Z.

CATATAN: ini BUKAN integrasi ke aplikasi (realtime.py / app_streamlit.py
tidak disentuh). Modul ini menyediakan kontrak input yang ketat (24, 63),
karena Keras GRU secara teknis menerima panjang waktu bervariasi.

Pemakaian:
    from dynamic_jz_inference import DynamicJZPredictor
    pred = DynamicJZPredictor()
    probs = pred.predict(sequence_24x63)     # -> (2,) [p(J), p(Z)]
    label, conf = pred.predict_label(sequence_24x63)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

SEQ_LEN = 24
FEATURE_COUNT = 63
CLASSES = ["J", "Z"]

_MODEL_PATH = MODELS / "dynamic_jz_gru.keras"
_SCALER_PATH = MODELS / "dynamic_jz_scaler.joblib"
_LABELS_PATH = MODELS / "dynamic_jz_labels.json"


class DynamicJZPredictor:
    """Wrapper inference dengan kontrak shape (24, 63) yang ketat."""

    def __init__(self, model_path: Path = _MODEL_PATH,
                 scaler_path: Path = _SCALER_PATH,
                 labels_path: Path = _LABELS_PATH) -> None:
        for p in (model_path, scaler_path, labels_path):
            if not p.exists():
                raise FileNotFoundError(f"Artefak tidak ditemukan: {p}")
        import keras  # lazy: impor TF berat
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.classes = self.labels.get("classes", CLASSES)

    @staticmethod
    def _prepare(sequence) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim == 2 and seq.shape == (SEQ_LEN, FEATURE_COUNT):
            seq = seq[None, ...]  # dukung satu sequence (24,63)
        if seq.ndim != 3 or seq.shape[1:] != (SEQ_LEN, FEATURE_COUNT):
            raise ValueError(
                f"Input sequence harus ber-shape ({SEQ_LEN}, {FEATURE_COUNT}) "
                f"atau (batch, {SEQ_LEN}, {FEATURE_COUNT}), ditemukan {seq.shape}."
            )
        if not np.all(np.isfinite(seq)):
            raise ValueError("Input sequence mengandung NaN/inf.")
        return seq

    def predict(self, sequence) -> np.ndarray:
        """Kembalikan probabilitas (batch, 2) untuk sequence (batch,24,63)."""
        seq = self._prepare(sequence)
        flat = seq.reshape(-1, FEATURE_COUNT)
        scaled = self.scaler.transform(flat).reshape(seq.shape).astype(np.float32)
        proba = np.asarray(self.model.predict(scaled, verbose=0), dtype=np.float64)
        if not np.all(np.isfinite(proba)):
            raise ValueError("Probabilitas model tidak finite.")
        return proba

    def predict_label(self, sequence) -> tuple[str, float]:
        proba = self.predict(sequence)
        idx = int(np.argmax(proba[0]))
        return self.classes[idx], float(proba[0, idx])
