"""Minimal WebRTC test (LOCAL) — tanpa MediaPipe/ANN/callback.

Tujuan: mengisolasi apakah browser <-> WebRTC <-> Streamlit lokal dapat connect
SEBELUM menambahkan pipeline AI.

Jalankan:
    python -m streamlit run test_webrtc_minimal.py

Dengan log WebRTC (ICE/SDP/DTLS):
    set SIBI_WEBRTC_LOG=debug
    python -m streamlit run test_webrtc_minimal.py

Catatan: tanpa rtc_configuration (localhost tidak butuh STUN/TURN). Jika browser
dan server mobil di mesin yang sama, ICE host candidate seharusnya cukup.
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# --- logging opsional (tanpa per-frame spam) ---
if (os.environ.get("SIBI_WEBRTC_LOG", "").lower() in ("debug", "1", "on", "true")):
    logging.basicConfig(level=logging.DEBUG)
    for name in ("streamlit_webrtc", "aiortc", "aioice", "aiortc.rtcicetransport"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    print("[test_webrtc_minimal] WebRTC debug logging ENABLED")

st.set_page_config(page_title="WebRTC Minimal Test", layout="wide")
st.title("Minimal WebRTC Test")
st.caption("Klik START. Jika kamera tampil, stack WebRTC lokal BERHASIL. "
           "Jika tidak, masalah ada di streamlit-webrtc/aiortc/jaringan — "
           "bukan di MediaPipe/ANN.")

ctx = webrtc_streamer(
    key="minimal-webrtc",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    # sengaja TANPA rtc_configuration
)

st.write("state:", {
    "playing": getattr(ctx.state, "playing", None) if ctx else None,
    "signalling": getattr(ctx.state, "signalling", None) if ctx else None,
})
st.info(
    "Langkah jika gagal:\n"
    "1. Coba http://127.0.0.1:8501 (hindari resolusi IPv6 'localhost').\n"
    "2. Cek chrome://webrtc-internals (ICE candidate pair, connectionState).\n"
    "3. Izinkan python.exe inbound di Windows Firewall (Private).\n"
    "4. Coba Edge / Chrome incognito / matikan VPN-extension.\n"
    "5. Jalankan: python scripts/webrtc_ice_selftest.py (candidate server).\n"
)
