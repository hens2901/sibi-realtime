"""TEST B - WebRTC dengan processor passthrough (tanpa MediaPipe/ANN).

Tujuan: memastikan transport WebRTC + lifecycle processor minimal bekerja.
Jika kamera tampil di sini, transport sehat; masalahnya di init berat.

Jalankan:
    python -m streamlit run test_webrtc_passthrough.py
"""

from __future__ import annotations

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


class PassthroughProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.count = 0
        print("[PASSTHROUGH] PROCESSOR CREATED", flush=True)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.count += 1
        if self.count == 1:
            print("[PASSTHROUGH] FIRST FRAME PASSTHROUGH", flush=True)
        return frame


st.set_page_config(page_title="WebRTC Passthrough Test", layout="wide")
st.title("WebRTC Passthrough Test (tanpa MediaPipe/ANN)")
st.caption("Jika kamera tampil di sini, transport WebRTC + processor minimal OK.")

ctx = webrtc_streamer(
    key="passthrough-webrtc",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PassthroughProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
st.write("playing:", getattr(ctx.state, "playing", None) if ctx else None)
