"""Server-side ICE self-test (aiortc/aioice) untuk isolasi masalah WebRTC lokal.

Membuat RTCPeerConnection + offer video, menunggu ICE gathering, lalu
melaporkan kandidat (host/srflx/relay). Berguna untuk memastikan sisi server
(aiortc/aioice/av) sehat tanpa browser.

Jalankan:
    python scripts/webrtc_ice_selftest.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main() -> int:
    from aiortc import RTCPeerConnection
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    await pc.setLocalDescription(await pc.createOffer())
    for _ in range(80):
        if pc.iceGatheringState == "complete":
            break
        await asyncio.sleep(0.1)

    sdp = pc.localDescription.sdp
    cands = [ln for ln in sdp.splitlines() if ln.startswith("a=candidate")]
    ips, types = [], []
    for c in cands:
        parts = c.split()
        try:
            ips.append(parts[4])
            types.append(parts[7])
        except IndexError:
            pass
    await pc.close()

    print(f"iceGatheringState : {pc.iceGatheringState}")
    print(f"candidate count   : {len(cands)}")
    for c in cands:
        print("  ", c)
    has_loopback = any(ip.startswith("127.") for ip in ips)
    print(f"has loopback cand : {has_loopback}")
    print(f"candidate types   : {sorted(set(types))}")

    ok = pc.iceGatheringState == "complete" and len(cands) > 0
    print("SERVER ICE:", "OK" if ok else "GAGAL")
    print("Catatan: server OK != browser terhubung. Jika hanya ada host LAN "
          "(mis. 192.168.x.x) tanpa loopback, Windows Firewall dapat memblokir "
          "traffic browser->python (inbound UDP). Uji http://127.0.0.1:8501 atau "
          "izinkan python.exe inbound.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
