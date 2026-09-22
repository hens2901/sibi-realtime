"""Static gesture reference viewer untuk kelas sulit: K, N, R, U, V, X.

Menampilkan contoh gambar dataset (contact sheet) agar pengguna dapat
membandingkan pose webcam dengan pose dataset. Read-only: tidak mengubah
dataset dan tidak melakukan training.

Output:
- reports/static_reference/<CLASS>_reference.png

Contoh pemakaian:
    python scripts/show_static_reference.py                 # buat semua sheet
    python scripts/show_static_reference.py --class R       # tampilkan R interaktif
    python scripts/show_static_reference.py --per-class 9   # jumlah contoh per kelas
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "Mono_Background"
OUT_DIR = ROOT / "reports" / "static_reference"
HAND_CACHE = ROOT / "data" / "processed" / "handedness_cache.npz"
HARD = ["K", "N", "R", "U", "V", "X"]
WINDOW = "SIBI Static Reference"


def load_handedness() -> dict:
    """path -> 'Left'/'Right' dari cache audit (bila ada)."""
    if not HAND_CACHE.exists():
        return {}
    d = np.load(HAND_CACHE, allow_pickle=True)
    paths = [str(p) for p in d["paths"]]
    handed = [str(h) for h in d["handed"]]
    return dict(zip(paths, handed))


def evenly_spaced(items: list, k: int) -> list:
    if k >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).astype(int)
    return [items[i] for i in sorted(set(idx))]


def pick_images(cdir: Path, per_class: int, hmap: dict) -> list[tuple[Path, str]]:
    """Pilih contoh representatif; utamakan variasi handedness bila ada."""
    imgs = sorted([p for p in cdir.iterdir() if p.is_file()])
    rel = {p: p.relative_to(ROOT).as_posix() for p in imgs}
    by_h: dict[str, list] = defaultdict(list)
    for p in imgs:
        by_h[hmap.get(rel[p], "Unknown")].append(p)

    groups = sorted(by_h.items(), key=lambda kv: -len(kv[1]))  # mayoritas dulu
    majority_h, majority = groups[0]
    minority = [(h, lst) for h, lst in groups[1:] if h != "Unknown" and len(lst) >= 2]

    chosen: list[Path] = []
    if minority:
        h_min, lst_min = minority[0]
        k_min = min(len(lst_min), max(2, per_class // 3))
        chosen += evenly_spaced(lst_min, k_min)
        remain = per_class - len(chosen)
        if remain > 0:
            chosen += evenly_spaced(majority, remain)
    else:
        chosen = evenly_spaced(imgs, per_class)

    chosen = list(dict.fromkeys(chosen))[:per_class]
    return [(p, hmap.get(rel[p], "Unknown")) for p in chosen]


def build_sheet(cls: str, items: list[tuple[Path, str]], tile: int = 260,
                cols: int = 3) -> np.ndarray:
    label_h = 38
    rows = (len(items) + cols - 1) // cols
    title_h = 54
    sheet = np.full((title_h + rows * (tile + label_h), cols * tile, 3), 245,
                    dtype=np.uint8)

    cv2.putText(sheet, f"{cls} reference  ({len(items)} contoh dataset Mono_Background)",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    for i, (path, handed) in enumerate(items):
        r, c = divmod(i, cols)
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (tile, tile), interpolation=cv2.INTER_AREA)
        except Exception:  # noqa: BLE001
            bgr = np.full((tile, tile, 3), 200, dtype=np.uint8)
            cv2.putText(bgr, "read error", (10, tile // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
        y0 = title_h + r * (tile + label_h)
        x0 = c * tile
        sheet[y0:y0 + tile, x0:x0 + tile] = bgr
        bar_y = y0 + tile
        cv2.rectangle(sheet, (x0, bar_y), (x0 + tile, bar_y + label_h), (35, 35, 35), -1)
        txt = f"{cls}  {path.name}  [{handed}]"
        cv2.putText(sheet, txt, (x0 + 6, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + tile - 1, bar_y + label_h),
                      (150, 150, 150), 1)
    return sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Static gesture reference viewer (K,N,R,U,V,X)")
    ap.add_argument("--class", dest="cls", choices=HARD + ["all"], default="all",
                    help="kelas yang ditampilkan interaktif (default: all = buat semua)")
    ap.add_argument("--per-class", type=int, default=9, help="jumlah contoh per kelas")
    ap.add_argument("--tile", type=int, default=260, help="ukuran tile (px)")
    ap.add_argument("--no-save", action="store_true", help="jangan simpan PNG")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hmap = load_handedness()

    sheets: dict[str, np.ndarray] = {}
    for cls in HARD:
        cdir = DATASET / cls
        if not cdir.is_dir():
            print(f"[WARN] folder tidak ada: {cdir}")
            continue
        items = pick_images(cdir, args.per_class, hmap)
        hcount = Counter(h for _, h in items)
        sheets[cls] = build_sheet(cls, items, tile=args.tile)
        if not args.no_save:
            out = OUT_DIR / f"{cls}_reference.png"
            cv2.imwrite(str(out), sheets[cls])
            print(f"{cls}: {len(items)} contoh {dict(hcount)} -> {out.relative_to(ROOT)}")

    print(f"\nSheet tersimpan di: {OUT_DIR.relative_to(ROOT)}")
    print("Jalankan interaktif, mis.: python scripts/show_static_reference.py --class R")
    print("Kontrol: N=next  P=prev  Q/ESC=quit")

    # ---- Interactive (hanya bila --class <X> dipilih; else hanya generate) ----
    if args.cls != "all" and args.cls in sheets:
        order = [args.cls]
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, min(900, args.tile * 3), 900)
            i = 0
            while True:
                cls = order[i]
                cv2.imshow(WINDOW, sheets[cls])
                cv2.setWindowTitle(WINDOW, f"SIBI Reference - {cls} "
                                           f"({i + 1}/{len(order)})")
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("n"):
                    i = (i + 1) % len(order)
                elif key == ord("p"):
                    i = (i - 1) % len(order)
            cv2.destroyAllWindows()
        except Exception as exc:  # noqa: BLE001
            print(f"[INFO] mode interaktif tidak tersedia di environment ini "
                  f"({type(exc).__name__}). PNG sudah disimpan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
