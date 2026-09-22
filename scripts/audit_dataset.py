"""Audit dataset SIBI Mono_Background tanpa mengubah file asli.

Menghasilkan:
- reports/dataset_summary.csv
- reports/DATASET_REPORT.md
"""

from __future__ import annotations

import csv
import hashlib
import string
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "raw" / "Mono_Background"
REPORTS_DIR = ROOT / "reports"
CSV_PATH = REPORTS_DIR / "dataset_summary.csv"
MD_PATH = REPORTS_DIR / "DATASET_REPORT.md"

ALL_LETTERS = list(string.ascii_uppercase)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def audit() -> dict:
    classes = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)
    class_names = [d.name for d in classes]

    per_class: dict[str, dict] = {}
    global_ext = Counter()
    global_modes = Counter()
    global_sizes = Counter()
    corrupt_files: list[str] = []
    total_bytes = 0
    total_images = 0

    for cdir in classes:
        images = sorted([f for f in cdir.iterdir() if f.is_file()])
        count = 0
        ext = Counter()
        modes = Counter()
        sizes = Counter()
        widths: list[int] = []
        heights: list[int] = []
        corrupt: list[str] = []

        for img_path in images:
            total_images += 1
            ext[img_path.suffix.lower()] += 1
            global_ext[img_path.suffix.lower()] += 1
            try:
                size = img_path.stat().st_size
                total_bytes += size
            except OSError:
                size = 0
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
                    mode = im.mode
                    fmt = im.format
                widths.append(w)
                heights.append(h)
                modes[f"{fmt}/{mode}"] += 1
                global_modes[f"{fmt}/{mode}"] += 1
                sizes[f"{w}x{h}"] += 1
                global_sizes[f"{w}x{h}"] += 1
                count += 1
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt.append(f"{img_path.name} ({type(exc).__name__})")
                corrupt_files.append(str(img_path.relative_to(ROOT)))

        per_class[cdir.name] = {
            "count": count,
            "total_files": len(images),
            "ext": dict(ext),
            "modes": dict(modes),
            "sizes": dict(sizes),
            "min_w": min(widths) if widths else 0,
            "max_w": max(widths) if widths else 0,
            "min_h": min(heights) if heights else 0,
            "max_h": max(heights) if heights else 0,
            "corrupt": corrupt,
        }

    present = set(class_names)
    missing = [ch for ch in ALL_LETTERS if ch not in present]
    extra = [name for name in class_names if name not in ALL_LETTERS]

    counts = [per_class[c]["count"] for c in class_names]
    total = sum(counts)
    n = len(counts)
    mean = total / n if n else 0.0
    variance = sum((c - mean) ** 2 for c in counts) / n if n else 0.0
    std = variance ** 0.5
    cmin = min(counts) if counts else 0
    cmax = max(counts) if counts else 0
    imbalance_ratio = (cmax / cmin) if cmin else float("inf")

    return {
        "class_names": class_names,
        "per_class": per_class,
        "present": present,
        "missing": missing,
        "extra": extra,
        "total": total,
        "mean": mean,
        "std": std,
        "cmin": cmin,
        "cmax": cmax,
        "imbalance_ratio": imbalance_ratio,
        "global_ext": dict(global_ext),
        "global_modes": dict(global_modes),
        "global_sizes": dict(global_sizes),
        "corrupt_files": corrupt_files,
        "total_bytes": total_bytes,
    }


def write_csv(res: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "class",
                "jumlah_gambar",
                "format",
                "resolusi_unik",
                "min_width",
                "max_width",
                "min_height",
                "max_height",
                "file_corrupt",
                "persentase",
            ]
        )
        total = res["total"] or 1
        for name in res["class_names"]:
            info = res["per_class"][name]
            writer.writerow(
                [
                    name,
                    info["count"],
                    ";".join(info["ext"].keys()),
                    ";".join(info["sizes"].keys()),
                    info["min_w"],
                    info["max_w"],
                    info["min_h"],
                    info["max_h"],
                    len(info["corrupt"]),
                    f"{info['count'] / total * 100:.2f}%",
                ]
            )
        writer.writerow(
            [
                "TOTAL",
                res["total"],
                ";".join(res["global_ext"].keys()),
                "",
                "",
                "",
                "",
                "",
                len(res["corrupt_files"]),
                "100.00%",
            ]
        )


def write_md(res: dict) -> None:
    lines: list[str] = []
    a = lines.append
    a("# DATASET REPORT - SIBI (Mono_Background)")
    a("")
    a(f"- Dibuat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- Sumber: `{DATASET_DIR.relative_to(ROOT).as_posix()}`")
    a("- Mode: audit read-only (tidak ada file asli yang diubah)")
    a("")
    a("## 1. Ringkasan")
    a("")
    a(f"- Jumlah kelas (folder): **{len(res['class_names'])}**")
    a(f"- Total gambar valid: **{res['total']}**")
    a(f"- Total ukuran: **{human_bytes(res['total_bytes'])}**")
    a(f"- Rata-rata gambar/kelas: **{res['mean']:.1f}**")
    a(f"- Kelas minimum: **{res['cmin']}** gambar")
    a(f"- Kelas maksimum: **{res['cmax']}** gambar")
    a(f"- Rasio imbalance (max/min): **{res['imbalance_ratio']:.2f}x**")
    a(f"- Standar deviasi jumlah gambar: **{res['std']:.2f}**")
    a(f"- File corrupt: **{len(res['corrupt_files'])}**")
    a("")
    a("## 2. Nama Seluruh Kelas")
    a("")
    a(", ".join(res["class_names"]))
    a("")
    a("## 3. Huruf A-Z yang Tidak Tersedia")
    a("")
    if res["missing"]:
        for ch in res["missing"]:
            if ch in ("J", "Z"):
                a(f"- **{ch}** - missing/dynamic class (gestur dinamis, tidak tersedia di dataset statis)")
            else:
                a(f"- **{ch}** - missing class")
    else:
        a("Tidak ada huruf yang hilang; semua A-Z tersedia.")
    if res["extra"]:
        a("")
        a(f"Kelas di luar A-Z: {', '.join(res['extra'])}")
    a("")
    a("> Catatan: J dan Z adalah kelas dinamis (gerakan). Sesuai instruksi, TIDAK dibuat data sintetis sebagai penggantinya.")
    a("")
    a("## 4. Tabel Jumlah Gambar per Kelas")
    a("")
    a("| Kelas | Jumlah | Persentase | Format | Resolusi Unik | Corrupt |")
    a("|:-----:|-------:|-----------:|:------:|:-------------:|--------:|")
    total = res["total"] or 1
    for name in res["class_names"]:
        info = res["per_class"][name]
        fmts = ", ".join(info["ext"].keys())
        sizes = ", ".join(info["sizes"].keys())
        a(
            f"| {name} | {info['count']} | {info['count'] / total * 100:.2f}% | "
            f"{fmts} | {sizes} | {len(info['corrupt'])} |"
        )
    a(
        f"| **TOTAL** | **{res['total']}** | **100.00%** | "
        f"{', '.join(res['global_ext'].keys())} | - | {len(res['corrupt_files'])} |"
    )
    a("")
    a("## 5. Format File")
    a("")
    a("| Format | Jumlah |")
    a("|:------:|-------:|")
    for ext, cnt in sorted(res["global_ext"].items(), key=lambda x: -x[1]):
        a(f"| {ext} | {cnt} |")
    a("")
    a("## 6. Resolusi Gambar")
    a("")
    a("| Resolusi | Jumlah |")
    a("|:--------:|-------:|")
    for size, cnt in sorted(res["global_sizes"].items(), key=lambda x: -x[1]):
        a(f"| {size} | {cnt} |")
    a("")
    a("| Mode (format/PIL mode) | Jumlah |")
    a("|:----------------------:|-------:|")
    for mode, cnt in sorted(res["global_modes"].items(), key=lambda x: -x[1]):
        a(f"| {mode} | {cnt} |")
    a("")
    a("## 7. File Corrupt")
    a("")
    if res["corrupt_files"]:
        for f in res["corrupt_files"]:
            a(f"- `{f}`")
    else:
        a("Tidak ditemukan file corrupt. Semua gambar berhasil dibuka dan diverifikasi.")
    a("")
    a("## 8. Analisis Imbalance")
    a("")
    if res["imbalance_ratio"] == float("inf"):
        a("- Tidak dapat dihitung (ada kelas dengan 0 gambar).")
    elif res["cmin"] == res["cmax"]:
        a(f"- Dataset **seimbang sempurna**: setiap kelas memiliki {res['cmin']} gambar.")
        a(f"- Rasio imbalance 1.00x, standar deviasi {res['std']:.2f}.")
    else:
        a(f"- Rasio imbalance (max/min): **{res['imbalance_ratio']:.2f}x**")
        a(f"- Standar deviasi: {res['std']:.2f}")
        a("- Kelas dengan jumlah terbanyak / tersedikit:")
        a(f"  - Terbanyak: {res['cmax']} gambar")
        a(f"  - Tersedikit: {res['cmin']} gambar")
    a("")
    a("## 9. Kesimpulan")
    a("")
    a(
        f"Dataset terdiri dari {len(res['class_names'])} kelas dengan total {res['total']} gambar "
        f"berformat {', '.join(res['global_ext'].keys())}."
    )
    a(
        f"Huruf yang tidak tersedia: {', '.join(res['missing']) if res['missing'] else 'tidak ada'} "
        "(J dan Z merupakan kelas dinamis)."
    )
    a(
        f"Status imbalance: {'seimbang' if res['cmin'] == res['cmax'] else 'tidak seimbang'} "
        f"({res['imbalance_ratio']:.2f}x). File corrupt: {len(res['corrupt_files'])}."
    )
    a("")
    a("> Tidak ada training dan tidak ada pembuatan data sintetis pada audit ini.")
    a("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    res = audit()
    write_csv(res)
    write_md(res)
    print(f"Classes         : {len(res['class_names'])}")
    print(f"Total images    : {res['total']}")
    print(f"Missing letters : {res['missing']}")
    print(f"Corrupt files   : {len(res['corrupt_files'])}")
    print(f"Imbalance ratio : {res['imbalance_ratio']:.2f}x")
    print(f"CSV             : {CSV_PATH.relative_to(ROOT)}")
    print(f"MD              : {MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
