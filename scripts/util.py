"""util.py - fungsi bantu umum (file, CSV, log). Tidak berisi pengolahan citra."""
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, IMG_EXT


def nama_aman(teks):
    """Ubah nama file menjadi aman (huruf, angka, - dan _)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(teks)).strip("-") or "x"


def kunci_alami(nama):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nama)]


def daftar_gambar(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT),
                  key=lambda p: kunci_alami(p.name))


def bersihkan(folder):
    """Hapus isi folder hasil sebuah tahap lalu buat ulang (hanya folder hasil, bukan sumber)."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def baca_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tulis_csv(path, baris, kolom):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        for b in baris:
            w.writerow(b)


def baca_gambar(path):
    """cv2.imread yang aman untuk path berisi spasi/unicode. Mengembalikan None bila rusak."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def simpan_gambar(path, img, kualitas=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    param = [cv2.IMWRITE_JPEG_QUALITY, kualitas] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, param)
    if not ok:
        raise IOError(f"gagal menyimpan {path}")
    buf.tofile(str(path))


def kelas_dari_nama(nama):
    """'pohon_abc.jpg' -> 'pohon' (prefiks nama file = nama kelas)."""
    awal = Path(nama).stem.split("_")[0]
    return awal if awal in CLASSES else None


def log(tahap, pesan):
    print(f"[{tahap}] {pesan}", flush=True)


def tabel(judul, baris, kolom):
    """Cetak tabel sederhana ke layar."""
    print(f"\n{judul}")
    lebar = [max(len(str(k)), *(len(str(b.get(k, ''))) for b in baris)) if baris else len(str(k))
             for k in kolom]
    print("  ".join(str(k).ljust(w) for k, w in zip(kolom, lebar)))
    for b in baris:
        print("  ".join(str(b.get(k, "")).ljust(w) for k, w in zip(kolom, lebar)))


def keluar(pesan):
    print(pesan)
    sys.exit(1)
