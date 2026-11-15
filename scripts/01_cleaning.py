"""TAHAP 1 - CLEANING

Membuang gambar yang tidak layak dipakai, tanpa mengubah dataset_sumber_asli/:
  rusak         : file tidak bisa dibuka
  duplikat      : file identik (MD5) atau hampir kembar (dHash) dalam satu kelas
  gelap_terang  : rata-rata keabuan hampir hitam / hampir putih (dibaca dari histogram, Acara 10-12)
  blur_parah    : tepi hilang di SEMUA arah (diukur dengan gradien Sobel, Acara 29)
  varian_lama   : file hasil pengayaan lama (nama berakhiran __gerak_horizontal / __gerak_vertikal /
                  __gerak_diagonal / __getar). Pengayaan dibuat ulang di tahap 4 dari foto asli, jadi
                  varian lama dibuang agar tidak dobel dan tidak bocor antara train dan val.
                  Matikan dengan --simpan-varian-lama.

Blur gerak dan blur ringan sengaja dipertahankan: rompi dipakai sambil berjalan, jadi foto
goyang adalah data yang berguna. Ketajaman diukur pada arah paling tajam; blur gerak hanya
menghapus tepi searah gerakan, sedangkan blur parah menghapus tepi di semua arah.

Gambar yang lolos disimpan (sisi terpanjang <= UKURAN_MAKS, tidak di-upscale) ke
hasil/01_cleaning/bersih/<kelas>/<kelas>_<nama>.jpg. Yang ditolak disalin ke ditolak/<alasan>/<kelas>/
supaya bisa dicek manual. Semua keputusan ada di laporan_cleaning.csv.

Jalankan:
    python 01_cleaning.py
    python 01_cleaning.py --kelas orang --blur-parah 50 --mirip 4
"""
import argparse
import hashlib
import re
import shutil

import cv2
import numpy as np

import bkpm
from config import CLASSES, D_CLEAN, SUMBER_DIR, UKURAN_MAKS
from util import (baca_gambar, bersihkan, daftar_gambar, log, nama_aman, simpan_gambar, tabel,
                  tulis_csv)

TAHAP = "01_cleaning"
POLA_VARIAN_LAMA = re.compile(r"__(gerak_horizontal|gerak_vertikal|gerak_diagonal|getar)$", re.I)
KOLOM = ["kelas", "file_sumber", "status", "alasan", "file_hasil", "lebar", "tinggi", "rata_keabuan",
         "ketajaman", "mirip_dengan"]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def dhash(gray, ukuran=8):
    """Sidik jari gambar: bandingkan tiap piksel dengan tetangga kanannya pada citra 9x8."""
    kecil = cv2.resize(gray, (ukuran + 1, ukuran), interpolation=cv2.INTER_AREA)
    bit = (kecil[:, 1:] > kecil[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bit), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def ketajaman(gray):
    """Gradien Sobel (Acara 29) pada arah paling tajam. Makin besar makin tajam."""
    h, w = gray.shape
    s = 640.0 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    terkuat = 0.0
    for sudut in np.arange(0, 180, 22.5):
        t = np.deg2rad(sudut)
        terkuat = max(terkuat, float(np.percentile(np.abs(gx * np.cos(t) + gy * np.sin(t)), 99)))
    return terkuat * 3.0


def simpan_bersih(img, tujuan):
    h, w = img.shape[:2]
    s = min(1.0, UKURAN_MAKS / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    simpan_gambar(tujuan, img, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelas", help="proses satu kelas saja (mis. orang)")
    ap.add_argument("--blur-parah", type=float, default=40.0,
                    help="buang jika ketajaman < nilai ini (default 40; makin besar makin ketat)")
    ap.add_argument("--simpan-varian-lama", action="store_true",
                    help="jangan buang file varian pengayaan lama (__gerak_*, __getar)")
    ap.add_argument("--mirip", type=int, default=5,
                    help="jarak dHash (0-64) <= nilai ini dianggap duplikat (default 5)")
    args = ap.parse_args()

    kelas_list = [args.kelas] if args.kelas else CLASSES
    bersihkan(D_CLEAN)
    laporan, nama_terpakai = [], set()

    for kelas in kelas_list:
        folder = next((d for d in SUMBER_DIR.iterdir() if d.is_dir() and d.name.lower() == kelas), None) \
            if SUMBER_DIR.is_dir() else None
        if folder is None:
            log(TAHAP, f"folder '{kelas}' tidak ada di {SUMBER_DIR}, dilewati")
            continue
        md5_ada, hash_ada = {}, []
        for p in daftar_gambar(folder):
            row = {"kelas": kelas, "file_sumber": p.name, "status": "bersih", "alasan": ""}
            img = baca_gambar(p)
            md5 = md5_file(p)
            alasan, mirip = "", ""
            if not args.simpan_varian_lama and POLA_VARIAN_LAMA.search(p.stem):
                alasan = "varian_lama"
            elif img is None:
                alasan = "rusak"
            else:
                gray = bkpm.rgb2gray(img)
                row.update(lebar=img.shape[1], tinggi=img.shape[0], rata_keabuan=f"{gray.mean():.1f}")
                dh = dhash(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
                tajam = ketajaman(gray)
                row["ketajaman"] = f"{tajam:.1f}"
                if md5 in md5_ada:
                    alasan, mirip = "duplikat", md5_ada[md5]
                elif gray.mean() < 25 or gray.mean() > 235:
                    alasan = "gelap_terang"
                elif tajam < args.blur_parah:
                    alasan = "blur_parah"
                else:
                    for h2, f2 in hash_ada:
                        if hamming(dh, h2) <= args.mirip:
                            alasan, mirip = "duplikat", f2
                            break
            nama = f"{kelas}_{nama_aman(p.stem)}.jpg"
            if (kelas, nama) in nama_terpakai:
                nama = f"{kelas}_{nama_aman(p.stem)}-{md5[:6]}.jpg"
            nama_terpakai.add((kelas, nama))
            row["file_hasil"], row["mirip_dengan"] = nama, mirip
            if alasan:
                row["status"], row["alasan"] = "ditolak", alasan
                tujuan = D_CLEAN / "ditolak" / alasan / kelas / nama
                tujuan.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tujuan)
            else:
                md5_ada[md5] = p.name
                hash_ada.append((dh, p.name))
                simpan_bersih(img, D_CLEAN / "bersih" / kelas / nama)
            laporan.append(row)

    tulis_csv(D_CLEAN / "laporan_cleaning.csv", laporan, KOLOM)
    ring = []
    for k in kelas_list:
        r = [x for x in laporan if x["kelas"] == k]
        if r:
            ring.append({"kelas": k, "total": len(r),
                         **{a: sum(x["alasan"] == a for x in r) for a in ("varian_lama", "duplikat", "blur_parah", "gelap_terang", "rusak")},
                         "LOLOS": sum(x["status"] == "bersih" for x in r)})
    tabel("Ringkasan cleaning", ring, ["kelas", "total", "varian_lama", "duplikat", "blur_parah", "gelap_terang", "rusak", "LOLOS"])
    log(TAHAP, f"selesai -> {D_CLEAN}")


if __name__ == "__main__":
    main()
