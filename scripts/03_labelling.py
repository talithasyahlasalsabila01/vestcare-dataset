"""TAHAP 3 - LABELLING OTOMATIS

Peta instans hasil segmentasi (tahap 2) -> label YOLO:
  labels_seg/<nama>.txt   class_id x1 y1 x2 y2 ...      (polygon, untuk yolo26n-seg)
  labels_det/<nama>.txt   class_id cx cy w h            (kotak,   untuk yolo26n)
Semua koordinat dinormalisasi 0-1. Citra latih = gambar bersih asli (bukan citra kontras).

Hanya gambar berstatus ok / roi_manual yang dilabel, karena label dari mask yang salah akan merusak
pelatihan. Gambar perlu_cek dapat diperbaiki lewat 02b_koreksi_roi.py lalu 02 diulang, atau dipaksa
ikut dengan --sertakan-perlu-cek.

Jalankan:
    python 03_labelling.py
    python 03_labelling.py --sertakan-perlu-cek
"""
import argparse

import cv2
import numpy as np

from config import CLASSES, D_CLEAN, D_LABEL, D_SEG
from seg_profil import PROFIL
from util import baca_csv, baca_gambar, bersihkan, log, simpan_gambar, tabel, tulis_csv
from yolo_util import gambar_berlabel, instans_ke_polygon, tulis_label

TAHAP = "03_labelling"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sertakan-perlu-cek", action="store_true", help="ikutkan gambar berstatus perlu_cek/cadangan_roi")
    args = ap.parse_args()

    laporan = baca_csv(D_SEG / "laporan_segmentasi.csv")
    if not laporan:
        raise SystemExit("laporan_segmentasi.csv tidak ada: jalankan 02_segmentasi.py dulu")
    bersihkan(D_LABEL)
    layak = {"ok", "roi_manual"} | ({"perlu_cek", "cadangan_roi"} if args.sertakan_perlu_cek else set())
    hasil, ring = [], {k: {"gambar": 0, "dilewati": 0, "objek": 0} for k in CLASSES}

    for r in laporan:
        kelas, nama = r["kelas"], r["file"]
        stem = nama.rsplit(".", 1)[0]
        # status dihitung ulang dari skor dengan ambang_cek di seg_profil.py, sehingga ambang bisa disetel
        # tanpa mengulang segmentasi (gambar roi_manual selalu dianggap layak)
        if r["status"] == "roi_manual":
            status = "roi_manual"
        elif r["status"] == "cadangan_roi" or float(r["skor"]) < PROFIL[kelas]["ambang_cek"]:
            status = "perlu_cek" if r["status"] != "cadangan_roi" else "cadangan_roi"
        else:
            status = "ok"
        r["status"] = status
        if status not in layak:
            ring[kelas]["dilewati"] += 1
            continue
        lab = cv2.imdecode(np.fromfile(str(D_SEG / "mask" / kelas / f"{stem}.png"), "uint8"),
                           cv2.IMREAD_UNCHANGED)
        img = baca_gambar(D_CLEAN / "bersih" / kelas / nama)
        h, w = img.shape[:2]
        polys = instans_ke_polygon(lab)
        if not polys:
            ring[kelas]["dilewati"] += 1
            continue
        tulis_label(D_LABEL / "labels_seg" / f"{stem}.txt", D_LABEL / "labels_det" / f"{stem}.txt",
                    CLASSES.index(kelas), polys, w, h)
        simpan_gambar(D_LABEL / "images" / nama, img, 95)
        simpan_gambar(D_LABEL / "gambar_berlabel" / nama, gambar_berlabel(img, polys, kelas), 90)
        ring[kelas]["gambar"] += 1
        ring[kelas]["objek"] += len(polys)
        hasil.append({"file": nama, "kelas": kelas, "n_objek": len(polys), "status_segmentasi": r["status"],
                      "skor": r["skor"], "lebar": w, "tinggi": h})

    tulis_csv(D_LABEL / "laporan_labelling.csv", hasil, ["file", "kelas", "n_objek", "status_segmentasi", "skor", "lebar", "tinggi"])
    tabel("Ringkasan labelling", [{"kelas": k, **v} for k, v in ring.items()], ["kelas", "gambar", "objek", "dilewati"])
    log(TAHAP, f"selesai -> {D_LABEL}")


if __name__ == "__main__":
    main()
