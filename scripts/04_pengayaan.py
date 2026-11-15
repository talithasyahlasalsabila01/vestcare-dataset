"""TAHAP 4 - PENGAYAAN (AUGMENTASI) DATA

Varian dibuat hanya dengan operasi BKPM:
  Fotometrik (label tetap)              : cerah / gelap (F+k, Acara 11), kontras (imadjust, Acara 11),
                                          histeq (Acara 12), gaussian (imgaussfilt, Acara 28),
                                          gerak (konvolusi kernel garis = blur gerak kamera, Acara 13-14)
  Geometrik (label dihitung ulang dari   : flip horizontal (Acara 8), crop (Acara 15)
  mask yang ikut ditransformasi)
Blur gerak dipertahankan karena rompi dipakai sambil berjalan.

Jumlah varian per gambar mengikuti --target: kelas dengan foto sedikit mendapat varian lebih banyak
(maksimum --maks-varian), supaya jumlah gambar antar kelas lebih seimbang. Foto asli selalu ikut.
Nama varian: <nama>__<varian>.jpg. Bagian sebelum "__" dipakai tahap 6 sebagai kunci kelompok.

Jalankan:
    python 04_pengayaan.py
    python 04_pengayaan.py --target 500 --maks-varian 6
"""
import argparse
import math
import random
import zlib
from collections import defaultdict

import cv2
import numpy as np

import bkpm
from config import CLASSES, D_AUG, D_LABEL, D_SEG
from util import baca_csv, baca_gambar, bersihkan, log, simpan_gambar, tabel, tulis_csv
from yolo_util import instans_ke_polygon, tulis_label

TAHAP = "04_pengayaan"
POOL = ["flip_h", "cerah", "gerak", "crop", "gelap", "histeq", "gaussian", "kontras"]


def rng_untuk(nama, varian):
    return random.Random(zlib.crc32(f"{nama}|{varian}".encode()))


def varian_foto(img, v, rng):
    if v == "cerah":
        return bkpm.brightness(img, rng.randint(35, 60))
    if v == "gelap":
        return bkpm.brightness(img, -rng.randint(35, 60))
    if v == "kontras":
        return bkpm.enhance_warna(img, "imadjust")
    if v == "histeq":
        return bkpm.enhance_warna(img, "histeq")
    if v == "gaussian":
        return bkpm.imgaussfilt(img, rng.uniform(1.2, 2.2))
    if v == "gerak":
        arah = rng.choice([(-5, 5), (85, 95), (30, 60)])
        k = bkpm.kernel_gerak(rng.randint(9, 17), rng.uniform(*arah))
        return bkpm.konvolusi(img, k)
    raise ValueError(v)


def varian_geometri(img, lab, v, rng):
    h, w = lab.shape
    if v == "flip_h":
        return np.ascontiguousarray(img[:, ::-1]), np.ascontiguousarray(lab[:, ::-1]), 1.0
    # crop: ambil 78-92% gambar dari posisi acak lalu kembalikan ke ukuran semula
    f = rng.uniform(0.78, 0.92)
    cw, ch = int(w * f), int(h * f)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    ci, cl = img[y0:y0 + ch, x0:x0 + cw], lab[y0:y0 + ch, x0:x0 + cw]
    return (cv2.resize(ci, (w, h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(cl, (w, h), interpolation=cv2.INTER_NEAREST), f * f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=450, help="target jumlah gambar per kelas (asli + varian)")
    ap.add_argument("--maks-varian", type=int, default=7, help="maksimum varian per foto asli")
    args = ap.parse_args()

    lap = baca_csv(D_LABEL / "laporan_labelling.csv")
    if not lap:
        raise SystemExit("laporan_labelling.csv kosong: jalankan 03_labelling.py dulu")
    bersihkan(D_AUG)
    per_kelas = defaultdict(list)
    for r in lap:
        per_kelas[r["kelas"]].append(r)

    manifest, ring = [], []
    for kelas in CLASSES:
        daftar = per_kelas.get(kelas, [])
        if not daftar:
            continue
        n_var = min(args.maks_varian, max(0, math.ceil(args.target / len(daftar)) - 1))
        kid = CLASSES.index(kelas)
        for idx, r in enumerate(daftar):
            nama = r["file"]
            stem = nama.rsplit(".", 1)[0]
            img = baca_gambar(D_LABEL / "images" / nama)
            lab = cv2.imdecode(np.fromfile(str(D_SEG / "mask" / kelas / f"{stem}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            simpan_gambar(D_AUG / "images" / nama, img, 95)
            for sub in ("labels_seg", "labels_det"):
                (D_AUG / sub).mkdir(parents=True, exist_ok=True)
                (D_AUG / sub / f"{stem}.txt").write_text((D_LABEL / sub / f"{stem}.txt").read_text())
            manifest.append({"file": nama, "sumber": stem, "kelas": kelas, "varian": "asli", "lebar": w, "tinggi": h})
            urut = [POOL[(idx + j) % len(POOL)] for j in range(len(POOL))][:n_var]
            for v in urut:
                rng = rng_untuk(stem, v)
                if v in ("flip_h", "crop"):
                    im2, lab2, skala = varian_geometri(img, lab, v, rng)
                    polys = instans_ke_polygon(lab2)
                    if v == "crop":      # buang objek yang terpotong sehingga tersisa < 40% luas semula
                        asli = {i: int((lab == i).sum()) for i in range(1, int(lab.max()) + 1)}
                        polys = [(i, p) for i, p in polys if (lab2 == i).sum() * skala >= 0.4 * asli.get(i, 1)]
                    if not polys:
                        continue
                else:
                    im2 = varian_foto(img, v, rng)
                    polys = instans_ke_polygon(lab)
                nv = f"{stem}__{v}"
                simpan_gambar(D_AUG / "images" / f"{nv}.jpg", im2, 95)
                tulis_label(D_AUG / "labels_seg" / f"{nv}.txt", D_AUG / "labels_det" / f"{nv}.txt", kid, polys, w, h)
                manifest.append({"file": f"{nv}.jpg", "sumber": stem, "kelas": kelas, "varian": v, "lebar": w, "tinggi": h})
        n_all = sum(1 for m in manifest if m["kelas"] == kelas)
        ring.append({"kelas": kelas, "foto_asli": len(daftar), "varian_per_foto": n_var, "total_gambar": n_all})

    tulis_csv(D_AUG / "manifest_pengayaan.csv", manifest, ["file", "sumber", "kelas", "varian", "lebar", "tinggi"])
    tabel("Ringkasan pengayaan", ring, ["kelas", "foto_asli", "varian_per_foto", "total_gambar"])
    log(TAHAP, f"selesai -> {D_AUG}")


if __name__ == "__main__":
    main()
