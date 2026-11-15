"""TAHAP 5 - EKSTRAKSI FITUR BENTUK (BKPM Acara 22-23)

Setiap objek pada label (gambar asli + varian) dirasterisasi menjadi citra biner, lalu:
  bwlabeln 8-tetangga -> area -> perimeter -> bentuk -> circularity -> kode rantai
    luas_px            jumlah piksel objek (area)
    keliling_batas_px  jumlah piksel batas objek (rumus source code Acara 23)
    keliling_rantai_px panjang kode rantai: tegak/mendatar = 1, diagonal = akar 2 (Acara 22)
    bentuk             keliling^2 / luas (faktor bentuk)
    circularity        4*pi*luas / keliling^2 (1 = lingkaran)
  Tambahan geometri sederhana: luas_rel, rasio tinggi/lebar, extent, solidity, eksentrisitas.

Fitur dipakai sebagai kontrol kualitas: objek dengan luas < 100 px atau luas_rel < 0,3% ditandai
terlalu_kecil dan DIBUANG oleh tahap 6 (gambar tanpa objek tersisa ikut dibuang).

Keluaran: hasil/05_fitur_bentuk/ fitur_bentuk.csv, ringkasan_per_kelas.csv, kode_rantai.csv, kamus_fitur.csv

Jalankan:
    python 05_ekstraksi_fitur.py
"""
import math
from collections import defaultdict

import cv2
import numpy as np

import bkpm
from config import CLASSES, D_AUG, D_FITUR
from util import baca_csv, bersihkan, log, tabel, tulis_csv
from yolo_util import baca_label_seg

TAHAP = "05_fitur_bentuk"
FITUR = ["luas_px", "keliling_batas_px", "keliling_rantai_px", "bentuk", "circularity", "luas_rel",
         "bbox_lebar", "bbox_tinggi", "rasio_tw", "extent", "solidity", "eksentrisitas"]
KOLOM = ["file", "sumber", "varian", "kelas", "id_objek"] + FITUR + ["qc"]
KAMUS = {
    "luas_px": "area: jumlah piksel objek", "keliling_batas_px": "perimeter: jumlah piksel batas (Acara 23)",
    "keliling_rantai_px": "perimeter dari kode rantai (Acara 22)", "bentuk": "keliling_batas^2 / luas",
    "circularity": "4*pi*luas/keliling_batas^2 (1 = lingkaran)", "luas_rel": "luas objek / luas gambar",
    "bbox_lebar": "lebar kotak pembatas", "bbox_tinggi": "tinggi kotak pembatas", "rasio_tw": "tinggi / lebar kotak",
    "extent": "luas / luas kotak pembatas", "solidity": "luas / luas convex hull",
    "eksentrisitas": "0 = bulat, mendekati 1 = memanjang (dari momen citra)",
    "qc": "terlalu_kecil (luas < 100 px atau luas_rel < 0,003) / ok",
}


def fitur_objek(m, h, w):
    a = int(m.sum())
    per = bkpm.perimeter_batas(m)
    rantai = bkpm.kode_rantai(m)
    ys, xs = np.nonzero(m)
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    hull = cv2.contourArea(cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)))
    mu = cv2.moments(m.astype(np.uint8), binaryImage=True)
    d = math.sqrt(max(0.0, ((mu["mu20"] - mu["mu02"]) / 2) ** 2 + mu["mu11"] ** 2))
    l1, l2 = (mu["mu20"] + mu["mu02"]) / 2 + d, (mu["mu20"] + mu["mu02"]) / 2 - d
    ecc = math.sqrt(max(0.0, 1 - l2 / l1)) if l1 > 0 else 0.0
    f = {"luas_px": a, "keliling_batas_px": per, "keliling_rantai_px": round(bkpm.keliling_rantai(rantai), 2),
         "bentuk": round(bkpm.faktor_bentuk(a, per), 3), "circularity": round(bkpm.circularity(a, per), 4),
         "luas_rel": round(a / (h * w), 5), "bbox_lebar": bw, "bbox_tinggi": bh, "rasio_tw": round(bh / bw, 3),
         "extent": round(a / (bw * bh), 4), "solidity": round(a / hull, 4) if hull > 0 else "",
         "eksentrisitas": round(ecc, 4)}
    f["qc"] = "terlalu_kecil" if (a < 100 or a / (h * w) < 0.003) else "ok"
    return f, rantai


def main():
    man = baca_csv(D_AUG / "manifest_pengayaan.csv")
    if not man:
        raise SystemExit("manifest_pengayaan.csv kosong: jalankan 04_pengayaan.py dulu")
    bersihkan(D_FITUR)
    baris, rantai_csv = [], []
    for r in man:
        stem = r["file"].rsplit(".", 1)[0]
        h, w = int(r["tinggi"]), int(r["lebar"])
        for idx, (kid, rel) in enumerate(baca_label_seg(D_AUG / "labels_seg" / f"{stem}.txt")):
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(rel * [w, h]).astype(np.int32)], 1)
            n, lab, st = bkpm.bwlabeln(m, 8)                       # komponen terbesar = objek
            if n == 0:
                continue
            m = lab == (1 + int(np.argmax(st[:, 4])))
            f, rantai = fitur_objek(m, h, w)
            baris.append({"file": r["file"], "sumber": r["sumber"], "varian": r["varian"], "kelas": CLASSES[kid],
                          "id_objek": idx, **f})
            rantai_csv.append({"file": r["file"], "id_objek": idx, "panjang_kode": len(rantai),
                               "kode_rantai": "".join(map(str, rantai))})
    tulis_csv(D_FITUR / "fitur_bentuk.csv", baris, KOLOM)
    tulis_csv(D_FITUR / "kode_rantai.csv", rantai_csv, ["file", "id_objek", "panjang_kode", "kode_rantai"])
    tulis_csv(D_FITUR / "kamus_fitur.csv", [{"fitur": k, "arti": v} for k, v in KAMUS.items()], ["fitur", "arti"])

    ring = []
    for k in CLASSES:
        x = [b for b in baris if b["kelas"] == k and b["qc"] == "ok"]
        if not x:
            continue
        row = {"kelas": k, "n_objek": len(x)}
        for f in ("luas_rel", "rasio_tw", "circularity", "bentuk", "solidity", "eksentrisitas"):
            v = np.array([float(b[f]) for b in x if b[f] != ""])
            row[f + "_rata"], row[f + "_sd"] = round(float(v.mean()), 3), round(float(v.std()), 3)
        ring.append(row)
    kol = list(ring[0].keys()) if ring else ["kelas"]
    tulis_csv(D_FITUR / "ringkasan_per_kelas.csv", ring, kol)
    tabel("Ringkasan fitur bentuk (objek qc=ok)", ring, ["kelas", "n_objek", "luas_rel_rata", "rasio_tw_rata", "circularity_rata",
                                                         "bentuk_rata", "solidity_rata", "eksentrisitas_rata"])
    log(TAHAP, f"{len(baris)} objek, {sum(b['qc'] != 'ok' for b in baris)} ditandai terlalu_kecil -> {D_FITUR}")


if __name__ == "__main__":
    main()
