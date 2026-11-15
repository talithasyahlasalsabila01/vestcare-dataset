"""TAHAP 2 - SEGMENTASI (adaptif per kelas dan per gambar, teknik BKPM)

Input : hasil/01_cleaning/bersih/<kelas>/*.jpg
Output: hasil/02_segmentasi/
          kontras/<kelas>/        citra setelah median+Gaussian+contrast stretching/histeq
          mask/<kelas>/*.png      peta instans: 0 = latar, 1..n = objek ke-n (dibaca tahap 3 dan 4)
          hasil_segmentasi/       objek saja, latar dihitamkan
          overlay/                kontur objek di atas gambar asli
          visualisasi_proses/     subplot proses: asli | kontras | tepi Canny | kandidat terbaik | hasil | overlay
          laporan_segmentasi.csv  metode terpilih, skor, status tiap gambar
          perlu_cek.txt           daftar gambar yang sebaiknya dikoreksi (status perlu_cek / cadangan_roi)

Status:
  ok            skor baik, dipakai langsung
  perlu_cek     skor rendah; tetap dihasilkan tetapi periksa overlay-nya
  cadangan_roi  semua metode gagal, dipakai ROI pusat (kasar) -> sebaiknya diberi ROI manual
  roi_manual    pengguna menandai ROI lewat 02b_koreksi_roi.py (Acara 15)

Jalankan:
    python 02_segmentasi.py
    python 02_segmentasi.py --kelas pohon
    python 02_segmentasi.py --kelas lubang --maks 20      # uji cepat
"""
import argparse

import cv2
import numpy as np

import bkpm
import refine_batas
from config import CLASSES, D_CLEAN, D_SEG, ROOT
from seg_core import segmentasi_gambar
from util import baca_csv, baca_gambar, bersihkan, daftar_gambar, log, simpan_gambar, tabel, tulis_csv

TAHAP = "02_segmentasi"
ROI_CSV = ROOT / "roi_manual.csv"
KOLOM = ["kelas", "file", "n_objek", "metode", "skor", "status", "luas_rel", "kontras", "fit", "sep", "luas", "pos",
         "bentuk", "sentuh"]
WARNA = [(0, 255, 0), (0, 200, 255), (255, 120, 0), (255, 0, 200), (0, 0, 255), (200, 255, 0),
         (255, 255, 0), (120, 0, 255)]


def muat_roi():
    """roi_manual.csv: kolom file,x,y,w,h (piksel pada gambar bersih)."""
    return {r["file"]: (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])) for r in baca_csv(ROI_CSV)}


def gambar_overlay(img, lab, teks=None):
    out = img.copy()
    n = int(lab.max())
    for i in range(1, n + 1):
        m = (lab == i).astype(np.uint8)
        w = WARNA[(i - 1) % len(WARNA)]
        out[m > 0] = (0.6 * out[m > 0] + 0.4 * np.array(w)).astype(np.uint8)
        k, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, k, -1, w, 2)
    if teks:
        cv2.putText(out, teks, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, teks, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def subplot(tiles, tinggi=220):
    """Susun beberapa citra berdampingan (padanan subplot(1,n,k) di BKPM)."""
    out = []
    for t in tiles:
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        s = tinggi / t.shape[0]
        out.append(cv2.resize(t, (max(1, int(t.shape[1] * s)), tinggi), interpolation=cv2.INTER_AREA))
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelas", help="proses satu kelas saja")
    ap.add_argument("--haluskan", action="store_true", help="rapikan batas mask gambar ber-ROI manual (region growing + tepi Canny + morfologi, refine_batas.py)")
    ap.add_argument("--maks", type=int, default=0, help="batasi jumlah gambar per kelas (uji cepat)")
    args = ap.parse_args()

    kelas_list = [args.kelas] if args.kelas else CLASSES
    if len(kelas_list) == len(CLASSES):
        bersihkan(D_SEG)
    roi = muat_roi()
    laporan, cek = [], []

    for kelas in kelas_list:
        files = daftar_gambar(D_CLEAN / "bersih" / kelas)
        if args.maks:
            files = files[::max(1, len(files) // args.maks)][:args.maks]
        if not files:
            log(TAHAP, f"tidak ada gambar bersih untuk '{kelas}' (jalankan 01_cleaning.py)")
            continue
        for sub in ("kontras", "mask", "hasil_segmentasi", "overlay", "visualisasi_proses"):
            (D_SEG / sub / kelas).mkdir(parents=True, exist_ok=True)
        log(TAHAP, f"{kelas}: {len(files)} gambar")
        for i, p in enumerate(files, 1):
            img = baca_gambar(p)
            r = segmentasi_gambar(img, kelas, roi.get(p.name))
            lab, P = r["label"], r["P"]
            if args.haluskan and p.name in roi:
                lab = refine_batas.refine(img, lab, roi[p.name], P)
            stem = p.stem

            simpan_gambar(D_SEG / "kontras" / kelas / f"{stem}.jpg", P["kontras"])
            cv2.imencode(".png", lab)[1].tofile(str(D_SEG / "mask" / kelas / f"{stem}.png"))
            obj = img.copy()
            obj[lab == 0] = 0
            simpan_gambar(D_SEG / "hasil_segmentasi" / kelas / f"{stem}.jpg", obj)
            teks = f"{r['metode']} {r['skor']:.2f} [{r['status']}]"
            ov = gambar_overlay(img, lab, teks)
            simpan_gambar(D_SEG / "overlay" / kelas / f"{stem}.jpg", ov, 90)
            vis = subplot([img, P["kontras"], (P["tepi"] * 255).astype(np.uint8),
                           (lab > 0).astype(np.uint8) * 255, obj, ov])
            simpan_gambar(D_SEG / "visualisasi_proses" / kelas / f"{stem}.jpg", vis, 85)

            rinci = r["rinci"]
            laporan.append({
                "kelas": kelas, "file": p.name, "n_objek": r["n"], "metode": r["metode"],
                "skor": f"{r['skor']:.3f}", "status": r["status"],
                "luas_rel": f"{(lab > 0).mean():.3f}", "kontras": P["metode_kontras"],
                **{k: f"{rinci.get(k, 0):.2f}" for k in ("fit", "sep", "luas", "pos", "bentuk", "sentuh")}})
            if r["status"] in ("perlu_cek", "cadangan_roi"):
                cek.append(f"{kelas}/{p.name}\t{r['status']}\t{r['metode']}\t{r['skor']:.2f}")
            if i % 50 == 0:
                log(TAHAP, f"  {kelas}: {i}/{len(files)}")

    # bila hanya sebagian kelas diproses, pertahankan baris kelas lain di laporan
    lama = [r for r in baca_csv(D_SEG / "laporan_segmentasi.csv") if r["kelas"] not in kelas_list]
    semua = lama + laporan
    tulis_csv(D_SEG / "laporan_segmentasi.csv", semua, KOLOM)
    cek = [f"{r['kelas']}/{r['file']}\t{r['status']}\t{r['metode']}\t{r['skor']}" for r in semua
           if r["status"] in ("perlu_cek", "cadangan_roi")]
    (D_SEG / "perlu_cek.txt").write_text("\n".join(cek), encoding="utf-8")

    ring = []
    for k in kelas_list:
        r = [x for x in laporan if x["kelas"] == k]
        if r:
            ring.append({"kelas": k, "gambar": len(r), "ok": sum(x["status"] == "ok" for x in r),
                         "perlu_cek": sum(x["status"] == "perlu_cek" for x in r),
                         "cadangan_roi": sum(x["status"] == "cadangan_roi" for x in r),
                         "roi_manual": sum(x["status"] == "roi_manual" for x in r),
                         "rata_skor": f"{np.mean([float(x['skor']) for x in r]):.2f}"})
    tabel("Ringkasan segmentasi", ring, ["kelas", "gambar", "ok", "perlu_cek", "cadangan_roi", "roi_manual", "rata_skor"])
    metode = {}
    for x in laporan:
        metode.setdefault(x["kelas"], {}).setdefault(x["metode"], 0)
        metode[x["kelas"]][x["metode"]] += 1
    for k, d in metode.items():
        print(f"  {k:7s} metode terpilih:", ", ".join(f"{m}={n}" for m, n in sorted(d.items(), key=lambda t: -t[1])))
    log(TAHAP, f"selesai -> {D_SEG}")
    if cek:
        log(TAHAP, f"{len(cek)} gambar perlu dicek: lihat perlu_cek.txt (opsional: python 02b_koreksi_roi.py)")


if __name__ == "__main__":
    main()
