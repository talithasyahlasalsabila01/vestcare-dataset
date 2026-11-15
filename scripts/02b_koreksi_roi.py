"""TAHAP 2b (opsional) - KOREKSI ROI MANUAL untuk gambar yang segmentasinya perlu dicek (BKPM Acara 15)

Menampilkan gambar berstatus perlu_cek / cadangan_roi satu per satu, per kelas. Tandai kotak ROI yang
mengelilingi SEMUA objek kelasnya (mis. semua orang), lalu tekan ENTER/SPASI.
Kalau tidak ingin menandai gambar itu, tekan ESC (atau ENTER tanpa kotak): terminal akan bertanya
Enter = lewati, q = berhenti.
ROI disimpan di <proyek>/roi_manual.csv setiap selesai satu gambar. Setelah itu jalankan lagi
02_segmentasi.py: pada gambar ber-ROI, area di luar ROI dipakai sebagai referensi latar dan hasil
dibatasi di dalam ROI.

Jalankan:
    python 02b_koreksi_roi.py
    python 02b_koreksi_roi.py --kelas pohon,motor,mobil
    python 02b_koreksi_roi.py --kelas orang,lubang --keluaran roi_manual_teman.csv
    python 02b_koreksi_roi.py --kelas lubang --maks 30
"""
import argparse

import cv2

from config import CLASSES, D_CLEAN, D_SEG, ROOT
from util import baca_csv, baca_gambar, tulis_csv

ROI_CSV = ROOT / "roi_manual.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelas", help="satu atau beberapa kelas dipisah koma, mis. pohon,motor,mobil")
    ap.add_argument("--maks", type=int, default=0, help="batasi jumlah gambar (per kelas terpilih, skor terendah dulu)")
    ap.add_argument("--keluaran", default=None,
                    help="nama file CSV ROI di folder proyek (default: roi_manual.csv)")
    args = ap.parse_args()

    roi_csv = ROOT / args.keluaran if args.keluaran else ROI_CSV
    pilih = CLASSES
    if args.kelas:
        pilih = [k.strip().lower() for k in args.kelas.split(",") if k.strip()]
        salah = [k for k in pilih if k not in CLASSES]
        if salah:
            raise SystemExit(f"kelas tidak dikenal: {salah}. Pilihan: {CLASSES}")

    semua = [r for r in baca_csv(D_SEG / "laporan_segmentasi.csv") if r["status"] in ("perlu_cek", "cadangan_roi")]
    lap = []
    for k in pilih:                                   # urut per kelas, lalu skor terendah dulu
        x = sorted((r for r in semua if r["kelas"] == k), key=lambda r: float(r["skor"]))
        lap += x[:args.maks] if args.maks else x
    roi = {r["file"]: r for r in baca_csv(roi_csv)}

    belum = [r for r in lap if r["file"] not in roi]
    print(f"Kelas: {', '.join(pilih)}")
    for k in pilih:
        n = sum(r["kelas"] == k for r in lap)
        print(f"  {k:7s}: {n} perlu cek, {sum(r['kelas'] == k for r in belum)} belum diberi ROI")
    print(f"File ROI: {roi_csv}\n")

    try:
        for i, r in enumerate(lap, 1):
            if r["file"] in roi:
                continue
            img = baca_gambar(D_CLEAN / "bersih" / r["kelas"] / r["file"])
            judul = f"[{i}/{len(lap)}] {r['kelas']} {r['file']}  ENTER=simpan  ESC=lewati/berhenti"
            x, y, w, h = cv2.selectROI(judul, img, showCrosshair=False)
            cv2.destroyAllWindows()
            for _ in range(4):                        # beri kesempatan jendela benar-benar tertutup
                cv2.waitKey(1)
            if w == 0 or h == 0:
                j = input("  Tidak ada ROI. Enter = lewati, q = berhenti: ").strip().lower()
                if j == "q":
                    break
                continue
            roi[r["file"]] = {"file": r["file"], "x": x, "y": y, "w": w, "h": h}
            tulis_csv(roi_csv, list(roi.values()), ["file", "x", "y", "w", "h"])
    except KeyboardInterrupt:
        print("\nDihentikan (Ctrl+C). ROI yang sudah ditandai tetap tersimpan.")
    finally:
        cv2.destroyAllWindows()
    print(f"{len(roi)} ROI tersimpan di {roi_csv}. Jalankan lagi: python 02_segmentasi.py")


if __name__ == "__main__":
    main()