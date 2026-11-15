"""run_all.py - menjalankan seluruh pipeline berurutan: 01 cleaning -> 02 segmentasi -> 03 labelling
-> 04 pengayaan -> 05 ekstraksi fitur bentuk -> 06 dataset YOLO26 (train:val:test 70:15:15, seed 0).
Untuk preset 70:20:10 (opsional, bersyarat -- lihat docstring-nya), jalankan
06b_dataset_yolo26_70_20_10.py secara terpisah setelah run_all.py selesai sampai tahap 5.

    python run_all.py
    python run_all.py --mulai 3 --sertakan-perlu-cek
    python run_all.py --zip
"""
import argparse
import subprocess
import sys
from pathlib import Path

LANGKAH = ["01_cleaning.py", "02_segmentasi.py", "03_labelling.py", "04_pengayaan.py",
           "05_ekstraksi_fitur.py", "06_dataset_yolo26.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mulai", type=int, default=1, help="mulai dari tahap ke-N (1-6)")
    ap.add_argument("--sampai", type=int, default=6)
    ap.add_argument("--sertakan-perlu-cek", action="store_true", help="tahap 3: ikutkan gambar perlu_cek")
    ap.add_argument("--zip", action="store_true", help="tahap 6: buat zip untuk Colab")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    for i, f in enumerate(LANGKAH, 1):
        if not (a.mulai <= i <= a.sampai):
            continue
        cmd = [sys.executable, str(here / f)]
        if i == 3 and a.sertakan_perlu_cek:
            cmd.append("--sertakan-perlu-cek")
        if i == 6 and a.zip:
            cmd.append("--zip")
        print(f"\n===== {f} =====", flush=True)
        if subprocess.call(cmd, cwd=here) != 0:
            sys.exit(f"gagal di {f}")


if __name__ == "__main__":
    main()
