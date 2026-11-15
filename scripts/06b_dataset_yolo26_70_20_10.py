"""TAHAP 6b (OPSIONAL) - DATASET YOLO26, SPLIT 70:20:10, seed 0

File TERPISAH dari 06_dataset_yolo26.py supaya memakai rasio 70:20:10 selalu menjadi PILIHAN SADAR,
bukan default yang kepakai tanpa sengaja lewat argumen CLI yang gampang terlewat. Default pipeline
(run_all.py, dan 06_dataset_yolo26.py tanpa argumen) tetap 70:15:15.

Boleh pakai file ini, TAPI dengan syarat:
  1. Kamu memang sengaja butuh porsi val LEBIH BESAR (20%, dibanding 15% default) -- misalnya karena
     sedang melakukan tuning hyperparameter yang lebih teliti dan ingin val yang lebih stabil.
  2. Kamu sadar porsi test jadi LEBIH KECIL (10%, dibanding 15% default). Untuk kelas yang foto
     aslinya sedikit (~100), test 10% bisa berarti ~10 foto saja -- satu kesalahan prediksi bisa
     menggeser mAP kelas itu cukup jauh. Kalau laporan utamamu menekankan keandalan per kelas,
     pakai default 70:15:15 (06_dataset_yolo26.py), bukan file ini.
  3. Semua aturan anti-kebocoran di 06_dataset_yolo26.py tetap berlaku di sini -- file ini HANYA
     mengganti rasio (val=0.20, test=0.10); logika split/grouping/audit/copy-nya dipanggil langsung
     dari modul 06_dataset_yolo26.py (fungsi jalankan()), tidak diduplikasi/ditulis ulang.
  4. Pakai rasio ini untuk SATU pipeline penuh secara konsisten (jangan campur train dari preset
     70:15:15 dengan val/test dari preset 70:20:10). Tiap preset kini ditulis ke folder sendiri
     (hasil/06_dataset_yolo26_70_20_10), jadi tidak menimpa hasil preset 70:15:15 dan keduanya bisa
     dibandingkan berdampingan lewat 07_bandingkan_model.py --dataset <nama folder>).

Jalankan:
    python 06b_dataset_yolo26_70_20_10.py
    python 06b_dataset_yolo26_70_20_10.py --seed 1 --zip
"""
import argparse
import importlib.util
from pathlib import Path

# nama modul "06_dataset_yolo26" diawali angka -> tidak valid untuk `import` biasa,
# jadi dimuat lewat importlib dari path filenya.
_spec = importlib.util.spec_from_file_location(
    "dataset_yolo26_core", Path(__file__).resolve().parent / "06_dataset_yolo26.py")
inti = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inti)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-dengan-varian", action="store_true")
    ap.add_argument("--test-dengan-varian", action="store_true")
    ap.add_argument("--buat-robustness-test", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    inti.log(inti.TAHAP, "PERHATIAN: menjalankan preset 70:20:10 (bukan default 70:15:15). "
                          "Pastikan ini pilihan sadar -- lihat syarat di docstring file ini.")
    inti.jalankan(val=0.20, test=0.10, seed=args.seed,
                   val_dengan_varian=args.val_dengan_varian,
                   test_dengan_varian=args.test_dengan_varian,
                   buat_robustness_test_=args.buat_robustness_test,
                   zip_=args.zip, _preset_eksplisit=False)


if __name__ == "__main__":
    main()
