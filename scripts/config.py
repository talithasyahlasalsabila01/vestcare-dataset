"""config.py - pengaturan bersama semua tahap.

Struktur folder yang diharapkan:
    <proyek>/
        dataset_sumber_asli/<kelas>/*.jpg     <- foto asli (tidak pernah diubah)
        scripts/                              <- folder ini
        hasil/                                <- dibuat otomatis
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROYEK_ROOT", Path(__file__).resolve().parent.parent))
SUMBER_DIR = ROOT / "dataset_sumber_asli"
HASIL = ROOT / "hasil"

# Urutan ID kelas YOLO: 0 pohon, 1 lubang, 2 orang, 3 motor, 4 mobil
CLASSES = ["pohon", "lubang", "orang", "motor", "mobil"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

UKURAN_MAKS = 640          # sisi terpanjang gambar kerja (tidak pernah di-upscale)

# ---- folder hasil tiap tahap ----
D_CLEAN = HASIL / "01_cleaning"
D_SEG = HASIL / "02_segmentasi"
D_LABEL = HASIL / "03_labelling"
D_AUG = HASIL / "04_pengayaan"
D_FITUR = HASIL / "05_fitur_bentuk"
D_YOLO = HASIL / "06_dataset_yolo26_70_15_15"   # nama sebenarnya mengikuti rasio split (lihat 06_dataset_yolo26.py)
D_BANDING = HASIL / "07_perbandingan_model"

SEG_SUB = ["kontras", "mask", "hasil_segmentasi", "overlay", "visualisasi_proses"]
