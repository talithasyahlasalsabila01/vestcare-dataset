"""seg_profil.py - profil segmentasi tiap kelas.

Satu profil = daftar kandidat yang dicoba + pengetahuan awal (prior) tentang objek: luas relatif,
rasio tinggi/lebar, circularity, dan posisi. Semua angka dipakai hanya untuk MEMBERI SKOR kandidat
(seg_core.skor_kandidat), bukan untuk memaksa hasil. Mengubah profil tidak perlu menyentuh kode inti.

ambang_cek : skor minimal agar hasil dianggap layak (status ok). Dikalibrasi dengan memeriksa overlay yang
             diurutkan menurut skor pada dataset ini; di bawah ambang = perlu_cek (lihat 02b_koreksi_roi.py).

Cara membaca trapesium (a, b, c, d): nilai < a atau > d mendapat skor 0, nilai di b..c mendapat
skor penuh, di antaranya naik/turun linear.

  luas      : luas semua objek / luas gambar
  rasio_tw  : tinggi / lebar kotak pembatas satu objek (orang tinggi >1, mobil lebar <1)
  circ      : circularity = 4*pi*A/P^2 (Acara 23); objek berbatas rumit mendapat nilai kecil
  pusat     : titik acuan posisi titik berat (x, y) relatif; skor turun bila jauh dari titik ini
"""

DASAR = {
    # morfologi (pecahan dari sisi terpendek gambar)
    "buka": 0.010, "tutup": 0.020,
    "min_luas": 0.010,          # komponen < 1% luas gambar (dan < 100 px) dibuang
    "min_rel": 0.10,            # komponen < 10% komponen terbesar dibuang
    "maks_objek": 6,
    # prior
    "luas": (0.02, 0.06, 0.60, 0.95),
    "rasio_tw": (0.3, 0.6, 2.0, 4.0),
    "circ": (0.03, 0.10, 0.90, 1.10),
    "pusat": (0.5, 0.5),
    "bobot": {"fit": 0.15, "sep": 0.35, "luas": 0.15, "pos": 0.05, "bentuk": 0.15, "sentuh": 0.15},
    "penalti_sentuh": True,
    "ambang_sentuh": 0.40,                  # sisi gambar dianggap "tersentuh" bila >40% panjangnya tertutup mask
    "tabel_sentuh": [1.0, 1.0, 0.85, 0.35, 0.0],   # skor untuk 0..4 sisi tersentuh
    "pos_skala": 0.70,                      # jarak titik berat dari titik acuan yang memberi skor 0
    "sep_ref": 1.5,                         # jarak warna/sebaran yang dianggap kontras penuh

    # ROI prior
    "pita": 0.06,                          # lebar pita tepi (latar) sebagai pecahan sisi pendek
    "roi_objek": (0.30, 0.25, 0.70, 0.80),  # kotak pusat yang diasumsikan berisi objek
    "roi_bentuk": "kotak",
    "roi_cadangan": (0.15, 0.10, 0.85, 0.95),
    "penalti_roi": 0.55,                   # skor ROI pusat dikalikan ini supaya hanya jadi cadangan
    # parameter kandidat
    "sigma_tekstur": 0.030, "tutup_tepi": 0.020,
    "ambang_cek": 0.50,
    "kandidat": ["latar_lab", "latar_baris", "latar_kolom", "kanal_terbaik", "kanal_terbaik2", "tepi_isi", "tekstur"],
    "gabung": True, "hull": False,
}


def _profil(**ubah):
    p = dict(DASAR)
    p.update(ubah)
    return p


PROFIL = {
    # Pohon: satu tajuk besar bertekstur, hijau, di tengah; latar langit (biru/putih/abu) dan rumput.
    "pohon": _profil(
        kandidat=["latar_lab", "latar_baris", "kanal_terbaik", "kanal_terbaik2", "tepi_isi", "tekstur", "hijau_tekstur"],
        buka=0.012, tutup=0.030, min_luas=0.03, min_rel=0.30, maks_objek=3,
        luas=(0.06, 0.18, 0.70, 0.97), rasio_tw=(0.35, 0.6, 1.8, 3.0), circ=(0.02, 0.08, 0.90, 1.10),
        pusat=(0.5, 0.5), roi_objek=(0.28, 0.22, 0.72, 0.75), roi_bentuk="elips",
        roi_cadangan=(0.10, 0.05, 0.90, 0.95),
        bobot={"fit": 0.10, "sep": 0.30, "luas": 0.15, "pos": 0.25, "bentuk": 0.10, "sentuh": 0.20, "dasar": 0.35},
        pos_skala=0.45,
        ambang_sentuh=0.30, tabel_sentuh=[1.0, 0.75, 0.40, 0.10, 0.0], ambang_cek=0.85,
    ),
    # Lubang jalan: area gelap/berbeda warna di aspal; bisa banyak; berbentuk gumpalan melebar.
    "lubang": _profil(
        kandidat=["gelap_lokal", "gelap_kecil", "air_cokelat", "gelap_atau_air", "selisih_aspal", "gelap_atau_selisih", "latar_lab", "tepi_isi"],
        buka=0.004, tutup=0.012, min_luas=0.002, min_rel=0.10, maks_objek=6,
        luas=(0.001, 0.004, 0.40, 0.80), rasio_tw=(0.12, 0.25, 1.4, 3.0), circ=(0.04, 0.15, 0.95, 1.10),
        pusat=(0.5, 0.65), roi_objek=(0.25, 0.40, 0.75, 0.95), roi_bentuk="elips",
        roi_cadangan=(0.20, 0.40, 0.80, 0.95), penalti_roi=0.5,
        bobot={"fit": 0.15, "sep": 0.35, "luas": 0.15, "pos": 0.05, "bentuk": 0.20, "sentuh": 0.10},
        gabung=False, zona_aspal=(0.0, 0.30, 1.0, 1.0), batasi_zona=True, blok_lokal=0.25, c_lokal=10, ambang_cek=0.80,
    ),
    # Orang: tegak (tinggi > lebar), pakaian beragam, sering berkelompok; latar ruang/pantai/putih.
    "orang": _profil(
        buka=0.012, tutup=0.020, min_luas=0.015, min_rel=0.12, maks_objek=6,
        luas=(0.03, 0.08, 0.55, 0.95), rasio_tw=(0.7, 1.4, 4.5, 8.0), circ=(0.02, 0.06, 0.45, 0.80),
        pusat=(0.5, 0.55), roi_objek=(0.30, 0.10, 0.70, 0.95), roi_cadangan=(0.20, 0.05, 0.80, 0.98),
        ambang_cek=0.84,
    ),
    # Motor: gabungan motor + pengendara/parkir, bentuk hampir kotak, sering banyak dan rapat.
    "motor": _profil(
        buka=0.010, tutup=0.020, min_luas=0.015, min_rel=0.12, maks_objek=6,
        luas=(0.03, 0.08, 0.70, 0.95), rasio_tw=(0.35, 0.65, 1.7, 3.0), circ=(0.04, 0.10, 0.65, 0.95),
        pusat=(0.5, 0.6), roi_objek=(0.25, 0.25, 0.75, 0.90), roi_cadangan=(0.10, 0.20, 0.90, 0.98),
        zona_aspal=(0.0, 0.22, 1.0, 1.0), batasi_zona=True, ambang_cek=0.88, hull=True,
    ),
    # Mobil: lebar (lebar > tinggi), badan halus berwarna seragam; latar jalan/gedung/pohon.
    "mobil": _profil(
        buka=0.012, tutup=0.020, min_luas=0.015, min_rel=0.12, maks_objek=6,
        luas=(0.03, 0.08, 0.70, 0.95), rasio_tw=(0.25, 0.40, 1.0, 1.8), circ=(0.06, 0.20, 0.85, 1.05),
        pusat=(0.5, 0.58), roi_objek=(0.25, 0.30, 0.75, 0.90), roi_cadangan=(0.08, 0.20, 0.92, 0.98),
        zona_aspal=(0.0, 0.18, 1.0, 1.0), batasi_zona=True, ambang_cek=0.95, hull=True,
    ),
}
