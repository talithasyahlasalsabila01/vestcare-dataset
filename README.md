# Pipeline dataset YOLO26 - rompi cerdas (BKPM Workshop Pengolahan Citra dan Vision)

## Struktur folder
```
<proyek>/
  dataset_sumber_asli/{pohon,lubang,orang,motor,mobil}/*.jpg
  scripts/            <- folder ini
  hasil/              <- dibuat otomatis
```
Kebutuhan: `pip install -r requirements.txt` (opencv-python, numpy). Tanpa GrabCut atau library segmentasi lain.

## Menjalankan
```
python run_all.py --zip                 # tahap 1-6
python run_all.py --mulai 3             # lanjut dari tahap 3
python 02_segmentasi.py --kelas pohon   # satu kelas
python 02b_koreksi_roi.py               # opsional: tandai ROI untuk gambar perlu_cek, lalu ulangi 02
```
Segmentasi memakan beberapa menit (sekitar 8-10 menit untuk ~680 gambar).

## Melihat hasil tiap tahap
Buka `scripts/visualisasi_hasil_tahapan.ipynb` di JupyterLab dan jalankan sel berurutan untuk menampilkan
before/after cleaning, subplot proses segmentasi (termasuk ROI manual), labelling, contoh pengayaan, serta
fitur bentuk dan visualisasi kode rantai. Galeri di bagian akhir menampilkan lima sampel: `lubang_92`,
`mobil_IMG_20815`, `motor_motor_IMG_4199p1_0809`, `orang_orang_4`, dan `pohon_sekunder-pohon-10`.
Tabel fitur numerik untuk kelima sampel juga ditampilkan sebagai output notebook, diambil dari CSV tahap 05.
Pada tahap 5, gambar segmentasi akhir dan polygon hasil labelling ditampilkan berdampingan dengan jalur kode rantai;
mask biner mentah tidak lagi menjadi panel utama.
Notebook hanya membaca berkas yang sudah ada di `hasil/` dan `roi_manual.csv`; notebook tidak menjalankan
ulang tahap 01-05. Sel konfigurasi contoh di bagian atas dapat diubah untuk melihat sampel lain.

## Tahap dan padanan BKPM
| Tahap | Isi | Acara BKPM |
|---|---|---|
| 01 cleaning | buang rusak, duplikat (MD5/dHash), gelap/terang (histogram), blur parah (Sobel, blur gerak dipertahankan), varian blur lama | 10-12, 30 |
| 02 segmentasi | praproses median+Gaussian+imadjust/histeq; kandidat mask (jarak warna Lab dari pita tepi, kanal terbaik+Otsu, Canny+closing+imfill, tekstur, hijau a*, ambang lokal, air cokelat b*, fusi Boolean); pilih skor terbaik per gambar | 8, 11-12, 15-19, 25-32 |
| 02b koreksi ROI | ROI manual untuk gambar perlu_cek | 15 |
| 03 labelling | mask -> polygon & kotak YOLO | 23 |
| 04 pengayaan | cerah/gelap, imadjust, histeq, Gaussian, blur gerak (konvolusi), flip, crop | 8, 11-14, 15, 28 |
| 05 fitur bentuk | bwlabeln 8, area, perimeter, bentuk P^2/A, circularity, kode rantai (+QC) | 22-23 |
| 06 dataset | split 90:10 per foto asli, `seg/` dan `det/` + data.yaml | - |

## Bagaimana satu kode bisa menangani bentuk dan latar berbeda
- `seg_profil.py` berisi profil tiap kelas: kandidat yang dicoba dan prior bentuk (luas, rasio tinggi/lebar, circularity, posisi).
- `seg_core.py` mencoba semua kandidat pada setiap gambar, memberi skor (kecocokan tepi Sobel, kontras objek terhadap cincin sekitarnya, luas, posisi, bentuk, solidity, penyentuhan tepi/dasar gambar), lalu memakai yang terbaik.
- Status per gambar di `hasil/02_segmentasi/laporan_segmentasi.csv`: `ok`, `perlu_cek`, `cadangan_roi`, `roi_manual`. Ambang `ambang_cek` per kelas dikalibrasi visual pada dataset ini; ubah di `seg_profil.py` bila dataset berganti.
- Hanya `ok`/`roi_manual` yang dilabel (`--sertakan-perlu-cek` untuk memaksa semua).

## Pelatihan (Ultralytics)
```
yolo segment train model=yolo26n-seg.pt data=hasil/06_dataset_yolo26/seg/data.yaml imgsz=640
yolo detect  train model=yolo26n.pt     data=hasil/06_dataset_yolo26/det/data.yaml imgsz=640
```
data.yaml tidak berisi `path:` sehingga dataset bisa dipindah (mis. ke Colab) tanpa mengubah apa pun.

## Gambar `perlu_cek` -> koreksi ROI (cara utama untuk latar ramai)
Segmentasi otomatis andal pada latar polos/berpita, tetapi tidak pada latar ramai. Gambar yang skornya di bawah
`ambang_cek` (seg_profil.py) tidak dilabel. Perbaiki dengan:
```
python 02b_koreksi_roi.py --kelas motor     # tarik kotak di sekeliling SEMUA objek, ENTER
python run_all.py --mulai 2
```
Pada gambar ber-ROI, latar diambil dari pita tepat di luar ROI dan hasil dibatasi di dalam ROI. Menyetel ambang
tidak perlu mengulang segmentasi: ubah `ambang_cek`, lalu `python run_all.py --mulai 3`.

## Batasan
- Segmentasi klasik tidak andal pada latar ramai (parkiran padat, kebun, jalan basah). Periksa `overlay/` dan `perlu_cek.txt`, koreksi dengan `02b_koreksi_roi.py`.
- Motor dan mobil hanya 30 foto asli per kelas; varian blur lama di folder sumber dibuang dan dibuat ulang di tahap 4.
- Label hanya untuk kelas pada folder foto tersebut; objek kelas lain di foto yang sama tidak dilabel.
- Val hanya berisi foto asli, jadi jumlahnya kecil.
