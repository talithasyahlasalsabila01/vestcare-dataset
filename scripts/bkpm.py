"""bkpm.py - padanan fungsi MATLAB pada modul BKPM Workshop Pengolahan Citra dan Vision
dalam Python (OpenCV + NumPy). Semua tahap pipeline hanya memakai fungsi di sini,
sehingga setiap langkah bisa ditelusuri ke Acara BKPM-nya.

  Acara 5-8   operasi dasar      : rgb2gray, flip, crop (ROI), operasi Boolean AND/OR
  Acara 9-12  image enhancement  : brightness (F+k), imadjust (contrast stretching), histeq
  Acara 13-14 konvolusi          : kernel garis (blur gerak), filter linear
  Acara 15    segmentasi ROI     : crop ROI (kotak / elips)
  Acara 16    background removal : ambang pada kanal RGB / HSV / L*a*b* (Color Thresholder)
  Acara 17    citra biner        : ambang global (T tetap), Otsu (graythresh)
  Acara 18-20 morfologi          : strel, imerode, imdilate, imopen, imclose, imfill
  Acara 22-23 fitur bentuk       : bwlabeln 8-tetangga, area, perimeter, bentuk, circularity
  Acara 25-28 filtering          : medfilt2, imgaussfilt
  Acara 29-32 deteksi tepi       : Sobel (skor kecocokan batas), Canny (kandidat tepi_isi/tekstur)

Catatan: Acara 30 (Prewitt) dan Acara 32 (Roberts) sempat diimplementasikan di modul ini tapi
tidak pernah dipanggil oleh pipeline manapun (seg_core.py hanya memakai Sobel dan Canny), jadi
keduanya dihapus supaya dokumentasi ini sesuai dengan kode yang benar-benar jalan.
"""
import math

import cv2
import numpy as np


# ======================================================================
# Acara 5-6: konversi warna
# ======================================================================
def rgb2gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def kanal_warna(bgr):
    """Kanal yang tersedia di Color Thresholder (Acara 16): R G B, S V (HSV), L* a* b* (CIELab).

    Kanal H tidak dipakai untuk pemilihan otomatis karena sifatnya melingkar (merah ada di 0 dan 179)
    sehingga ambang tunggal tidak bermakna. Semua kanal berskala 0-255 (uint8).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(bgr)
    return {"R": r, "G": g, "B": b, "S": hsv[..., 1], "V": hsv[..., 2],
            "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}


# ======================================================================
# Acara 9-12: image enhancement
# ======================================================================
def brightness(img, k):
    """F_o = F_i + k (Acara 11)."""
    return cv2.convertScaleAbs(img, alpha=1.0, beta=k)


def imadjust(gray, p_bawah=1.0, p_atas=99.0):
    """Contrast stretching: perlebar sebaran nilai keabuan ke 0-255 (Acara 11)."""
    lo, hi = np.percentile(gray, [p_bawah, p_atas])
    if hi - lo < 1:
        return gray.copy()
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def histeq(gray):
    """Histogram equalization (Acara 12)."""
    return cv2.equalizeHist(gray)


def enhance_warna(bgr, metode="imadjust"):
    """Terapkan imadjust / histeq pada kecerahan (kanal L*) agar warna tidak bergeser."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    lab[..., 0] = imadjust(L) if metode == "imadjust" else histeq(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ======================================================================
# Acara 25-28: filtering
# ======================================================================
def medfilt2(img, k=3):
    return cv2.medianBlur(img, k)


def imgaussfilt(img, sigma=1.0):
    """Ukuran jendela mengikuti MATLAB: 2*ceil(2*sigma)+1."""
    k = int(2 * math.ceil(2 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


# ======================================================================
# Acara 13-14: konvolusi
# ======================================================================
def kernel_gerak(panjang, sudut):
    """Kernel blur gerak (garis) sepanjang `panjang` piksel pada `sudut` derajat (fspecial 'motion')."""
    n = max(3, int(round(panjang))) | 1
    k = np.zeros((n, n), np.float32)
    c = n // 2
    dx = math.cos(math.radians(sudut)) * c
    dy = math.sin(math.radians(sudut)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))),
             (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return k / k.sum()


def konvolusi(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ======================================================================
# Acara 17: citra biner
# ======================================================================
def graythresh(gray):
    """Ambang Otsu (0-255) dari citra uint8 atau larik 1-D uint8."""
    g = np.asarray(gray, np.uint8).reshape(-1, 1)
    if g.size < 2 or g.min() == g.max():
        return int(g.min()) if g.size else 128
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def otsu_float(arr, p_bawah=0.5, p_atas=99.5):
    """Otsu untuk peta bernilai real (mis. jarak warna). Mengembalikan ambang dalam satuan arr."""
    lo, hi = np.percentile(arr, [p_bawah, p_atas])
    if hi - lo < 1e-6:
        return float(hi)
    g = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return lo + graythresh(g) * (hi - lo) / 255.0


def ambang_lokal(gray, ukuran, c):
    """Locally adaptive thresholding (Acara 17): piksel gelap jika < rata-rata lokal Gaussian - c."""
    sigma = max(1.0, ukuran / 3.0)
    k = int(ukuran) | 1
    rata = cv2.GaussianBlur(gray.astype(np.float32), (k, k), sigma)
    return gray.astype(np.float32) < (rata - c)


# ======================================================================
# Acara 18-20: morfologi
# ======================================================================
def strel(bentuk, r, sudut=0):
    """strel('disk'|'square'|'diamond'|'line', r). Untuk 'line' r = panjang, sudut dalam derajat."""
    r = max(1, int(r))
    if bentuk == "disk":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    if bentuk == "square":
        return np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    if bentuk == "diamond":
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (np.abs(x) + np.abs(y) <= r).astype(np.uint8)
    if bentuk == "line":
        n = r | 1
        k = np.zeros((n, n), np.uint8)
        c = n // 2
        dx, dy = math.cos(math.radians(sudut)) * c, -math.sin(math.radians(sudut)) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
        return k
    raise ValueError(bentuk)


def _u8(bw):
    return (np.asarray(bw) > 0).astype(np.uint8)


def imerode(bw, se):
    return cv2.erode(_u8(bw), se) > 0


def imdilate(bw, se):
    return cv2.dilate(_u8(bw), se) > 0


def imopen(bw, se):
    """Opening = erosi lalu dilasi: membuang objek kecil/kurus, menghaluskan batas (Acara 19)."""
    return cv2.morphologyEx(_u8(bw), cv2.MORPH_OPEN, se) > 0


def imclose(bw, se):
    """Closing = dilasi lalu erosi: menutup lubang kecil, menyatukan objek berdekatan (Acara 19)."""
    return cv2.morphologyEx(_u8(bw), cv2.MORPH_CLOSE, se) > 0


def imfill(bw):
    """Isi lubang: latar yang tidak terhubung ke tepi gambar dijadikan objek."""
    m = _u8(bw) * 255
    p = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8)
    cv2.floodFill(p, ff, (0, 0), 255)
    lubang = cv2.bitwise_not(p)[1:-1, 1:-1]
    return (m | lubang) > 0


# ======================================================================
# Operasi Boolean (Acara 8)
# ======================================================================
def bw_and(a, b):
    return np.logical_and(a, b)


def bw_or(a, b):
    return np.logical_or(a, b)


# ======================================================================
# Acara 29, 31: deteksi tepi (hanya metode yang benar-benar dipakai pipeline)
# ======================================================================
def gradien(gray):
    """Besar gradien (float32) dengan operator Sobel (Acara 29). Dipakai di seg_core.py untuk
    skor kecocokan batas (s_fit), bukan untuk membuat mask."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge(gray):
    """Peta tepi biner Canny (Acara 31): ambang atas = ambang Otsu citra, ambang bawah = setengahnya.
    Dipakai di seg_core.py sebagai dasar kandidat mask tepi_isi/tekstur/hijau_tekstur."""
    hi = max(20, graythresh(gray))
    return cv2.Canny(gray, 0.5 * hi, hi) > 0


# ======================================================================
# Acara 22-23: pelabelan komponen dan fitur bentuk
# ======================================================================
def bwlabeln(bw, konektivitas=8):
    """Pelabelan komponen terhubung. Mengembalikan (jumlah, label, stats[x,y,w,h,area])."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(_u8(bw), connectivity=konektivitas)
    return n - 1, lab, stats[1:]


def perimeter_batas(bw):
    """Perimeter = jumlah piksel batas objek: piksel objek yang punya tetangga (atas/bawah/kiri/kanan)
    bukan objek (rumus pada source code Acara 23)."""
    m = _u8(bw)
    p = np.pad(m, 1)
    dalam = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return int((m & (1 - dalam)).sum())


# arah kode rantai 8-tetangga: 0 timur, berlawanan arah jarum jam (Acara 22, Gambar 35a)
ARAH_RANTAI = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3, (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}


def kode_rantai(bw):
    """Kode rantai (Freeman 8 arah) kontur luar komponen bw (satu komponen). Mengembalikan list int."""
    k, _ = cv2.findContours(_u8(bw), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not k:
        return []
    c = max(k, key=len)[:, 0, :]
    rantai = []
    for i in range(len(c)):
        dx, dy = int(c[(i + 1) % len(c)][0] - c[i][0]), int(c[(i + 1) % len(c)][1] - c[i][1])
        if (dx, dy) in ARAH_RANTAI:
            rantai.append(ARAH_RANTAI[(dx, dy)])
    return rantai


def keliling_rantai(rantai):
    """Panjang kode rantai: langkah genap (tegak/mendatar) = 1, ganjil (diagonal) = akar 2 (Acara 22)."""
    genap = sum(1 for c in rantai if c % 2 == 0)
    return genap + math.sqrt(2) * (len(rantai) - genap)


def faktor_bentuk(area, perimeter):
    """bentuk = perimeter^2 / area (Acara 22-23)."""
    return perimeter ** 2 / area if area > 0 else float("nan")


def circularity(area, perimeter):
    """C = 4*pi*area / perimeter^2 (Acara 23); 1 = lingkaran."""
    return 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else float("nan")
