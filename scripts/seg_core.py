"""seg_core.py - inti segmentasi adaptif (hanya teknik BKPM).

Masalah: bentuk pohon, lubang, orang, motor, dan mobil berbeda, dan latar belakangnya bervariasi,
sehingga satu rumus segmentasi tidak cukup. Cara menyelesaikannya mengikuti cara kerja mahasiswa di
BKPM Acara 16 (Color Thresholder): mencoba beberapa kanal/ambang sampai hasilnya bagus. Di sini
percobaan itu dilakukan otomatis untuk setiap gambar:

  1. PRAPROSES      median + Gaussian (Acara 25-28), contrast stretching / histeq (Acara 11-12)
  2. KANDIDAT MASK  beberapa cara pemisahan objek-latar (semuanya teknik BKPM):
        latar_lab      jarak warna L*a*b* dari referensi latar di pita tepi gambar (Acara 16)
        kanal_terbaik  kanal RGB/HSV/Lab yang paling memisahkan ROI pusat dari pita tepi + Otsu (16-17)
        tepi_isi       tepi Canny -> closing -> imfill (Acara 18-19, 29-32)
        tekstur        kepadatan tepi -> Otsu -> closing -> imfill (untuk objek berdaun/bertekstur)
        hijau          kanal a* Lab sisi hijau (Acara 16), dengan/tanpa tekstur (khusus pohon)
        gelap_lokal    ambang adaptif lokal pada citra keabuan (Acara 17), (khusus lubang)
        selisih_aspal  jarak warna Lab dari warna aspal (khusus lubang)
        fusi           suara terbanyak 2 dari 3 kandidat terbaik (operasi Boolean, Acara 8)
        roi_pusat      kotak/elips ROI di tengah (Acara 15) - cadangan terakhir
        air_cokelat    kanal b* Lab sisi kuning-cokelat di zona jalan (genangan air pada lubang)
  3. PASCA-PROSES   imopen, imclose, imfill, bwlabeln 8-tetangga, saring luas (Acara 18-19, 23)
  4. PEMILIHAN      setiap kandidat diberi skor: kecocokan batas dengan tepi Sobel, luas, posisi,
                    bentuk (circularity, rasio aspek), dan penyentuhan tepi gambar. Kandidat dengan
                    skor tertinggi dipakai. Bobot dan batas per kelas ada di seg_profil.py.
"""
import cv2
import numpy as np

import bkpm
from seg_profil import PROFIL

SD_HOMOGEN = 24.0         # pita tepi dianggap "latar" bila sebaran warnanya (Lab) kecil


# ======================================================================
# 1. PRAPROSES
# ======================================================================
def praproses(bgr):
    """Kembalikan dict citra kerja. `kontras` = citra yang dipakai semua kandidat."""
    halus = bkpm.imgaussfilt(bkpm.medfilt2(bgr, 3), 1.0)
    lab = cv2.cvtColor(halus, cv2.COLOR_BGR2LAB)
    p1, p99 = np.percentile(lab[..., 0], [1, 99])
    metode = "histeq" if (p99 - p1) < 110 else "imadjust"      # histogram sempit -> histeq
    kontras = bkpm.enhance_warna(halus, metode)
    gray = bkpm.rgb2gray(kontras)
    return {
        "halus": halus, "kontras": kontras, "gray": gray, "metode_kontras": metode,
        "lab": cv2.cvtColor(kontras, cv2.COLOR_BGR2LAB).astype(np.float32),
        "kanal": bkpm.kanal_warna(kontras),
        "grad": bkpm.gradien(gray),
        "tepi": bkpm.edge(gray),
    }


# ======================================================================
# ROI prior: pita tepi (latar) dan kotak pusat (objek)
# ======================================================================
def pita_tepi(h, w, lebar_frac=0.06):
    bw = max(4, int(lebar_frac * min(h, w)))
    sisi = {k: np.zeros((h, w), bool) for k in ("atas", "bawah", "kiri", "kanan")}
    sisi["atas"][:bw, :] = True
    sisi["bawah"][-bw:, :] = True
    sisi["kiri"][:, :bw] = True
    sisi["kanan"][:, -bw:] = True
    return sisi


def roi_kotak(h, w, frac):
    x0, y0, x1, y1 = frac
    m = np.zeros((h, w), bool)
    m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    return m


def roi_elips(h, w, frac):
    x0, y0, x1, y1 = frac
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int((x0 + x1) / 2 * w), int((y0 + y1) / 2 * h)),
                (max(1, int((x1 - x0) / 2 * w)), max(1, int((y1 - y0) / 2 * h))), 0, 0, 360, 1, -1)
    return m > 0


# ======================================================================
# 2. KANDIDAT MASK (masing-masing mengembalikan mask bool mentah atau None)
# ======================================================================
def kand_latar_lab(P, pita, prof):
    """Jarak warna Lab ke referensi latar. Referensi = median tiap sisi yang warnanya seragam."""
    lab = P["lab"]
    refs = []
    for pm in pita.values():
        pix = lab[pm]
        if pix.std(0).mean() <= SD_HOMOGEN:
            refs.append(np.median(pix, axis=0))
    if not refs:
        return None
    d = np.min([np.linalg.norm(lab - r, axis=2) for r in refs], axis=0)
    return d > bkpm.otsu_float(d)


def kand_kanal_terbaik(P, pita, roi_obj, jumlah=1):
    """Pilih kanal Color Thresholder yang paling memisahkan ROI pusat (objek) dari pita tepi (latar),
    ukuran pemisahan = kriteria Fisher. Ambang = Otsu dihitung dari piksel kedua area itu."""
    pm = np.logical_or.reduce(list(pita.values()))
    skor = []
    for nama, c in P["kanal"].items():
        o, b = c[roi_obj].astype(np.float32), c[pm].astype(np.float32)
        if o.size < 10 or b.size < 10:
            continue
        j = (o.mean() - b.mean()) ** 2 / (o.var() + b.var() + 1e-3)
        skor.append((j, nama, o.mean() > b.mean()))
    skor.sort(reverse=True)
    out = []
    for j, nama, obj_terang in skor[:jumlah]:
        c = P["kanal"][nama]
        t = bkpm.graythresh(np.concatenate([c[roi_obj], c[pm]]))
        out.append(c > t if obj_terang else c <= t)
    if not out:
        return None
    m = out[0]
    for o in out[1:]:
        m = bkpm.bw_and(m, o)
    return m


def kand_latar_garis(P, sumbu):
    """Latar berpita (langit/laut/pasir, jalan/dinding): latar tiap BARIS diperkirakan dari median pita
    kiri dan kanan baris itu (atau tiap KOLOM dari pita atas dan bawah), dihaluskan dengan Gaussian.
    Objek = piksel yang jaraknya (Lab) besar dari kedua referensi, ambang Otsu. Memperluas
    latar_lab (satu warna per sisi) menjadi satu warna per baris/kolom."""
    lab = P["lab"].astype(np.float32)
    h, w = lab.shape[:2]
    if sumbu == "baris":
        bw = max(6, int(0.10 * w))
        refs = [np.median(lab[:, :bw], axis=1), np.median(lab[:, -bw:], axis=1)]           # (h, 3)
        k = int(2 * np.ceil(0.02 * h) * 2 + 1)
        refs = [cv2.GaussianBlur(r.reshape(h, 1, 3), (1, k), max(1.0, 0.02 * h)).reshape(h, 3)[:, None, :] for r in refs]
    else:
        bh = max(6, int(0.10 * h))
        refs = [np.median(lab[:bh], axis=0), np.median(lab[-bh:], axis=0)]                  # (w, 3)
        k = int(2 * np.ceil(0.02 * w) * 2 + 1)
        refs = [cv2.GaussianBlur(r.reshape(1, w, 3), (k, 1), max(1.0, 0.02 * w)).reshape(w, 3)[None, :, :] for r in refs]
    d = np.minimum(*[np.linalg.norm(lab - r, axis=2) for r in refs])
    return d > bkpm.otsu_float(d)


def kand_tepi_isi(P, r_tutup):
    e = bkpm.imdilate(P["tepi"], bkpm.strel("disk", 1))
    e = bkpm.imclose(e, bkpm.strel("disk", r_tutup))
    return bkpm.imfill(e)


def peta_tekstur(P, sigma):
    dens = bkpm.imgaussfilt(P["tepi"].astype(np.float32), sigma)
    return dens > bkpm.otsu_float(dens)


def kand_tekstur(P, sigma, r_tutup):
    m = bkpm.imclose(peta_tekstur(P, sigma), bkpm.strel("disk", r_tutup))
    return bkpm.imfill(m)


def kand_hijau(P, dengan_tekstur, sigma, a_maks=124):
    """Vegetasi = sisi hijau kanal a* (nilai kecil). Ambang = min(Otsu, a_maks)."""
    a = P["kanal"]["a"]
    t = min(bkpm.graythresh(a), a_maks)
    m = a < t
    if dengan_tekstur:
        m = bkpm.bw_and(m, peta_tekstur(P, sigma))
    return m


def kand_gelap_lokal(P, ukuran, c):
    return bkpm.ambang_lokal(P["gray"], ukuran, c)


def kand_selisih_aspal(P, zona):
    """Warna aspal = median Lab pada zona jalan; objek = jarak warna besar (Otsu)."""
    lab = P["lab"]
    ref = np.median(lab[zona], axis=0)
    d = np.linalg.norm(lab - ref, axis=2)
    return (d > bkpm.otsu_float(d[zona])) & zona


def kand_air(P, zona):
    """Genangan air keruh = kuning-cokelat pada kanal b* (nilai tinggi), bukan vegetasi (a* tidak hijau).
    Aspal kering abu-abu punya b* mendekati netral (128)."""
    b, a = P["kanal"]["b"], P["kanal"]["a"]
    z = zona if zona is not None else np.ones(b.shape, bool)
    if z.sum() < 50:
        return None
    t = max(bkpm.graythresh(b[z]), 134)
    return (b > t) & (a >= 126) & z


def pohon_perbaiki(raw, P, sigma):
    """Khusus pohon. (1) buang tanah: area halus (tidak bertekstur) dan hijau di 45% bawah gambar = rumput;
    batang cokelat tidak ikut terbuang. (2) isi tajuk gelap: convex hull tajuk (setelah batang
    dibuang dengan opening besar) dikurangi area halus (langit/rumput) - bagian dalam tajuk yang
    gelap tetap bertekstur sehingga ikut terisi."""
    h, w = raw.shape
    s = min(h, w)
    tek = bkpm.imclose(peta_tekstur(P, sigma), bkpm.strel("disk", max(2, int(0.01 * s))))
    hijau = P["kanal"]["a"] < 124
    bawah = np.zeros((h, w), bool)
    bawah[int(0.55 * h):, :] = True
    m = np.asarray(raw, bool) & ~(bawah & ~tek & hijau)
    tajuk = bkpm.imopen(m, bkpm.strel("disk", max(2, int(0.04 * s))))
    n, lab, st = bkpm.bwlabeln(tajuk, 8)
    if n:
        pilih = [i for i in range(n) if st[i, 4] >= 0.3 * st[:, 4].max()]
        pts = np.concatenate([np.argwhere(lab == i + 1)[:, ::-1] for i in pilih]).astype(np.int32)
        hull = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        m = m | ((hull > 0) & tek & ~(bawah & ~tek & hijau))
    return m


def kand_roi_pusat(h, w, prof):
    f = prof["roi_cadangan"]
    return roi_elips(h, w, f) if prof["roi_bentuk"] == "elips" else roi_kotak(h, w, f)


# ======================================================================
# 3. PASCA-PROSES: morfologi + bwlabeln + saring luas
# ======================================================================
def pasca(mask, prof, zona=None):
    """Kembalikan peta label instans (0 = latar, 1..n) dan jumlah n."""
    h, w = mask.shape
    s = min(h, w)
    m = np.asarray(mask, bool)
    if zona is not None:
        m = m & zona
    m = bkpm.imopen(m, bkpm.strel("disk", max(1, int(prof["buka"] * s))))
    m = bkpm.imclose(m, bkpm.strel("disk", max(1, int(prof["tutup"] * s))))
    m = bkpm.imfill(m)
    n, lab, st = bkpm.bwlabeln(m, 8)
    if n == 0:
        return np.zeros((h, w), np.uint8), 0
    area = st[:, 4]
    min_px = max(100, prof["min_luas"] * h * w)                 # BKPM Acara 23: buang area < 100 px
    ada = [i for i in range(n) if area[i] >= min_px and area[i] >= prof["min_rel"] * area.max()]
    ada.sort(key=lambda i: -area[i])
    ada = ada[:prof["maks_objek"]]
    out = np.zeros((h, w), np.uint8)
    for k, i in enumerate(ada, 1):
        out[lab == i + 1] = k
    return out, len(ada)


# ======================================================================
# 4. SKOR KANDIDAT
# ======================================================================
def _trapesium(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


def skor_kandidat(lab, n, P, prof, roi_box=None):
    """Skor 0..1 (makin besar makin baik) + rincian.

    roi_box = (x, y, w, h) bila ROI manual ada. ROI manual melingkupi SEMUA objek kelas, jadi mask yang benar
    harus (a) membentang hampir seluas ROI dan (b) mengisi ROI cukup padat. Serpihan bagian objek (kaca, plat,
    celana) kontrasnya tinggi sehingga skor lama menyukainya; suku s_roi di bawah yang menghukumnya."""
    if n == 0:
        return 0.0, {}
    h, w = lab.shape
    m = lab > 0
    luas = m.mean()
    s_luas = _trapesium(luas, *prof["luas"])

    # kecocokan batas: rata-rata gradien Sobel di batas mask (di luar tepi gambar) / persentil-90 gradien
    batas = m & ~bkpm.imerode(m, bkpm.strel("diamond", 1))
    batas[:2, :] = batas[-2:, :] = False
    batas[:, :2] = batas[:, -2:] = False
    gref = np.percentile(P["grad"], 90) + 1e-6
    s_fit = float(min(1.0, P["grad"][batas].mean() / gref)) if batas.any() else 0.0

    # penyentuhan tepi gambar: jumlah sisi yang tertutup mask lebih dari ambang_sentuh
    at = prof["ambang_sentuh"]
    sisi = sum([m[0, :].mean() > at, m[-1, :].mean() > at, m[:, 0].mean() > at, m[:, -1].mean() > at])
    s_sentuh = prof["tabel_sentuh"][sisi] if prof["penalti_sentuh"] else 1.0

    # kontras objek terhadap cincin sekitarnya (jarak warna Lab / sebaran): mask yang bocor ke latar
    # atau hanya mengambil sebagian objek akan menghasilkan nilai rendah
    labimg = P["lab"]
    r_ring = bkpm.strel("disk", max(3, int(0.05 * min(h, w))))
    semua = bkpm.imdilate(m, bkpm.strel("disk", 2))
    sep, bsep = [], []
    for i in range(1, n + 1):
        mi = lab == i
        ring = bkpm.imdilate(mi, r_ring) & ~semua
        if ring.sum() < 30 or mi.sum() < 30:
            continue
        pin, pout = labimg[mi], labimg[ring]
        d = np.linalg.norm(pin.mean(0) - pout.mean(0))
        sd = np.sqrt(pin.var(0).mean()) + np.sqrt(pout.var(0).mean()) + 1e-3
        sep.append(min(1.0, d / sd / prof["sep_ref"]))
        bsep.append(mi.sum())
    s_sep = float(np.average(sep, weights=bsep)) if sep else 0.0

    # posisi titik berat terhadap titik acuan kelas
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean() / w, ys.mean() / h
    px, py = prof["pusat"]
    s_pos = max(0.0, 1.0 - np.hypot(cx - px, (cy - py)) / prof["pos_skala"])

    # bentuk tiap instans (circularity & rasio aspek) dibobot luas
    skor_b, bobot = [], []
    for i in range(1, n + 1):
        mi = lab == i
        a = int(mi.sum())
        ys_i, xs_i = np.nonzero(mi)
        bw_, bh_ = xs_i.max() - xs_i.min() + 1, ys_i.max() - ys_i.min() + 1
        rasio = bh_ / bw_                                          # tinggi / lebar
        cir = bkpm.circularity(a, bkpm.perimeter_batas(mi))
        pts = np.stack([xs_i, ys_i], 1).astype(np.int32)
        sol = a / max(1.0, cv2.contourArea(cv2.convexHull(pts)))          # solidity: luas / luas convex hull
        s_sol = float(np.clip((sol - 0.45) / 0.40, 0, 1))
        ok = (_trapesium(rasio, *prof["rasio_tw"]) * 0.40 + _trapesium(cir, *prof["circ"]) * 0.25
              + s_sol * 0.35)
        skor_b.append(ok)
        bobot.append(a)
    s_bentuk = float(np.average(skor_b, weights=bobot))

    # tanah/rumput: mask yang melebar di dasar gambar hampir pasti ikut mengambil tanah (khusus pohon)
    cov_dasar = m[-max(2, int(0.03 * h)):, :].mean()
    s_dasar = 1.0 - float(np.clip((cov_dasar - 0.30) / 0.45, 0, 1))

    wt = prof["bobot"]
    skor = (wt["fit"] * s_fit + wt["sep"] * s_sep + wt["luas"] * s_luas + wt["pos"] * s_pos
            + wt["bentuk"] * s_bentuk + wt["sentuh"] * s_sentuh + wt.get("dasar", 0) * s_dasar) / sum(wt.values())
    skor *= 0.35 + 0.65 * s_bentuk          # gerbang bentuk: pita latar / serpihan tidak boleh menang
    if s_luas == 0.0:                       # luas mustahil untuk kelas ini
        skor *= 0.4
    s_roi = 1.0
    if roi_box is not None:
        rx, ry, rw, rh = roi_box
        ys_a, xs_a = np.nonzero(m)
        # bbox gabungan semua objek vs ROI: IoU kotak (serpihan -> kecil; bocor keluar ROI -> kecil)
        ix = max(0, min(xs_a.max() + 1, rx + rw) - max(xs_a.min(), rx))
        iy = max(0, min(ys_a.max() + 1, ry + rh) - max(ys_a.min(), ry))
        bb = (xs_a.max() - xs_a.min() + 1) * (ys_a.max() - ys_a.min() + 1)
        iou = ix * iy / max(1.0, bb + rw * rh - ix * iy)
        isi = m[ry:ry + rh, rx:rx + rw].sum() / max(1.0, rw * rh)       # seberapa padat ROI terisi
        s_roi = float(min(1.0, iou / 0.80) * _trapesium(isi, *prof.get("isi_roi", (0.10, 0.35, 0.92, 1.01))))
        skor *= 0.15 + 0.85 * s_roi
    return float(skor), {"fit": s_fit, "sep": s_sep, "luas": s_luas, "pos": s_pos, "bentuk": s_bentuk,
                         "sentuh": s_sentuh}


# ======================================================================
# PROSES SATU GAMBAR
# ======================================================================
def segmentasi_gambar(bgr, kelas, roi_manual=None):
    """Segmentasi adaptif satu gambar. roi_manual = (x, y, w, h) piksel bila pengguna menandai ROI
    (Acara 15): area di luar ROI dipakai sebagai referensi latar dan hasil dibatasi di dalam ROI.

    Mengembalikan dict: label (peta instans), n, metode, skor, status, kandidat (nama -> (label, n, skor)),
    praproses P.
    """
    prof = PROFIL[kelas]
    P = praproses(bgr)
    h, w = bgr.shape[:2]
    pita = pita_tepi(h, w, prof["pita"])
    roi_obj = roi_elips(h, w, prof["roi_objek"]) if prof["roi_bentuk"] == "elips" else roi_kotak(h, w, prof["roi_objek"])
    zona = None
    if roi_manual is not None:
        x, y, rw, rh = roi_manual
        zona = np.zeros((h, w), bool)
        zona[y:y + rh, x:x + rw] = True
        roi_obj = zona.copy()
        # referensi latar = 4 pita tipis tepat di luar ROI (latar lokal di sekitar objek), bukan seluruh gambar
        t = max(4, int(0.04 * min(h, w)))
        x, y, rw, rh = roi_manual
        pita = {k: np.zeros((h, w), bool) for k in ("atas", "bawah", "kiri", "kanan")}
        pita["atas"][max(0, y - t):y, x:x + rw] = True
        pita["bawah"][y + rh:y + rh + t, x:x + rw] = True
        pita["kiri"][y:y + rh, max(0, x - t):x] = True
        pita["kanan"][y:y + rh, x + rw:x + rw + t] = True
        pita = {k: v for k, v in pita.items() if v.sum() >= 30}
        if not pita:
            pita = pita_tepi(h, w, prof["pita"])
        # objek diasumsikan di bagian tengah ROI (60%)
        roi_obj = np.zeros((h, w), bool)
        mx, my = int(0.2 * rw), int(0.2 * rh)
        roi_obj[y + my:y + rh - my, x + mx:x + rw - mx] = True
    s = min(h, w)
    sigma_tek = max(2.0, prof["sigma_tekstur"] * s)
    r_tutup = max(2, int(prof["tutup_tepi"] * s))
    zona_jalan = zona if zona is not None else roi_kotak(h, w, prof["zona_aspal"]) if prof.get("zona_aspal") else None
    ukuran_lokal = max(15, int(prof["blok_lokal"] * s)) if prof.get("blok_lokal") else 0

    pembuat = {
        "latar_lab": lambda: kand_latar_lab(P, pita, prof),
        "kanal_terbaik": lambda: kand_kanal_terbaik(P, pita, roi_obj, 1),
        "kanal_terbaik2": lambda: kand_kanal_terbaik(P, pita, roi_obj, 2),
        "latar_baris": lambda: kand_latar_garis(P, "baris"),
        "latar_kolom": lambda: kand_latar_garis(P, "kolom"),
        "gelap_kecil": lambda: kand_gelap_lokal(P, max(15, int(0.08 * s)), 15),
        "tepi_isi": lambda: kand_tepi_isi(P, r_tutup),
        "tekstur": lambda: kand_tekstur(P, sigma_tek, r_tutup),
        "hijau": lambda: kand_hijau(P, False, sigma_tek),
        "hijau_tekstur": lambda: kand_hijau(P, True, sigma_tek),
        "gelap_lokal": lambda: kand_gelap_lokal(P, ukuran_lokal, prof.get("c_lokal", 8)),
        "selisih_aspal": lambda: kand_selisih_aspal(P, zona_jalan),
        "air_cokelat": lambda: kand_air(P, zona_jalan),
        "gelap_atau_air": lambda: bkpm.bw_or(kand_gelap_lokal(P, ukuran_lokal, prof.get("c_lokal", 8)),
                                              kand_air(P, zona_jalan)),
        "gelap_atau_selisih": lambda: bkpm.bw_or(kand_gelap_lokal(P, ukuran_lokal, prof.get("c_lokal", 8)),
                                                  kand_selisih_aspal(P, zona_jalan)),
    }

    hasil = {}
    for nama in prof["kandidat"]:
        try:
            raw = pembuat[nama]()
        except Exception:
            raw = None
        if raw is None:
            continue
        if kelas == "pohon":
            raw = pohon_perbaiki(raw, P, sigma_tek)
        lab, n = pasca(raw, prof, zona if roi_manual is not None else (zona_jalan if prof.get("batasi_zona") else None))
        sk, rinci = skor_kandidat(lab, n, P, prof, roi_manual)
        hasil[nama] = (lab, n, sk, rinci)

    # penyatuan fragmen: objek yang hanya terambil sebagian (spakbor motor, lampu mobil) disatukan dengan
    # closing besar (+ convex hull seluruh fragmen untuk kelas satu-objek) pada 3 kandidat terbaik
    if prof.get("gabung"):
        for nama, (lab0, n0, sk0, _) in sorted(hasil.items(), key=lambda kv: -kv[1][2])[:3]:
            if n0 == 0:
                continue
            m0 = lab0 > 0
            lab_g, n_g = pasca(m0, {**prof, "tutup": max(prof["tutup"], 0.07), "min_rel": 0.0}, zona)
            hasil[nama + "+gabung"] = (lab_g, n_g, *skor_kandidat(lab_g, n_g, P, prof, roi_manual))
            if prof.get("hull"):
                pts = np.argwhere(m0)[:, ::-1].astype(np.int32)
                hm = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(hm, cv2.convexHull(pts), 1)
                lab_h, n_h = pasca(hm > 0, {**prof, "buka": 0.0, "min_rel": 0.0}, zona)
                hasil[nama + "+hull"] = (lab_h, n_h, *skor_kandidat(lab_h, n_h, P, prof, roi_manual))

    # ROI manual: suara dari SEMUA kandidat. Tiap kandidat biasanya hanya benar di sebagian objek (kaca, badan,
    # plat ...), tetapi bagian yang berbeda-beda; piksel yang dipilih >= 1/3 kandidat hampir pasti objek.
    if roi_manual is not None:
        dasar = [v[0] > 0 for k, v in hasil.items() if v[1] > 0 and "+" not in k]
        if len(dasar) >= 3:
            vote = np.sum(dasar, axis=0) >= max(2, int(np.ceil(len(dasar) / 3)))
            for nm, pr in (("fusi_semua", prof), ("fusi_semua+gabung", {**prof, "tutup": max(prof["tutup"], 0.07), "min_rel": 0.0})):
                lab_v, n_v = pasca(vote, pr, zona)
                hasil[nm] = (lab_v, n_v, *skor_kandidat(lab_v, n_v, P, prof, roi_manual))

    # fusi: suara terbanyak (>=2 dari 3) kandidat terbaik - operasi Boolean AND/OR (Acara 8)
    urut = sorted(hasil.items(), key=lambda kv: -kv[1][2])
    if len(urut) >= 3:
        a, b, c = [(kv[1][0] > 0) for kv in urut[:3]]
        vote = (a.astype(np.uint8) + b + c) >= 2
        lab, n = pasca(vote, prof, zona if roi_manual is not None else None)
        sk, rinci = skor_kandidat(lab, n, P, prof, roi_manual)
        hasil["fusi"] = (lab, n, sk, rinci)

    # cadangan ROI pusat (selalu dihitung; hanya menang bila lainnya buruk)
    rp = kand_roi_pusat(h, w, prof) if roi_manual is None else zona
    lab_rp, n_rp = pasca(rp, {**prof, "buka": 0.0, "min_rel": 0.0}, None)
    sk_rp, ri_rp = skor_kandidat(lab_rp, n_rp, P, prof, roi_manual)
    hasil["roi_pusat"] = (lab_rp, n_rp, sk_rp * prof["penalti_roi"], ri_rp)

    terbaik = max(hasil.items(), key=lambda kv: kv[1][2])
    nama, (lab, n, sk, rinci) = terbaik
    if roi_manual is not None:
        status = "roi_manual"
    elif nama == "roi_pusat":
        status = "cadangan_roi"
    elif sk < prof["ambang_cek"]:
        status = "perlu_cek"
    else:
        status = "ok"
    return {"label": lab, "n": n, "metode": nama, "skor": sk, "status": status, "rinci": rinci,
            "kandidat": {k: (v[0], v[1], v[2]) for k, v in hasil.items()}, "P": P}
