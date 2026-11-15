"""refine_batas.py - penghalusan batas mask gambar ber-ROI manual, HANYA teknik BKPM.

Mask kasar dari seg_core (sudah mengisi ROI) biasanya melebar/menggumpal di batasnya. Batas dirapikan dengan
region growing berbasis kemiripan warna (Acara 15: segmentasi berbasis similaritas), dibatasi tepi Canny
(Acara 31), dan dirapikan dengan morfologi (Acara 18-19):

  1. objek pasti  = erosi mask kasar;   latar pasti = di luar dilasi mask kasar (termasuk di luar ROI)
  2. pita ragu    = selisih dilasi dan erosi mask kasar (di dalam ROI)
  3. rata-rata warna Lab LOKAL objek dan latar dihitung dengan Gaussian ternormalisasi (Acara 28);
     piksel pita masuk objek bila warnanya lebih dekat ke objek lokal daripada latar lokal
  4. tepi Canny menjadi pembatas: piksel tepi tidak ikut tumbuh; hanya yang terhubung ke objek pasti dipertahankan
  5. opening / closing kecil + imfill, simpan komponen besar
  Diulang beberapa kali dengan pita makin sempit.
"""
import numpy as np

import bkpm


def _rata_lokal(lab, mask, sigma):
    w = bkpm.imgaussfilt(mask.astype(np.float32), sigma)
    num = np.stack([bkpm.imgaussfilt(lab[..., c] * mask, sigma) for c in range(3)], -1)
    return num / (w[..., None] + 1e-4), w


def refine(bgr, lab_inst, roi, P, maks_objek=6, putaran=2, bias=1.6):
    h, w = lab_inst.shape
    s = min(h, w)
    x, y, rw, rh = roi
    zona = np.zeros((h, w), bool)
    zona[y:y + rh, x:x + rw] = True
    m = (lab_inst > 0) & zona
    if not m.any():
        return lab_inst
    lab = P["lab"].astype(np.float32)
    tepi = bkpm.imdilate(P["tepi"], bkpm.strel("disk", 1))
    sigma = max(5.0, 0.10 * s)
    r = max(3, int(0.035 * s))
    for _ in range(putaran):
        pasti_o = bkpm.imerode(m, bkpm.strel("disk", r))
        if pasti_o.sum() < 50:                      # objek terlalu tipis untuk erosi sebesar itu
            pasti_o = bkpm.imerode(m, bkpm.strel("disk", max(1, r // 3)))
        luar = ~bkpm.imdilate(m, bkpm.strel("disk", r))
        pita = bkpm.imdilate(m, bkpm.strel("disk", r)) & ~pasti_o & zona
        mo, wo = _rata_lokal(lab, pasti_o, sigma)
        ml, wl = _rata_lokal(lab, luar, sigma)
        d_o = np.linalg.norm(lab - mo, axis=2)
        d_l = np.linalg.norm(lab - ml, axis=2)
        ada = (wo > 1e-3) & (wl > 1e-3)
        # konservatif: keputusan hanya diubah bila selisih jaraknya jelas (faktor bias); selain itu pertahankan mask kasar
        jelas_o = ada & (d_o * bias < d_l)
        jelas_l = ada & (d_l * bias < d_o)
        tumbuh = pasti_o | (pita & ((m & ~jelas_l) | (jelas_o & ~tepi)))
        # region growing: hanya yang terhubung ke objek pasti
        n, lb, st = bkpm.bwlabeln(tumbuh, 8)
        keep = np.zeros_like(tumbuh)
        for i in range(1, n + 1):
            if (pasti_o & (lb == i)).any():
                keep |= lb == i
        keep = bkpm.imopen(keep, bkpm.strel("disk", 2))
        keep = bkpm.imclose(keep, bkpm.strel("disk", 3))
        keep = bkpm.imfill(keep) & zona
        m = keep
        r = max(2, int(r * 0.6))
    n, lb, st = bkpm.bwlabeln(m, 8)
    if n == 0:
        return lab_inst
    area = st[:, 4]
    ada = [i for i in range(n) if area[i] >= max(100, 0.15 * area.max())]
    ada.sort(key=lambda i: -area[i])
    out = np.zeros((h, w), np.uint8)
    for k, i in enumerate(ada[:maks_objek], 1):
        out[lb == i + 1] = k
    return out
