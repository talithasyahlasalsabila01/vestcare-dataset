"""yolo_util.py - konversi peta instans <-> label YOLO (segmentasi polygon dan deteksi kotak)."""
import cv2
import numpy as np

import bkpm

WARNA = [(0, 255, 0), (0, 200, 255), (255, 120, 0), (255, 0, 200), (0, 0, 255), (200, 255, 0)]


def instans_ke_polygon(lab, epsilon=0.003, min_px=100):
    """Peta instans -> daftar polygon piksel (Nx2). Satu instans = kontur luar terbesar (Douglas-Peucker
    menyederhanakan titik). Instans yang luasnya < min_px (aturan area >= 100 px, BKPM Acara 23) dibuang."""
    hasil = []
    for i in range(1, int(lab.max()) + 1):
        m = (lab == i).astype(np.uint8)
        if m.sum() < min_px:
            continue
        k, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not k:
            continue
        c = max(k, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True)[:, 0, :]
        if len(c) >= 3:
            hasil.append((i, c.astype(np.float32)))
    return hasil


def baris_seg(kelas_id, poly, w, h):
    p = poly / [w, h]
    p = np.clip(p, 0, 1)
    return f"{kelas_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p)


def baris_det(kelas_id, poly, w, h):
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
    return f"{kelas_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def tulis_label(path_seg, path_det, kelas_id, polys, w, h):
    path_seg.parent.mkdir(parents=True, exist_ok=True)
    path_det.parent.mkdir(parents=True, exist_ok=True)
    path_seg.write_text("\n".join(baris_seg(kelas_id, p, w, h) for _, p in polys) + "\n")
    path_det.write_text("\n".join(baris_det(kelas_id, p, w, h) for _, p in polys) + "\n")


def baca_label_seg(path):
    """Baca label polygon -> list (id_kelas, array Nx2 relatif)."""
    out = []
    for baris in path.read_text().splitlines():
        t = baris.split()
        if len(t) >= 7:
            out.append((int(t[0]), np.array(t[1:], np.float32).reshape(-1, 2)))
    return out


def gambar_berlabel(img, polys, nama_kelas):
    out = img.copy()
    for k, (i, p) in enumerate(polys):
        w = WARNA[k % len(WARNA)]
        pi = p.astype(np.int32)
        ov = out.copy()
        cv2.fillPoly(ov, [pi], w)
        out = cv2.addWeighted(ov, 0.30, out, 0.70, 0)
        cv2.polylines(out, [pi], True, w, 2)
        x0, y0 = pi.min(0)
        x1, y1 = pi.max(0)
        cv2.rectangle(out, (x0, y0), (x1, y1), w, 1)
        cv2.putText(out, nama_kelas, (x0 + 2, max(12, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out
