"""TAHAP 6 - DATASET YOLO26, SPLIT TRAIN:VAL:TEST (default 70:15:15, seed 0)

Input : hasil/04_pengayaan (gambar asli + varian + label) dan hasil/05_fitur_bentuk/fitur_bentuk.csv
Output: hasil/06_dataset_yolo26_<train>_<val>_<test>/   (mis. 06_dataset_yolo26_70_15_15)
          Nama folder mengikuti rasio split, jadi tiap preset punya folder sendiri dan TIDAK saling menimpa.
          seg/  images/{train,val,test}  labels/{train,val,test}  data.yaml -> yolo segment, yolo26n-seg.pt
          det/  images/{train,val,test}  labels/{train,val,test}  data.yaml -> yolo detect,  yolo26n.pt
          test_robustness/  images/  labels/   (opsional, --buat-robustness-test)
          laporan_split.csv  audit_split.txt

Kenapa train:val:test, bukan train:val:
  Dataset ini cuma punya 100-300 FOTO ASLI per kelas (yang ribuan itu sebagian besar varian
  augmentasi). Dengan 90:10 train:val saja tidak ada test set yang benar-benar "tidak pernah dilihat"
  sama sekali selama pengembangan -- val kepake berulang buat tuning sehingga ikut "bocor" secara
  tidak langsung. test terpisah dipakai SEKALI di akhir untuk angka final yang dilaporkan.

Aturan supaya tidak ada kebocoran data (lihat README/diskusi split):
  1. Objek yang ditandai terlalu_kecil oleh tahap 5 dibuang; gambar tanpa objek tersisa ikut dibuang.
  2. Yang dipisah adalah KELOMPOK FOTO ASLI (kelas, sumber): foto asli beserta SEMUA variannya selalu
     berada di sisi yang sama (train/val/test), jadi tidak ada varian dari foto test/val yang nyasar
     ke train (atau sebaliknya) -> mAP yang dilaporkan tidak dipalsukan oleh kemiripan varian.
  3. Split per kelas (stratified) menurut JUMLAH FOTO ASLI (bukan total gambar+varian). Rasio default
     70:15:15 dipilih supaya val dan test tetap berisi belasan-puluhan foto per kelas (lihat --preset).
  4. Val DAN test secara default HANYA berisi foto asli (tanpa varian augmentasi) -- augmentasi cuma
     dipakai untuk menambah variasi training, bukan untuk "soal ujian". Matikan proteksi ini dengan
     --val-dengan-varian / --test-dengan-varian (TIDAK disarankan untuk angka yang dilaporkan resmi).
  5. Audit 3-sisi: memastikan tidak ada kelompok foto asli yang berada di lebih dari satu sisi
     (train/val/test), dicek pasangan demi pasangan.
  6. (opsional, --buat-robustness-test) uji ketahanan: membuat varian "goyang" (blur gerak) dan
     "cahaya" (terang/gelap) KHUSUS dari foto test asli saja -> disimpan terpisah di test_robustness/,
     TIDAK pernah masuk data.yaml train/val/test utama. Laporkan skornya terpisah dari mAP test utama
     (mis. "mAP test asli" vs "mAP test + goyang"), karena mensimulasikan kamera rompi yang goyang
     saat pengguna berjalan adalah nilai tambah, bukan pengganti, test set aslinya.

Preset rasio (pilih salah satu lewat --preset, atau atur manual dengan --val dan --test):
  70-15-15  (DEFAULT, disarankan)  - cocok untuk ~100-300 foto asli/kelas: val & test masih berisi
                                     belasan-puluhan foto per kelas sehingga skor per kelas lebih stabil.
  70-20-10  (boleh, BERSYARAT)     - hanya pakai kalau kamu sengaja butuh val lebih besar (mis. untuk
                                     tuning hyperparameter yang lebih teliti) DAN sadar test jadi lebih
                                     kecil (~10%) sehingga kelas yang foto aslinya sedikit lebih gampang
                                     meleset skornya. Karena itu preset ini TIDAK dijadikan default --
                                     lihat 06b_dataset_yolo26_70_20_10.py untuk menjalankannya secara
                                     eksplisit (bukan dengan mengetik --preset setiap kali, supaya jelas
                                     ini pilihan sadar, bukan default yang terpakai tanpa sengaja).
  80-10-10  (boleh, BERSYARAT)     - test paling kecil; cuma masuk akal kalau foto asli per kelas sudah
                                     banyak (ratusan+).

Jalankan:
    python 06_dataset_yolo26.py                       # 70:15:15, seed 0 (default)
    python 06_dataset_yolo26.py --preset 80-10-10 --seed 1
    python 06_dataset_yolo26.py --val 0.2 --test 0.1   # rasio manual, menimpa --preset
    python 06_dataset_yolo26.py --buat-robustness-test --zip
    python 06_dataset_yolo26.py --keluaran hasil/dataset_percobaan_a   # nama folder bebas
"""
import argparse
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import bkpm
from config import CLASSES, D_AUG, D_FITUR, HASIL
from util import baca_csv, baca_gambar, bersihkan, log, simpan_gambar, tabel, tulis_csv

TAHAP = "06_dataset_yolo26"

# Folder keluaran tahap ini. Diisi oleh jalankan() sesuai rasio split (atau --keluaran), sehingga
# preset berbeda otomatis ditulis ke folder berbeda. Fungsi-fungsi di bawah membaca variabel ini.
D_YOLO = HASIL / "06_dataset_yolo26_70_15_15"

# val_frac, test_frac (train = sisanya)
PRESET = {
    "70-15-15": (0.15, 0.15),
    "70-20-10": (0.20, 0.10),
    "80-10-10": (0.10, 0.10),
}
SISI = ("train", "val", "test")


def muat_item():
    man = baca_csv(D_AUG / "manifest_pengayaan.csv")
    fitur = baca_csv(D_FITUR / "fitur_bentuk.csv")
    if not man or not fitur:
        raise SystemExit("jalankan tahap 4 dan 5 dulu")
    buang = defaultdict(set)                                    # file -> id objek yang dibuang
    for f in fitur:
        if f["qc"] != "ok":
            buang[f["file"]].add(int(f["id_objek"]))
    item = []
    for r in man:
        stem = r["file"].rsplit(".", 1)[0]
        asli = (D_AUG / "labels_seg" / f"{stem}.txt").read_text().splitlines()
        det = (D_AUG / "labels_det" / f"{stem}.txt").read_text().splitlines()
        ok = [i for i in range(len(asli)) if i not in buang[r["file"]]]
        if ok:
            item.append({**r, "stem": stem, "seg": [asli[i] for i in ok], "det": [det[i] for i in ok]})
    return item


def split_grup(grup, val_frac, test_frac, seed):
    """Split per KELOMPOK foto asli (kelas, sumber), stratified per kelas, seed tetap -> hasil sama
    tiap dijalankan ulang. Mengembalikan dict (kelas, sumber) -> 'train'/'val'/'test'.

    Jaminan: test >= 1 kelompok bila kelasnya punya >= 2 kelompok, val >= 1 bila masih tersisa
    >= 2 kelompok setelah test, dan train tidak pernah kosong (dikurangi dari val lalu test bila perlu).
    """
    rng = random.Random(seed)
    sisi = {}
    for kelas in CLASSES:
        ids = sorted(s for (k, s) in grup if k == kelas)
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            continue
        n_test = round(test_frac * n)
        n_val = round(val_frac * n)
        if n >= 2:
            n_test = max(1, n_test)
        if n - n_test >= 2:
            n_val = max(1, n_val)
        else:
            n_val = 0
        while n_test + n_val >= n and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break
        for i, s in enumerate(ids):
            if i < n_test:
                sisi[(kelas, s)] = "test"
            elif i < n_test + n_val:
                sisi[(kelas, s)] = "val"
            else:
                sisi[(kelas, s)] = "train"
    return sisi


def tulis_dataset(grup, sisi, val_dengan_varian, test_dengan_varian):
    """Salin gambar+label ke D_YOLO/{seg,det}/images|labels/{train,val,test}. Mengembalikan dict
    jumlah[(kelas, sisi, metrik)] untuk laporan."""
    jumlah = defaultdict(int)
    hanya_asli = {"val": not val_dengan_varian, "test": not test_dengan_varian}
    for (kelas, sumber), anggota in grup.items():
        s = sisi[(kelas, sumber)]
        for it in anggota:
            if s in hanya_asli and hanya_asli[s] and it["varian"] != "asli":
                continue
            for mode in ("seg", "det"):
                dst_i = D_YOLO / mode / "images" / s
                dst_l = D_YOLO / mode / "labels" / s
                dst_i.mkdir(parents=True, exist_ok=True)
                dst_l.mkdir(parents=True, exist_ok=True)
                shutil.copy2(D_AUG / "images" / it["file"], dst_i / it["file"])
                (dst_l / f"{it['stem']}.txt").write_text("\n".join(it[mode]) + "\n")
            jumlah[(kelas, s, "gambar")] += 1
            jumlah[(kelas, s, "objek")] += len(it["seg"])
            jumlah[(kelas, s, "foto_asli")] += it["varian"] == "asli"
    return jumlah


def tulis_data_yaml(val_frac, test_frac, test_robustness):
    nama = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    train_pct = round((1 - val_frac - test_frac) * 100)
    for mode in ("seg", "det"):
        extra = "\ntest_robustness: ../test_robustness/images\n" if test_robustness else ""
        (D_YOLO / mode / "data.yaml").write_text(
            f"# dataset YOLO26 ({'segmentasi polygon' if mode == 'seg' else 'deteksi kotak'}), "
            f"split {train_pct}:{round(val_frac * 100)}:{round(test_frac * 100)} (train:val:test)\n"
            "# 'path' sengaja tidak ditulis: Ultralytics memakai folder data.yaml ini sebagai akar dataset\n"
            "# test_robustness (jika ada) BUKAN bagian dari evaluasi utama -- panggil model.val(split="
            "'test_robustness') terpisah, jangan dicampur dengan angka mAP test.\n"
            f"train: images/train\nval: images/val\ntest: images/test\n{extra}names:\n{nama}\n")


def audit_kebocoran(grup_sisi):
    """grup_sisi: dict (kelas, sumber) -> sisi. Mengecek tidak ada kelompok yang kepeta ke >1 sisi
    (secara konstruksi tidak mungkin karena dict 1 key -> 1 value, tapi tetap dicek ulang per
    pasangan sisi untuk dokumentasi/laporan dan sebagai jaring pengaman kalau logika di atas berubah)."""
    per_sisi = {s: {k for k, v in grup_sisi.items() if v == s} for s in SISI}
    bocor = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        irisan = per_sisi[a] & per_sisi[b]
        if irisan:
            bocor[(a, b)] = irisan
    return per_sisi, bocor


ROBUST_VARIAN = ["goyang", "cahaya_terang", "cahaya_gelap"]


def buat_robustness_test(grup, sisi):
    """Uji ketahanan: varian fotometrik (blur gerak = kamera goyang; terang/gelap) HANYA dari foto
    test ASLI. Disimpan terpisah di D_YOLO/test_robustness/, tidak pernah ikut train/val/test utama."""
    rng = random.Random(0)
    jumlah = 0
    for mode_dir in ("images", "labels"):
        (D_YOLO / "test_robustness" / mode_dir).mkdir(parents=True, exist_ok=True)
    for (kelas, sumber), anggota in grup.items():
        if sisi[(kelas, sumber)] != "test":
            continue
        for it in anggota:
            if it["varian"] != "asli":
                continue
            img = baca_gambar(D_AUG / "images" / it["file"])
            if img is None:
                continue
            for v in ROBUST_VARIAN:
                if v == "goyang":
                    k = bkpm.kernel_gerak(rng.randint(9, 15), rng.uniform(0, 180))
                    im2 = bkpm.konvolusi(img, k)
                elif v == "cahaya_terang":
                    im2 = bkpm.brightness(img, rng.randint(35, 60))
                else:
                    im2 = bkpm.brightness(img, -rng.randint(35, 60))
                nv = f"{it['stem']}__{v}"
                simpan_gambar(D_YOLO / "test_robustness" / "images" / f"{nv}.jpg", im2, 95)
                # label tidak berubah (varian fotometrik, bukan geometrik) -> salin label seg (polygon)
                (D_YOLO / "test_robustness" / "labels" / f"{nv}.txt").write_text(
                    "\n".join(it["seg"]) + "\n")
                jumlah += 1
    return jumlah


def jalankan(preset="70-15-15", val=None, test=None, seed=0, val_dengan_varian=False,
             test_dengan_varian=False, buat_robustness_test_=False, zip_=False,
             _preset_eksplisit=True, keluaran=None):
    """Logika inti tahap 6, dipanggil oleh main() (lewat CLI) MAUPUN langsung oleh skrip preset lain
    seperti 06b_dataset_yolo26_70_20_10.py -- supaya rasio 70:20:10 tidak perlu menduplikasi logika
    split/grouping/audit/copy ini, cukup memanggil fungsi ini dengan val/test yang berbeda."""
    val_frac, test_frac = PRESET[preset]
    if val is not None:
        val_frac = val
    if test is not None:
        test_frac = test
    if val_frac + test_frac >= 1:
        raise SystemExit(f"val+test ({val_frac + test_frac}) harus < 1 supaya train tidak kosong")

    # nama folder mengikuti rasio, mis. 06_dataset_yolo26_70_20_10, kecuali --keluaran diisi
    global D_YOLO
    tr_pct, va_pct, te_pct = round((1 - val_frac - test_frac) * 100), round(val_frac * 100), round(test_frac * 100)
    D_YOLO = Path(keluaran) if keluaran else HASIL / f"06_dataset_yolo26_{tr_pct}_{va_pct}_{te_pct}"
    if not D_YOLO.is_absolute():
        D_YOLO = (Path.cwd() / D_YOLO).resolve()

    item = muat_item()
    grup = defaultdict(list)
    for it in item:
        grup[(it["kelas"], it["sumber"])].append(it)

    sisi = split_grup(grup, val_frac, test_frac, seed)

    bersihkan(D_YOLO)
    jumlah = tulis_dataset(grup, sisi, val_dengan_varian, test_dengan_varian)
    tulis_data_yaml(val_frac, test_frac, buat_robustness_test_)

    n_robust = buat_robustness_test(grup, sisi) if buat_robustness_test_ else 0

    # ---- laporan ----
    lap = []
    for k in CLASSES:
        baris = {"kelas": k}
        for s in SISI:
            baris[f"foto_asli_{s}"] = jumlah[(k, s, "foto_asli")]
            baris[f"gambar_{s}"] = jumlah[(k, s, "gambar")]
            baris[f"objek_{s}"] = jumlah[(k, s, "objek")]
        lap.append(baris)
    kol = ["kelas"] + [f"{m}_{s}" for m in ("foto_asli", "gambar", "objek") for s in SISI]
    tulis_csv(D_YOLO / "laporan_split.csv", lap, kol)
    tabel("Ringkasan split (train:val:test)", lap, kol)

    # ---- audit kebocoran ----
    per_sisi, bocor = audit_kebocoran(sisi)
    n_tr, n_va, n_te = len(per_sisi["train"]), len(per_sisi["val"]), len(per_sisi["test"])
    n_grup = n_tr + n_va + n_te
    g_img_tr = sum(r["gambar_train"] for r in lap)
    g_img_va = sum(r["gambar_val"] for r in lap)
    g_img_te = sum(r["gambar_test"] for r in lap)
    teks_audit = (
        f"preset={tr_pct}-{va_pct}-{te_pct} val_frac={val_frac} test_frac={test_frac} seed={seed}\n"
        f"kelompok foto asli: train={n_tr} val={n_va} test={n_te} (total {n_grup})\n"
        f"rasio foto asli train:val:test = {n_tr / n_grup * 100:.1f}:{n_va / n_grup * 100:.1f}:"
        f"{n_te / n_grup * 100:.1f}\n"
        f"gambar (termasuk varian di train): train={g_img_tr} val={g_img_va} test={g_img_te}\n"
        f"val berisi varian augmentasi: {val_dengan_varian}\n"
        f"test berisi varian augmentasi: {test_dengan_varian}\n"
        f"kelompok bocor (ada di >1 sisi): {sum(len(v) for v in bocor.values())}\n"
        + "".join(f"  {a} & {b}: {sorted(v)}\n" for (a, b), v in bocor.items())
        + (f"robustness test: {n_robust} varian dibuat dari foto test asli -> "
           f"{D_YOLO}/test_robustness/ (laporkan TERPISAH dari mAP test utama)\n"
           if buat_robustness_test_ else "robustness test: tidak dibuat (--buat-robustness-test)\n")
    )
    (D_YOLO / "audit_split.txt").write_text(teks_audit)
    print(teks_audit)
    assert not bocor, "ada kelompok foto asli yang bocor ke lebih dari satu sisi!"
    if preset != "70-15-15" and _preset_eksplisit:
        log(TAHAP, f"PERHATIAN: memakai preset '{preset}', bukan default '70-15-15'. "
                    "Pastikan ini pilihan sadar (lihat syarat di docstring), bukan default pipeline.")

    if zip_:
        z = HASIL / f"{D_YOLO.name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in D_YOLO.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(D_YOLO))
        log(TAHAP, f"zip: {z}")
    log(TAHAP, f"selesai -> {D_YOLO}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESET), default="70-15-15",
                     help="rasio train:val:test (default 70-15-15; 70-20-10 dan 80-10-10 boleh "
                          "dipakai tapi bukan default -- lihat docstring)")
    ap.add_argument("--val", type=float, default=None, help="porsi val manual, menimpa --preset")
    ap.add_argument("--test", type=float, default=None, help="porsi test manual, menimpa --preset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-dengan-varian", action="store_true",
                     help="val ikut berisi varian pengayaan (default: hanya foto asli)")
    ap.add_argument("--test-dengan-varian", action="store_true",
                     help="test ikut berisi varian pengayaan (TIDAK disarankan untuk angka resmi)")
    ap.add_argument("--buat-robustness-test", action="store_true",
                     help="buat varian goyang/cahaya dari foto TEST asli, disimpan terpisah "
                          "(test_robustness/), tidak masuk train/val/test utama")
    ap.add_argument("--zip", action="store_true", help="buat zip untuk Google Colab")
    ap.add_argument("--keluaran", default=None,
                     help="folder keluaran (default: hasil/06_dataset_yolo26_<rasio>, mis. ..._70_15_15)")
    args = ap.parse_args()
    jalankan(preset=args.preset, val=args.val, test=args.test, seed=args.seed,
              val_dengan_varian=args.val_dengan_varian, test_dengan_varian=args.test_dengan_varian,
              buat_robustness_test_=args.buat_robustness_test, zip_=args.zip,
              _preset_eksplisit=(args.val is None and args.test is None), keluaran=args.keluaran)


if __name__ == "__main__":
    main()
