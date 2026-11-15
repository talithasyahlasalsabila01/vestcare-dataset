"""TAHAP 7 (OPSIONAL) - BANDINGKAN YOLO11n vs YOLO26n

Latih dan evaluasi kedua model pada dataset yang sama (pilih dengan --dataset, mis. hasil/06_dataset_yolo26_70_15_15), lalu rangkum
perbandingannya: mAP50, mAP50-95, kecepatan inferensi, jumlah parameter, ukuran file bobot.

Kenapa ditulis seperti ini (lihat juga diskusi split sebelumnya):
  1. Dua model dilatih dengan DATA, EPOCH, UKURAN GAMBAR, DAN SEED YANG SAMA -> perbandingannya adil
     (perbedaan skor karena modelnya, bukan karena kondisi latihan beda).
  2. Evaluasi FINAL dipakai dari TEST SET (bukan val) -- val cuma buat pantau selama latihan; angka
     yang dilaporkan di hasil akhir harus dari data yang modelnya benar-benar belum pernah "lihat".
  3. Kalau <folder dataset>/test_robustness ada (dibuat dengan --buat-robustness-test di
     tahap 6), skornya dihitung TERPISAH dan ditulis di kolom sendiri -- jangan dicampur ke mAP test
     utama (lihat alasan kebocoran/representativitas di tahap 6).
  4. --seeds bisa diisi lebih dari satu (mis. "0,1,2") supaya kelihatan seberapa stabil skornya,
     bukan cuma hasil satu kali latihan yang mungkin kebetulan bagus/jelek.
  5. --cek-cepat menjalankan 1 epoch dulu untuk kedua model sebelum latihan sungguhan -- supaya kalau
     nama checkpoint YOLO26n belum dikenali versi ultralytics yang terpasang, ketahuan dalam hitungan
     menit, bukan setelah menunggu 100 epoch.
  6. Tiap kombinasi (model, seed) dibungkus try/except: kalau salah satu gagal (mis. checkpoint tidak
     ketemu, OOM), yang lain tetap lanjut -- barisnya ditandai status=error, bukan menghentikan
     seluruh perbandingan.

Perlu `pip install -U ultralytics` dulu (lihat requirements-train.txt). Latihan sungguhan butuh GPU;
tanpa GPU skrip tetap jalan tapi sangat lambat (ultralytics otomatis pakai CPU kalau tidak ada GPU).

Jalankan:
    python 07_bandingkan_model.py --cek-cepat                      # sanity check 1 epoch dulu
    python 07_bandingkan_model.py --mode seg --seeds 0,1,2 --epochs 100
    python 07_bandingkan_model.py --mode det --models yolo11n       # satu model saja
    python 07_bandingkan_model.py --mode seg --resume              # lanjutkan training yang terputus
"""
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

from config import D_BANDING, HASIL
from util import log, tabel, tulis_csv

TAHAP = "07_bandingkan_model"
MODEL_DASAR = ["yolo11n", "yolo26n"]


def nama_bobot(base, mode):
    return f"{base}-seg.pt" if mode == "seg" else f"{base}.pt"


def _ambil(d, *kunci, default=None):
    """Ambil nilai pertama yang ada dari beberapa kemungkinan nama kunci (API ultralytics berbeda
    sedikit antar versi/task untuk nama metrik)."""
    for k in kunci:
        if hasattr(d, k):
            v = getattr(d, k)
            if v is not None:
                return v
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def metrik_dari_val(hasil, mode):
    """Ambil mAP50 dan mAP50-95 dari objek hasil model.val(). Untuk model -seg, pakai metrik 'seg'
    (mask); untuk model deteksi biasa pakai 'box'. Mengembalikan (map50, map5095) atau (None, None)
    bila atributnya tidak ditemukan (API ultralytics berubah antar versi)."""
    try:
        grup = hasil.seg if mode == "seg" and hasattr(hasil, "seg") else hasil.box
        return float(grup.map50), float(grup.map)
    except Exception:
        return None, None


def ukuran_file_mb(path):
    path = Path(path)
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else None


def jumlah_parameter_juta(model):
    try:
        return round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3)
    except Exception:
        return None


def satu_run(YOLO, base, mode, seed, args, proyek_dir, d_yolo):
    """Latih + evaluasi satu kombinasi (model, seed). Mengembalikan satu baris dict untuk laporan."""
    nama_run = f"{base}_seed{seed}"
    data_yaml = d_yolo / mode / "data.yaml"
    baris = {"mode": mode, "model": base, "seed": seed, "status": "error"}
    try:
        model = YOLO(nama_bobot(base, mode))
        t0 = time.time()
        ckpt_lanjut = proyek_dir / nama_run / "weights" / "last.pt"
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            seed=seed, patience=args.patience, project=str(proyek_dir), name=nama_run,
            save_period=args.save_period, exist_ok=True, device=args.device or None,
            resume=bool(args.resume and ckpt_lanjut.exists()),
        )
        waktu_latih_menit = round((time.time() - t0) / 60, 1)

        val_akhir = model.val(data=str(data_yaml), split="val")
        test_akhir = model.val(data=str(data_yaml), split="test")
        map50_val, map5095_val = metrik_dari_val(val_akhir, mode)
        map50_test, map5095_test = metrik_dari_val(test_akhir, mode)

        map50_rob = map5095_rob = None
        ada_rob = (d_yolo / "test_robustness" / "images").is_dir() and \
            any((d_yolo / "test_robustness" / "images").iterdir())
        if args.robustness and ada_rob:
            try:
                rob = model.val(data=str(data_yaml), split="test_robustness")
                map50_rob, map5095_rob = metrik_dari_val(rob, mode)
            except Exception as e:
                log(TAHAP, f"  {nama_run}: eval test_robustness gagal ({e}); dilewati, bukan error fatal")

        ms_total = _ambil(getattr(test_akhir, "speed", {}), "inference", default=None)
        bobot_terbaik = proyek_dir / nama_run / "weights" / "best.pt"
        baris.update({
            "status": "ok",
            "epoch_target": args.epochs,
            "waktu_latih_menit": waktu_latih_menit,
            "params_juta": jumlah_parameter_juta(model),
            "ukuran_bobot_mb": ukuran_file_mb(bobot_terbaik),
            "ms_inferensi_per_gambar": round(ms_total, 2) if ms_total is not None else None,
            "map50_val": map50_val, "map50_95_val": map5095_val,
            "map50_test": map50_test, "map50_95_test": map5095_test,
            "map50_test_robustness": map50_rob, "map50_95_test_robustness": map5095_rob,
            "folder_run": str(proyek_dir / nama_run),
        })
    except Exception as e:
        baris["pesan_error"] = str(e)
        log(TAHAP, f"  GAGAL {nama_run}: {e}")
    return baris


def ringkas(baris_semua):
    """Rata-rata +- simpangan baku antar seed, per (mode, model)."""
    import statistics as st
    grup = defaultdict(list)
    for b in baris_semua:
        if b["status"] == "ok":
            grup[(b["mode"], b["model"])].append(b)
    out = []
    kol_angka = ["waktu_latih_menit", "params_juta", "ukuran_bobot_mb", "ms_inferensi_per_gambar",
                 "map50_val", "map50_95_val", "map50_test", "map50_95_test",
                 "map50_test_robustness", "map50_95_test_robustness"]
    for (mode, model), rr in grup.items():
        r = {"mode": mode, "model": model, "n_seed": len(rr)}
        for k in kol_angka:
            vals = [x[k] for x in rr if x.get(k) is not None]
            if vals:
                r[f"{k}_rata"] = round(st.mean(vals), 4)
                r[f"{k}_std"] = round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seg", "det", "both"], default="both")
    ap.add_argument("--models", default=",".join(MODEL_DASAR),
                     help="daftar model dasar dipisah koma, mis. 'yolo11n,yolo26n'")
    ap.add_argument("--seeds", default="0", help="daftar seed dipisah koma, mis. '0,1,2'")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--save-period", type=int, default=10)
    ap.add_argument("--device", default="", help="mis. '0' untuk GPU pertama; kosong = otomatis")
    ap.add_argument("--resume", action="store_true", help="lanjutkan dari last.pt bila ada (sesi putus)")
    ap.add_argument("--robustness", action="store_true",
                     help="ikut evaluasi ke test_robustness bila sudah dibuat di tahap 6")
    ap.add_argument("--cek-cepat", action="store_true",
                     help="sanity check: paksa epochs=1, seeds='0', abaikan --epochs/--seeds")
    ap.add_argument("--dataset", default="06_dataset_yolo26_70_15_15",
                     help="folder dataset hasil tahap 6: nama di bawah hasil/ atau path (default: 06_dataset_yolo26_70_15_15)")
    ap.add_argument("--keluaran", default=None,
                     help="folder output (default: hasil/07_perbandingan_model/<nama dataset>)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "paket 'ultralytics' belum terpasang. Jalankan: pip install -U ultralytics\n"
            "(lihat requirements-train.txt)")

    d_yolo = Path(args.dataset)
    if not d_yolo.exists():
        d_yolo = HASIL / args.dataset
    if not d_yolo.is_dir():
        ada = sorted(p.name for p in HASIL.glob("06_dataset_yolo26*") if p.is_dir())
        raise SystemExit(f"folder dataset '{args.dataset}' tidak ditemukan. Yang tersedia: {ada or 'belum ada'}")
    log(TAHAP, f"dataset: {d_yolo}")
    for mode in ("seg", "det"):
        yaml_mode = d_yolo / mode / "data.yaml"
        if (args.mode in (mode, "both")) and not yaml_mode.exists():
            raise SystemExit(f"{yaml_mode} tidak ada -- jalankan 06_dataset_yolo26.py dulu")

    if args.cek_cepat:
        log(TAHAP, "mode --cek-cepat aktif: epochs dipaksa 1, seeds dipaksa '0'")
        args.epochs, args.seeds = 1, "0"

    modes = ["seg", "det"] if args.mode == "both" else [args.mode]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    proyek_root = Path(args.keluaran) if args.keluaran else D_BANDING / d_yolo.name.replace("06_dataset_yolo26_", "")

    baris_semua = []
    for mode in modes:
        proyek_dir = proyek_root / mode
        for base in models:
            for seed in seeds:
                log(TAHAP, f"latih+eval {base} ({mode}) seed={seed} ...")
                baris_semua.append(satu_run(YOLO, base, mode, seed, args, proyek_dir, d_yolo))

    kol = ["mode", "model", "seed", "status", "epoch_target", "waktu_latih_menit", "params_juta",
           "ukuran_bobot_mb", "ms_inferensi_per_gambar", "map50_val", "map50_95_val",
           "map50_test", "map50_95_test", "map50_test_robustness", "map50_95_test_robustness",
           "folder_run", "pesan_error"]
    proyek_root.mkdir(parents=True, exist_ok=True)
    tulis_csv(proyek_root / "perbandingan_detail.csv", baris_semua, kol)
    tabel("Hasil tiap run (model x seed)", baris_semua,
          ["mode", "model", "seed", "status", "map50_test", "map50_95_test", "ms_inferensi_per_gambar"])

    ring = ringkas(baris_semua)
    if ring:
        kol_r = sorted({k for r in ring for k in r})
        kol_r = ["mode", "model", "n_seed"] + [k for k in kol_r if k not in ("mode", "model", "n_seed")]
        tulis_csv(proyek_root / "perbandingan_ringkasan.csv", ring, kol_r)
        tabel("Ringkasan rata-rata antar seed", ring,
              ["mode", "model", "n_seed", "map50_test_rata", "map50_95_test_rata",
               "ms_inferensi_per_gambar_rata", "params_juta_rata"])

    gagal = [b for b in baris_semua if b["status"] != "ok"]
    if gagal:
        log(TAHAP, f"PERHATIAN: {len(gagal)}/{len(baris_semua)} run gagal -- cek kolom pesan_error "
                    f"di {proyek_root / 'perbandingan_detail.csv'}")
    log(TAHAP, f"selesai -> {proyek_root}")


if __name__ == "__main__":
    main()
