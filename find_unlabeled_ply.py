#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulduğu şeyler:
- label/class/seg_id alanı OLMAYAN .ply dosyaları (train & val)
- Okuma hatası veren bozuk .ply dosyaları (ör. early end-of-file)

Çıktılar:
- logs/missing_labels_train.txt
- logs/missing_labels_val.txt
- logs/bad_ply_train.txt
- logs/bad_ply_val.txt
- Özet: ekrana yazılır

Kullanım:
  python find_unlabeled_ply.py           # config.yaml'dan yolları okur
  python find_unlabeled_ply.py --root /home/erdem/stpls3d-lite --train data/train --val data/val
"""

import os
import sys
import glob
import argparse
import yaml

from plyfile import PlyData

def read_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_paths(args):
    if args.root and args.train and args.val:
        root = args.root
        train_dir = os.path.join(root, args.train)
        val_dir   = os.path.join(root, args.val)
        logs_dir  = os.path.join(root, "logs")
    else:
        # config.yaml'dan oku
        cfg = read_yaml("config.yaml")
        root = cfg["paths"]["root"]
        train_dir = os.path.join(root, cfg["paths"]["data"]["train"])
        val_dir   = os.path.join(root, cfg["paths"]["data"]["val"])
        logs_dir  = os.path.join(root, cfg["paths"]["logs"])
    os.makedirs(logs_dir, exist_ok=True)
    return root, train_dir, val_dir, logs_dir

def list_ply(folder):
    files = []
    for ext in ("*.ply","*.PLY"):
        files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    # Windows "Zone.Identifier" artıkları atla
    files = [f for f in sorted(files) if not f.endswith(":Zone.Identifier")]
    return files

def has_any_label_field(ply_vertex_dtype_names):
    names = set(ply_vertex_dtype_names)
    return ("label" in names) or ("class" in names) or ("seg_id" in names)

def scan_split(split_dir):
    unlabeled = []  # label/class/seg_id olmayan dosyalar
    broken    = []  # okunamayan/bozuk dosyalar
    files = list_ply(split_dir)

    for f in files:
        try:
            ply = PlyData.read(f)
            if "vertex" not in ply:
                # vertex yoksa zaten eğitimde kullanılamaz
                unlabeled.append(f)
                continue
            v = ply["vertex"].data
            if not has_any_label_field(v.dtype.names):
                unlabeled.append(f)
        except Exception as e:
            # Örn: early end-of-file veya header bozukluğu
            broken.append(f)
    return files, unlabeled, broken

def write_list(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(r + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  type=str, default=None, help="Proje kökü (config.yaml'daki paths.root yerine)")
    ap.add_argument("--train", type=str, default=None, help="Train dizini (paths.data.train yerine)")
    ap.add_argument("--val",   type=str, default=None, help="Val dizini (paths.data.val yerine)")
    args = ap.parse_args()

    root, train_dir, val_dir, logs_dir = resolve_paths(args)

    print("\n===== FIND UNLABELED/ BROKEN PLY: START =====")
    print(f"[ROOT]  {root}")
    print(f"[TRAIN] {train_dir}")
    print(f"[VAL]   {val_dir}")
    print(f"[LOGS]  {logs_dir}\n")

    # TRAIN
    tr_all, tr_unlabeled, tr_broken = scan_split(train_dir)
    # VAL
    va_all, va_unlabeled, va_broken = scan_split(val_dir)

    # Yaz
    missing_tr_txt = os.path.join(logs_dir, "missing_labels_train.txt")
    missing_va_txt = os.path.join(logs_dir, "missing_labels_val.txt")
    bad_tr_txt     = os.path.join(logs_dir, "bad_ply_train.txt")
    bad_va_txt     = os.path.join(logs_dir, "bad_ply_val.txt")

    write_list(missing_tr_txt, tr_unlabeled)
    write_list(missing_va_txt, va_unlabeled)
    write_list(bad_tr_txt, tr_broken)
    write_list(bad_va_txt, va_broken)

    # Özet
    def pct(n, d): return (100.0 * n / d) if d else 0.0
    print("---- TRAIN ----")
    print(f"Toplam .ply           : {len(tr_all)}")
    print(f"Etiketsiz (no label)  : {len(tr_unlabeled)} ({pct(len(tr_unlabeled), len(tr_all)):.2f}%)")
    print(f"Bozuk (read error)    : {len(tr_broken)} ({pct(len(tr_broken), len(tr_all)):.2f}%)")
    print(f"→ Yazıldı: {missing_tr_txt}")
    print(f"→ Yazıldı: {bad_tr_txt}")

    print("\n---- VAL ----")
    print(f"Toplam .ply           : {len(va_all)}")
    print(f"Etiketsiz (no label)  : {len(va_unlabeled)} ({pct(len(va_unlabeled), len(va_all)):.2f}%)")
    print(f"Bozuk (read error)    : {len(va_broken)} ({pct(len(va_broken), len(va_all)):.2f}%)")
    print(f"→ Yazıldı: {missing_va_txt}")
    print(f"→ Yazıldı: {bad_va_txt}")

    print("\n[NOTE] Bu listeleri doğrudan kara-liste (skip) olarak kullanabilirsin.")
    print("===== FIND UNLABELED/ BROKEN PLY: END =====\n")

if __name__ == "__main__":
    main()
