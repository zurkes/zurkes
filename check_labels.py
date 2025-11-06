from plyfile import PlyData
import numpy as np
import glob
import os

# Tüm train klasörlerini tara
folders = [
    "data/train/real/sensaturban",
    "data/train/real/stpls3d",
    "data/train/sentetik"
]

all_labels = set()
no_label_files = []
file_labels = {}

for folder in folders:
    if not os.path.exists(folder):
        print(f"[SKIP] {folder} yok")
        continue
    
    files = glob.glob(os.path.join(folder, "*.ply"))
    print(f"\n[{folder}] {len(files)} dosya bulundu")
    
    for f in files:
        try:
            ply = PlyData.read(f)
            v = ply["vertex"].data
            
            # Label field var mı?
            if "label" in v.dtype.names:
                labels = v["label"]
            elif "class" in v.dtype.names:
                labels = v["class"]
            else:
                no_label_files.append(f)
                print(f"  ❌ {os.path.basename(f)} - LABEL YOK!")
                continue
            
            # Unique label'ları topla
            unique = np.unique(labels)
            file_labels[f] = unique
            all_labels.update(unique)
            
        except Exception as e:
            print(f"  ⚠️ {os.path.basename(f)} - HATA: {e}")

print("\n" + "="*60)
print("ÖZET:")
print("="*60)
print(f"Toplam unique label'lar: {sorted(all_labels)}")
print(f"Label olmayan dosyalar: {len(no_label_files)}")

if no_label_files:
    print("\nLabel olmayan dosyalar:")
    for f in no_label_files:
        print(f"  - {f}")

print("\nKlasör bazında label dağılımı:")
for folder in folders:
    folder_files = [k for k in file_labels.keys() if folder in k]
    if folder_files:
        folder_labels = set()
        for f in folder_files:
            folder_labels.update(file_labels[f])
        print(f"  {folder}: {sorted(folder_labels)}")