import os
from plyfile import PlyData
from collections import Counter

# === DÜZENLEME GEREKMEZ ===
ply_dir = "/home/erdem/stpls3d-lite/datasets/sensaturban/train"
label_counts = Counter()

print(f"\n🔍 Tarama başlıyor: {ply_dir}\n")

for fname in os.listdir(ply_dir):
    if fname.endswith(".ply"):
        fpath = os.path.join(ply_dir, fname)
        try:
            plydata = PlyData.read(fpath)
            labels = plydata['vertex']['class']
            label_counts.update(labels)
            print(f"✓ {fname} — OK (n={len(labels)} noktadan okundu)")
        except Exception as e:
            print(f"⚠️ {fname} — HATA: {e}")

# === SONUÇLAR ===
print("\n📊 Etiket Dağılımı (Sınıf ID : Nokta Sayısı):")
for label_id, count in sorted(label_counts.items()):
    print(f"  Sınıf {label_id:>2} : {count:,} nokta")
