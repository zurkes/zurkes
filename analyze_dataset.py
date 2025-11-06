#!/usr/bin/env python3
"""
STPLS3D Dataset Analyzer
Her dosyadaki class dağılımını analiz eder
"""

import numpy as np
import os
from pathlib import Path
import json

# Class order (train.py'den)
CLASS_ORDER = ["Ground", "Vegetation", "Road", "Vehicle", "Building", "Clutter"]

def read_ply(path):
    """PLY dosyasını oku"""
    try:
        with open(path, 'rb') as f:
            # Header'ı oku
            line = f.readline().decode('ascii').strip()
            if line != 'ply':
                return None, None, None
            
            # Vertex sayısını bul
            num_vertices = 0
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line == 'end_header':
                    break
            
            # Binary data'yı oku (x,y,z,r,g,b,label = 7 floats)
            dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', '<f4'), ('g', '<f4'), ('b', '<f4'),
                ('label', '<f4')
            ])
            
            data = np.fromfile(f, dtype=dtype, count=num_vertices)
            
            XYZ = np.column_stack([data['x'], data['y'], data['z']])
            RGB = np.column_stack([data['r'], data['g'], data['b']])
            Y = data['label'].astype(np.int32)
            
            return XYZ, RGB, Y
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None, None

def analyze_file(filepath):
    """Tek bir dosyayı analiz et"""
    XYZ, RGB, Y = read_ply(filepath)
    
    if Y is None:
        return None
    
    # -100 olan label'ları filtrele
    valid_mask = (Y >= 0)
    Y_valid = Y[valid_mask]
    
    # Class dağılımı
    unique, counts = np.unique(Y_valid, return_counts=True)
    
    class_dist = {i: 0 for i in range(len(CLASS_ORDER))}
    for cls_idx, count in zip(unique, counts):
        if cls_idx < len(CLASS_ORDER):
            class_dist[cls_idx] = int(count)
    
    total_points = len(Y_valid)
    
    return {
        'filename': os.path.basename(filepath),
        'total_points': total_points,
        'class_counts': class_dist,
        'class_percentages': {i: (count/total_points*100 if total_points>0 else 0) 
                             for i, count in class_dist.items()}
    }

def main():
    # Train ve Val dosyalarını topla
    base_dir = Path.home() / "stpls3d-lite" / "data"
    
    print("=" * 80)
    print("STPLS3D DATASET ANALYSIS")
    print("=" * 80)
    
    # Tüm .ply dosyalarını bul (recursive)
    train_files = list((base_dir / "train").glob("**/*.ply"))
    val_files = list((base_dir / "val").glob("*.ply"))
    ply_files = train_files + val_files
    
    if not ply_files:
        print(f"No .ply files found in {base_dir}")
        return
    
    print(f"\nFound {len(ply_files)} files ({len(train_files)} train, {len(val_files)} val)\n")
    
    results = []
    
    for ply_file in sorted(ply_files):
        print(f"Analyzing: {ply_file.name}...", end=" ")
        result = analyze_file(str(ply_file))
        
        if result:
            results.append(result)
            print("✓")
        else:
            print("✗ (failed)")
    
    # Sonuçları yazdır
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Her dosya için detay
    for res in results:
        print(f"\n📄 {res['filename']}")
        print(f"   Total points: {res['total_points']:,}")
        print(f"   Class distribution:")
        
        for i, class_name in enumerate(CLASS_ORDER):
            count = res['class_counts'][i]
            pct = res['class_percentages'][i]
            
            if count > 0:
                bar = "█" * int(pct / 2)  # Bar chart
                print(f"      {class_name:12s}: {count:8,} ({pct:5.2f}%) {bar}")
    
    # Özet istatistikler
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Her class için toplam
    class_totals = {i: 0 for i in range(len(CLASS_ORDER))}
    files_with_class = {i: [] for i in range(len(CLASS_ORDER))}
    
    for res in results:
        for i in range(len(CLASS_ORDER)):
            count = res['class_counts'][i]
            if count > 0:
                class_totals[i] += count
                files_with_class[i].append(res['filename'])
    
    print("\nClass totals across all files:")
    for i, class_name in enumerate(CLASS_ORDER):
        total = class_totals[i]
        num_files = len(files_with_class[i])
        print(f"  {class_name:12s}: {total:10,} points in {num_files:2d} files")
    
    # Vehicle ve Clutter içeren dosyaları göster
    print("\n" + "=" * 80)
    print("FILES WITH VEHICLE:")
    vehicle_idx = CLASS_ORDER.index("Vehicle")
    if files_with_class[vehicle_idx]:
        for fname in sorted(files_with_class[vehicle_idx]):
            print(f"  ✓ {fname}")
    else:
        print("  ✗ No files with Vehicle class!")
    
    print("\nFILES WITH CLUTTER:")
    clutter_idx = CLASS_ORDER.index("Clutter")
    if files_with_class[clutter_idx]:
        for fname in sorted(files_with_class[clutter_idx]):
            print(f"  ✓ {fname}")
    else:
        print("  ✗ No files with Clutter class!")
    
    # JSON olarak kaydet
    output_file = Path.home() / "stpls3d-lite" / "dataset_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'files': results,
            'summary': {
                'class_totals': class_totals,
                'files_with_class': files_with_class
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()