import os, glob, json, yaml, sys
import numpy as np
from collections import Counter
from plyfile import PlyData

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

try:
    cfg = yaml.safe_load(open("config.yaml", "r"))
    mapping = json.load(open("class_mapping.json", "r"))
except Exception as e:
    fail(f"Konfig okuma hatası: {e}")

root = cfg["paths"]["root"]
train_dir = os.path.join(root, cfg["paths"]["data"]["train"])

def dataset_of(path:str)->str:
    p = path.lower()
    if "sensaturban" in p: return "sensaturban"
    if "stpls" in p or "stpls3d" in p: return "stpls3d"
    return "stpls3d"

merge_stpls = {k:set(v) for k,v in mapping["merge_source_ids"]["stpls3d"].items()}
merge_sensat= {k:set(v) for k,v in mapping["merge_source_ids"]["sensaturban"].items()}
ignore_stpls= set(mapping["ignore_ids"]["stpls3d"])
ignore_sensat=set(mapping["ignore_ids"]["sensaturban"])

def map_label(raw_id: int, src:str):
    if src=="sensaturban":
        if raw_id in ignore_sensat: return "IGNORED"
        for target_name, src_ids in merge_sensat.items():
            if raw_id in src_ids: return target_name
        return "UNKNOWN"
    else:
        if raw_id in ignore_stpls: return "IGNORED"
        for target_name, src_ids in merge_stpls.items():
            if raw_id in src_ids: return target_name
        return "UNKNOWN"

cands = []
for ext in ("*.ply","*.PLY"):
    cands += glob.glob(os.path.join(train_dir, "**", ext), recursive=True)
if not cands:
    fail(f"Train altında PLY bulunamadı: {train_dir}")

ply_path = sorted([c for c in cands if not c.endswith(':Zone.Identifier')])[0]

try:
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
except Exception as e:
    fail(f"PLY okuma hatası: {e}")

def col(name, alt=None):
    if name in v.dtype.names: return v[name]
    if alt:
        if isinstance(alt, (list, tuple)):
            for a in alt:
                if a in v.dtype.names: return v[a]
        else:
            if alt in v.dtype.names: return v[alt]
    fail(f"Alan yok: {name}")

x,y,z = col("x"), col("y"), col("z")
r = col("red", alt=None) if "red" in v.dtype.names else np.zeros_like(x, dtype=np.uint8)
g = col("green", alt=None) if "green" in v.dtype.names else np.zeros_like(x, dtype=np.uint8)
b = col("blue", alt=None) if "blue" in v.dtype.names else np.zeros_like(x, dtype=np.uint8)

label = None
for cand in ("label","class","seg_id"):
    if cand in v.dtype.names:
        label = col(cand)
        label_src = cand
        break
if label is None:
    fail("label/class/seg_id alanı bulunamadı")

n = len(x)
for arr in (y,z,r,g,b,label):
    if len(arr) != n:
        fail("Sütun uzunlukları eşit değil")

u, c = np.unique(label, return_counts=True)
raw_hist = dict(zip(u.astype(int).tolist(), c.tolist()))

src = dataset_of(ply_path)
mapped = [map_label(int(l), src) for l in label]
cnt = Counter(mapped)
ignored = cnt.get("IGNORED", 0)

print("\n===== SMOKE SUMMARY: START =====")
print(f"PLY_Path: {ply_path}")
print(f"label_source: {label_src}")
print(f"Points: {n:,}")
print(f"Raw_Label_Hist: {raw_hist}")
print("Mapped_Distribution:", {k:int(v) for k,v in cnt.items()})
print(f"Ignored_Count: {ignored}")
print("===== SMOKE SUMMARY: END =====\n")