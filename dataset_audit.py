import os, glob, json, yaml, csv, random
import numpy as np
from plyfile import PlyData, PlyElement

random.seed(42)

cfg = yaml.safe_load(open("config.yaml", "r"))
root = cfg["paths"]["root"]
train_dir = os.path.join(root, cfg["paths"]["data"]["train"])
val_dir   = os.path.join(root, cfg["paths"]["data"]["val"])
logs_dir  = os.path.join(root, cfg["paths"]["logs"])
badlist   = cfg["paths"].get("badlist", None)
os.makedirs(logs_dir, exist_ok=True)

mapping = json.load(open("class_mapping.json", "r"))
selected = ["Ground","Vegetation","Road","Vehicle","Building","Clutter"]

merge_stpls = {k:set(v) for k,v in mapping["merge_source_ids"]["stpls3d"].items()}
merge_sensat= {k:set(v) for k,v in mapping["merge_source_ids"]["sensaturban"].items()}
ignore_stpls= set(mapping["ignore_ids"]["stpls3d"])
ignore_sensat=set(mapping["ignore_ids"]["sensaturban"])

def dataset_of(path:str)->str:
    p = path.lower()
    if "sensaturban" in p: return "sensaturban"
    if "stpls" in p or "stpls3d" in p: return "stpls3d"
    return "stpls3d"

def map_label(raw_id:int, src:str):
    if src=="sensaturban":
        if raw_id in ignore_sensat: return "IGNORED"
        for tgt, srcs in merge_sensat.items():
            if raw_id in srcs: return tgt
        return "UNKNOWN"
    else:
        if raw_id in ignore_stpls: return "IGNORED"
        for tgt, srcs in merge_stpls.items():
            if raw_id in srcs: return tgt
        return "UNKNOWN"

def read_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    need = ["x","y","z"]
    missing = [k for k in need if k not in v.dtype.names]
    if "label" not in v.dtype.names and "class" not in v.dtype.names and "seg_id" not in v.dtype.names:
        missing.append("label/class/seg_id")
    if missing:
        raise ValueError(f"{os.path.basename(path)} eksik alanlar: {missing}")
    if "label" in v.dtype.names: label = v["label"]
    elif "class" in v.dtype.names: label = v["class"]
    else: label = v["seg_id"]
    # RGB opsiyonel
    if all(k in v.dtype.names for k in ("red","green","blue")):
        rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.float32)
    else:
        rgb = np.zeros((len(v["x"]),3), dtype=np.float32)
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return xyz, rgb, label.astype(np.int64)

def collect_files(folder):
    files=[]
    for ext in ("*.ply","*.PLY"):
        files += glob.glob(os.path.join(folder,"**",ext), recursive=True)
    files = [f for f in sorted(files) if not f.endswith(":Zone.Identifier")]
    if badlist and os.path.exists(os.path.join(root, badlist)):
        skipset = set(open(os.path.join(root, badlist)).read().splitlines())
        files = [f for f in files if f not in skipset]
    return files

def per_class_stats(xyz, rgb, mapped):
    stats={}
    arr = np.array(mapped)
    for name in set(arr):
        mask = (arr==name)
        if mask.sum()==0: continue
        z = xyz[mask,2]
        r,g,b = rgb[mask,0], rgb[mask,1], rgb[mask,2]
        stats[name] = {
            "count": int(mask.sum()),
            "z_min": float(z.min()), "z_max": float(z.max()),
            "R_min": int(r.min()), "R_max": int(r.max()),
            "G_min": int(g.min()), "G_max": int(g.max()),
            "B_min": int(b.min()), "B_max": int(b.max()),
        }
    return stats

def write_samples(xyz, rgb, raw_label, mapped, out_dir, per_class_n=20000):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.array(mapped)
    for name in set(arr):
        if name in ("IGNORED","UNKNOWN"): continue
        mask = np.where(arr==name)[0]
        if len(mask)==0: continue
        idx = mask if len(mask)<=per_class_n else np.random.choice(mask, per_class_n, replace=False)
        pts = xyz[idx]; col = rgb[idx].astype(np.uint8); lab = raw_label[idx].astype(np.int32)

        verts = np.empty(len(idx), dtype=[('x','f4'),('y','f4'),('z','f4'),
                                          ('red','u1'),('green','u1'),('blue','u1'),
                                          ('label','i4')])
        verts['x']=pts[:,0]; verts['y']=pts[:,1]; verts['z']=pts[:,2]
        verts['red']=col[:,0]; verts['green']=col[:,1]; verts['blue']=col[:,2]
        verts['label']=lab
        el = PlyElement.describe(verts, 'vertex')
        PlyData([el], text=False).write(os.path.join(out_dir, f"sample_{name}.ply"))

def audit_split(split_dir, tag):
    files = collect_files(split_dir)
    assert files, f"{tag} içinde PLY bulunamadı: {split_dir}"
    total_stats = {}
    bad_ids = set()

    first = files[0]
    xyz, rgb, lab = read_ply(first)
    src = dataset_of(first)
    mapped = [map_label(int(i), src) for i in lab]
    s = per_class_stats(xyz, rgb, mapped)
    out_samples = os.path.join(logs_dir, f"samples_{tag}")
    write_samples(xyz, rgb, lab, mapped, out_samples, per_class_n=20000)

    counts = {}
    total = 0
    for p in files:
        _, _, L = read_ply(p)
        u, c = np.unique(L, return_counts=True)
        src = dataset_of(p)
        for rid, cc in zip(u, c):
            name = map_label(int(rid), src)
            counts[name] = counts.get(name,0) + int(cc)
            total += int(cc)
            if name in ("UNKNOWN",): bad_ids.add(int(rid))

    total_stats["per_class"] = {k:int(v) for k,v in counts.items()}
    total_stats["detail_first_file"] = s
    total_stats["bad_raw_ids"] = sorted(list(bad_ids))
    total_stats["total_points"] = int(total)
    return total_stats

def save_reports(report, tag):
    jf = os.path.join(logs_dir, f"dataset_audit_{tag}.json")
    json.dump(report, open(jf,"w"), indent=2)
    cf = os.path.join(logs_dir, f"dataset_audit_{tag}.csv")
    import csv
    with open(cf,"w",newline="") as f:
        w=csv.writer(f, delimiter=',')
        w.writerow(["Class","Count"])
        for k,v in sorted(report["per_class"].items()):
            w.writerow([k,v])
    return jf, cf

print("\n===== DATASET AUDIT SUMMARY: START =====")
tr = audit_split(train_dir, "train")
va = audit_split(val_dir, "val")
jf_tr, cf_tr = save_reports(tr, "train")
jf_va, cf_va = save_reports(va, "val")

print("[TRAIN] counts:", tr["per_class"])
print("[VAL]   counts:", va["per_class"])
print("[TRAIN] first-file stats:", tr["detail_first_file"])
print("[VAL]   first-file stats:", va["detail_first_file"])
print("[WARN] unknown raw ids (if any):", tr["bad_raw_ids"], va["bad_raw_ids"])
print(f"[OUT] reports: {jf_tr}, {cf_tr} | {jf_va}, {cf_va}")
print("[OUT] samples for CC:", os.path.join(logs_dir,"samples_train"), os.path.join(logs_dir,"samples_val"))
print("===== DATASET AUDIT SUMMARY: END =====\n")