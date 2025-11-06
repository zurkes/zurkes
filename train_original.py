import os, sys, json, yaml, glob, random, numpy as np
import torch as th
import torch.nn as nn
import time
from math import pi, cos
from plyfile import PlyData

# ==== 0) CLI flags ====
cli_args = " ".join(sys.argv[1:]).lower()
NO_RESUME = ("--no-resume" in cli_args) or ("--resume false" in cli_args) or ("--resume=false" in cli_args)

# ==== 1) Config ve ortam ====
cfg = yaml.safe_load(open("config.yaml"))
ROOT = cfg["paths"]["root"]
TR = os.path.join(ROOT, cfg["paths"]["data"]["train"])
VA = os.path.join(ROOT, cfg["paths"]["data"]["val"])
CKPT = os.path.join(ROOT, cfg["paths"]["checkpoints"])
LOGS = os.path.join(ROOT, cfg["paths"]["logs"])
os.makedirs(CKPT, exist_ok=True); os.makedirs(LOGS, exist_ok=True)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
seed = int(cfg["project"]["seed"])
th.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# ==== 1.b) Hız / rotasyon / denge ayarları (config'ten override edilebilir) ====
FILES_PER_EPOCH      = int(cfg.get("training", {}).get("files_per_epoch", 15))
FILE_STEP            = int(cfg.get("training", {}).get("file_step", 17))
MAX_POINTS_PER_FILE  = int(cfg.get("training", {}).get("max_points_per_file", 500_000))
MIN_REAL_PER_EPOCH   = int(cfg.get("training", {}).get("min_real_per_epoch", 2))
REAL_OVERSAMPLE      = int(cfg.get("training", {}).get("real_oversample", 10))
RESUME_DEFAULT       = bool(cfg.get("training", {}).get("resume", True))
RESUME               = RESUME_DEFAULT and (not NO_RESUME)

# ==== 2) Mapping ve class weights ====
mapping = json.load(open("class_mapping.json"))
order = ["Ground","Vegetation","Road","Vehicle","Building","Clutter"]
name_to_id = {n:i for i,n in enumerate(order)}

# Dataset-aware mapping
merge_stpls3d = {k: set(v) for k, v in mapping["merge_source_ids"]["stpls3d"].items()}
merge_sensaturban = {k: set(v) for k, v in mapping["merge_source_ids"]["sensaturban"].items()}
ignore_stpls3d = set(mapping["ignore_ids"]["stpls3d"])
ignore_sensaturban = set(mapping["ignore_ids"]["sensaturban"])

if os.path.exists("weights.json"):
    W = json.load(open("weights.json"))
    class_weights = th.tensor(W["weights_norm"], dtype=th.float32, device=device)
else:
    class_weights = th.ones(len(order), dtype=th.float32, device=device)

def map_id(raw, is_sensaturban=False):
    """Map raw label to target class ID based on dataset"""
    if is_sensaturban:
        if raw in ignore_sensaturban: 
            return None
        for k, src in merge_sensaturban.items():
            if raw in src: 
                return name_to_id[k]
    else:  # STPLS3D
        if raw in ignore_stpls3d: 
            return None
        for k, src in merge_stpls3d.items():
            if raw in src: 
                return name_to_id[k]
    return None

# ==== 3) I/O yardımcıları ====
def read_ply(path):
    # Dataset detection
    path_lower = path.lower()
    is_sensaturban = any(x in path_lower for x in ['birmingham', 'cambridge'])
    
    v = PlyData.read(path)["vertex"].data
    x,y,z = v["x"], v["y"], v["z"]
    r,g,b = v["red"], v["green"], v["blue"]
    label = v["label"] if "label" in v.dtype.names else v["class"]
    mapped = [map_id(int(L), is_sensaturban=is_sensaturban) for L in label]
    mask = np.array([m is not None for m in mapped])
    XYZ = np.stack([x,y,z],1).astype(np.float32)[mask]
    RGB = np.stack([r,g,b],1).astype(np.float32)[mask]/255.0
    Y = np.array([m for m in mapped if m is not None], dtype=np.int64)
    return XYZ, RGB, Y

def voxel_downsample(xyz, rgb, y, voxel=0.10):
    key = np.floor(xyz/voxel).astype(np.int64)
    h = key[:,0]*73856093 ^ key[:,1]*19349663 ^ key[:,2]*83492791
    _, idx = np.unique(h, return_index=True)
    return xyz[idx], rgb[idx], y[idx]

def list_plys(folder):
    files = []
    for ext in ("*.ply","*.PLY"):
        files += glob.glob(os.path.join(folder, "**", ext), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No PLY in {folder}")
    return files

def is_real_file(path):
    name = os.path.basename(path).lower()
    if ("realworlddata" in path.lower()) or any(k in name for k in ["occc", "usc", "wmsc", "ra_points"]):
        return True
    return False

def augment_xyz(xyz):
    # Z ekseni etrafında rotasyon
    ang = np.random.uniform(-np.pi, np.pi)
    cz, sz = np.cos(ang), np.sin(ang)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    xyz = xyz @ Rz.T
    # Küçük ölçekleme
    scale = np.random.uniform(0.95, 1.05)
    xyz = xyz * scale
    # Jitter
    xyz = xyz + np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
    return xyz

# ==== 4) Veri yükleme (çoklu dosya + real dengeleme) ====
tr_files_all = list_plys(TR)
va_files = list_plys(VA)
vox = cfg["dataset"]["voxel"]["size_m"]

# real / synthetic ayır
real_files = [p for p in tr_files_all if is_real_file(p)]
syn_files  = [p for p in tr_files_all if not is_real_file(p)]

# oversample real
tr_files = syn_files + real_files * max(1, REAL_OVERSAMPLE)

print(f"[INFO] Train files: total={len(tr_files)} (syn={len(syn_files)} + real*{REAL_OVERSAMPLE}={len(real_files)*max(1,REAL_OVERSAMPLE)}) "
      f"| unique={len(tr_files_all)} | Val files: {len(va_files)} | Voxel={vox} m")
if len(real_files)==0:
    print("[WARN] No REAL files detected in train set; check paths/names.")

# ==== 5) Modeller ====
class TinyMLP(nn.Module):
    def __init__(self, in_ch=6, n_cls=len(order), hid=64, layers=2, dropout=0.0):
        super().__init__()
        blocks = []
        last = in_ch
        for _ in range(layers):
            blocks += [nn.Linear(last, hid), nn.ReLU(True)]
            if dropout>0: blocks += [nn.Dropout(dropout)]
            last = hid
        blocks += [nn.Linear(last, n_cls)]
        self.net = nn.Sequential(*blocks)
    def forward(self, x):  # x: [N, in_ch]
        return self.net(x)

class RandLALite(nn.Module):
    """'lite' RandLA: batch içi kNN, mean/max + relative (neigh - x) havuzu, ardından MLP."""
    def __init__(self, in_ch=6, n_cls=len(order), k=32, dropout=0.20, hid=128):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_ch * 5, hid), nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hid, hid), nn.ReLU(True),
            nn.Linear(hid, n_cls)
        )

    @th.no_grad()
    def _knn_idx(self, coords):
        N = coords.size(0)
        k_eff = max(1, min(self.k, N - 1))  # dinamik k
        d = th.cdist(coords, coords)
        d.fill_diagonal_(float('inf'))
        _, idx = th.topk(-d, k_eff, dim=1)  # k_eff kullan
        return idx

    def forward(self, x):
        coords = x[:, :3]
        idx = self._knn_idx(coords)
        neigh = x[idx, :]                     # [N,k,C]
        rel = neigh - x.unsqueeze(1)          # [N,k,C]
        mean_feat = neigh.mean(dim=1)
        max_feat, _ = neigh.max(dim=1)
        mean_rel = rel.mean(dim=1)
        max_rel, _ = rel.max(dim=1)
        fused = th.cat([x, mean_feat, max_feat, mean_rel, max_rel], dim=1)  # [N,5*C]
        return self.mlp(fused)

# === Model seçimi (config.yaml'dan) ===
model_name = cfg["model"]["name"].lower()
if model_name == "randla_like":
    k = int(cfg.get("neighbors", {}).get("k", 32))
    dropout = float(cfg["model"].get("dropout", 0.20))
    hid = int(cfg["model"].get("hidden", 128))
    model = RandLALite(in_ch=6, n_cls=len(order), k=k, dropout=dropout, hid=hid).to(device)
else:
    hid = int(cfg["model"].get("hidden", 64))
    layers = int(cfg["model"].get("layers", 2))
    dropout = float(cfg["model"].get("dropout", 0.0))
    model = TinyMLP(in_ch=6, n_cls=len(order), hid=hid, layers=layers, dropout=dropout).to(device)

base_lr = float(cfg["training"]["lr"])
opt = th.optim.Adam(model.parameters(), lr=base_lr)
crit = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

# AMP + GradScaler (yeni API)
use_amp = bool(cfg["model"].get("amp", True))
scaler = th.amp.GradScaler('cuda', enabled=use_amp)

# ==== LR Scheduler: Warmup + Cosine ====
E = int(cfg["training"]["epochs"])
warmup_epochs = int(cfg.get("training", {}).get("warmup_epochs", 5))
sched_name = str(cfg.get("training", {}).get("scheduler", "cosine")).lower()

def lr_lambda(epoch_idx):
    # epoch_idx: 0-based
    if sched_name != "cosine":
        return 1.0
    if epoch_idx < warmup_epochs and warmup_epochs > 0:
        return float(epoch_idx + 1) / float(warmup_epochs)  # lineer warmup
    # cosine anneal from 1 -> 0 over remaining epochs
    t = (epoch_idx - warmup_epochs) / max(1, (E - warmup_epochs))
    return 0.5 * (1.0 + cos(pi * min(1.0, max(0.0, t))))

scheduler = th.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

# ==== 6) Epoch çalıştırma ====
def run_epoch(file_list, train=True, batch=65536, shuffle_files=True, max_points=None):
    th.cuda.empty_cache()  # GPU cache temizle
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    n_cls = len(order)
    conf = th.zeros((n_cls, n_cls), dtype=th.long, device=device)

    files = list(file_list)
    if train and shuffle_files:
        random.shuffle(files)

    for path in files:
        XYZ, RGB, Y = read_ply(path)

        if train:
            XYZ = augment_xyz(XYZ)

        # voxel + normalize
        XYZ, RGB, Y = voxel_downsample(XYZ, RGB, Y, voxel=vox)
        
        # Boş dosya kontrolü
        if len(XYZ) == 0:
            print(f"[WARN] Skipping empty file: {os.path.basename(path)}")
            continue
            
        if cfg.get("dataset", {}).get("preprocess", {}).get("mean_center", True):
            XYZ = XYZ - XYZ.mean(axis=0, keepdims=True)
        XYZ = XYZ / vox

        # büyük dosyada eğitim için üst sınır
        if train and (max_points is not None) and (len(Y) > max_points):
            sel = np.random.choice(len(Y), size=max_points, replace=False)
            XYZ, RGB, Y = XYZ[sel], RGB[sel], Y[sel]

        X = np.concatenate([XYZ, RGB], 1)

        idx = np.arange(len(Y))
        if train:
            np.random.shuffle(idx)
        for i in range(0, len(Y), batch):
            j = idx[i:i+batch]
            if len(j) < 2:
                continue

            xb = th.from_numpy(X[j]).to(device)
            yb = th.from_numpy(Y[j]).to(device)

            if train:
                opt.zero_grad()
            with th.amp.autocast('cuda', enabled=use_amp):
                out = model(xb)
                loss = crit(out, yb)
            if train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
            loss_sum += float(loss.item()) * len(yb)
            
            # -100 olan label'ları filtrele
            valid_mask = (yb >= 0)
            if valid_mask.any():
                conf.index_put_((yb[valid_mask], pred[valid_mask]), 
                                th.ones(valid_mask.sum(), dtype=th.long, device=device), 
                                accumulate=True)
    
    # Per-class IoU hesapla
    conf_np = conf.cpu().numpy()
    per_class_iou = []
    for i in range(len(order)):
        tp = conf_np[i, i]
        fp = conf_np[:, i].sum() - tp
        fn = conf_np[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8)
        per_class_iou.append(iou * 100)
        print(f"  {order[i]:12s}: {iou*100:5.2f}%")

    return (loss_sum / total) if total>0 else 0.0, (correct / total) if total>0 else 0.0, conf.cpu()

def miou_from_conf(conf):
    conf = conf.cpu().numpy()
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    valid = denom > 0
    iou = np.zeros_like(denom, dtype=np.float64)
    iou[valid] = tp[valid] / denom[valid]
    miou = iou[valid].mean() if valid.any() else 0.0
    return float(miou), iou

best_miou = 0.0

# Batch boyutları (güvenli artış)
if model_name == "randla_like":
    B_TR, B_VA = 8192, 16384
else:
    B_TR, B_VA = 65536, 65536

# ==== Rotasyon: sabit karışım sırası + kayan pencere ====
rng = np.random.RandomState(seed)
base_order = tr_files.copy()
rng.shuffle(base_order)

# ==== Resume ====
resume_path = os.path.join(CKPT, "resume.pth")
start_epoch = 1
ep_times = []
if RESUME and os.path.exists(resume_path):
    ck = th.load(resume_path, map_location=device)
    try:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        best_miou = ck.get("best_miou", 0.0)
        start_epoch = int(ck.get("epoch", 0)) + 1
        ep_times = list(ck.get("ep_times", []))
        base_order = ck.get("base_order", base_order)
        # scheduler'ı yeniden başlat (epoch'a göre advance edelim)
        for ep in range(start_epoch-1):
            scheduler.step()
        print(f"[RESUME] epoch={start_epoch} best_mIoU={best_miou*100:.2f}% (lr={opt.param_groups[0]['lr']:.2e})")
    except Exception as e:
        print(f"[RESUME] yüklenemedi: {e} (sıfırdan başlıyorum)")

# ==== Eğitim döngüsü ====
for e in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
    t0 = time.time()

    # kayan pencere ile alt küme seç
    M = max(1, min(FILES_PER_EPOCH, len(base_order)))
    start = ((e - 1) * FILE_STEP) % len(base_order)
    if start + M <= len(base_order):
        tr_subset = base_order[start:start+M]
    else:
        tr_subset = base_order[start:] + base_order[:(start+M) % len(base_order)]

    # garanti: en az MIN_REAL_PER_EPOCH gerçek dosya olsun
    if MIN_REAL_PER_EPOCH > 0 and len(real_files) > 0:
        have_real = [p for p in tr_subset if is_real_file(p)]
        need = max(0, MIN_REAL_PER_EPOCH - len(have_real))
        if need > 0:
            rr = np.random.RandomState(seed + e)
            pool = (real_files * max(1, REAL_OVERSAMPLE)).copy()
            rr.shuffle(pool)
            add_list = []
            for p in pool:
                if p not in tr_subset:
                    add_list.append(p)
                if len(add_list) >= need: break
            if len(add_list) < need:
                add_list += pool[:(need - len(add_list))]
            for i in range(need):
                tr_subset[-(i+1)] = add_list[i]

    tr_loss, tr_acc, _    = run_epoch(tr_subset, train=True,  batch=B_TR, shuffle_files=False, max_points=MAX_POINTS_PER_FILE)
    va_loss, va_acc, conf = run_epoch(va_files,   train=False, batch=B_VA, shuffle_files=False, max_points=None)
    val_miou, _ = miou_from_conf(conf)

    dt = time.time() - t0
    ep_times.append(dt)
    avg = sum(ep_times) / len(ep_times)
    eta_sec = avg * (int(cfg["training"]["epochs"]) - e)

    print(
        f"[E{e:02d}] [{dt:.1f}s | avg{avg:.1f}s | eta~{eta_sec/60:.1f}m | lr={opt.param_groups[0]['lr']:.2e}] "
        f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% | "
        f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}% | val/miou={val_miou*100:.2f}% "
        f"| files[{start}:{start+M}] real_in_subset={sum(1 for p in tr_subset if is_real_file(p))}"
    )

    # en iyi model kaydı
    if val_miou > best_miou:
        best_miou = val_miou
        path = os.path.join(CKPT, f"{model_name}_e{e:02d}_miou{val_miou*100:.2f}.pt")
        th.save(model.state_dict(), path)
        print(f"[CKPT] saved: {path}")

    # scheduler step (epoch bazlı)
    scheduler.step()

    # resume için durum kaydı
    th.save({
        "epoch": e,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "best_miou": best_miou,
        "ep_times": ep_times,
        "base_order": base_order,
        "cfg": cfg,
    }, resume_path)

print(f"[DONE] best_val_miou={best_miou*100:.2f}%")
