# %% [markdown]
# # RSNA Knee — 06: cache DINOv3 features for every training study
#
# **Settings:** Accelerator **GPU T4 x2** (one GPU is used) · Internet **On** ·
# Inputs: the competition data **and** the `knee-mri-fold-weights` dataset.
#
# Reads each training study's DICOMs exactly the way the public knee-tuned DINOv3 encoder expects,
# runs that frozen encoder once, and keeps only a small feature vector per study.
# 570 GB of DICOMs become a ~250 MB file you can train on in minutes on a laptop.
#
# Output: `dv3_features_f0.npz` in `/kaggle/working/`.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.20",
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls"], check=False)

# %%
import os, glob, time, math, json
import numpy as np, pandas as pd, cv2, pydicom
import torch, torch.nn as nn, timm
from concurrent.futures import ProcessPoolExecutor
cv2.setNumThreads(1)

# ---- the public encoder's preprocessing recipe, copied exactly ----
CROP_MM, SIZE, N_SLICE = 130.0, 336, 16
SLICE_BAND = (0.12, 0.88)
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 1), ("Axial", 0)]
N_SLOT = len(SLOTS)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
SMOKE = int(os.environ.get("KNEE_SMOKE", "0"))        # >0: only this many studies, for a quick wiring check
HAS_GPU = torch.cuda.is_available()
TIME_BUDGET_MIN = float(os.environ.get("KNEE_BUDGET", "480" if HAS_GPU else "660"))
os.makedirs(OUT, exist_ok=True)
NCPU = os.cpu_count() or 4
# On CPU the encoder needs the cores more than the DICOM readers do.
CHUNK, WORKERS = 24, (max(1, min(8, NCPU)) if HAS_GPU else max(1, NCPU // 2))
if not HAS_GPU:
    torch.set_num_threads(NCPU)

def find(name):
    for cur, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        if len(dirs) > 200:
            dirs[:] = []
        if name in files:
            return os.path.join(cur, name)

DATA = os.path.dirname(find("train_series.csv"))
SERIES_ROOT = os.path.join(DATA, "train_series")
CKPT = find("m_f0.pt")
assert CKPT, "attach the knee-mri-fold-weights dataset"
print("data:", DATA, "\nckpt:", CKPT, "\nworkers:", WORKERS)

train = pd.read_csv(os.path.join(DATA, "train.csv"), dtype={"StudyInstanceUID": str})
ser = pd.read_csv(os.path.join(DATA, "train_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
train_ids = train.StudyInstanceUID.astype(str).tolist()
if SMOKE:
    train_ids = train_ids[:SMOKE]
by_study = {u: g.to_dict("records") for u, g in ser[ser.StudyInstanceUID.isin(set(train_ids))].groupby("StudyInstanceUID")}
print("studies:", len(train_ids))

# %%
# ---- preprocessing (identical to the public recipe) ----
def ordered_files(sdir, cap=64):
    keyed = []
    for f in glob.glob(os.path.join(sdir, "*.dcm")):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            keyed.append((int(ds.InstanceNumber), f))
        except Exception:
            continue
        if len(keyed) >= cap * 4:
            break
    return [f for _, f in sorted(keyed)]

def series_side(path):
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0

def read_crop(path):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def window(crop, lo, hi, flip):
    c = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def render(path, flip):
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return window(crop, lo, hi, flip)

def build_study(args):
    idx, study, recs = args
    out = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    if len(rows):
        for s_i, (plane, fs) in enumerate(SLOTS):
            sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
            if sub.empty:
                continue
            files = ordered_files(os.path.join(SERIES_ROOT, study, sub.iloc[0].SeriesInstanceUID))
            if not files:
                continue
            flip = plane != "Sagittal" and series_side(files[0]) < 0
            lo, hi = SLICE_BAND
            i0, i1 = int(round(lo * (len(files) - 1))), int(round(hi * (len(files) - 1)))
            avail = list(range(i0, i1 + 1))
            if len(avail) >= N_SLICE:
                picks, off = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, N_SLICE)], 0
            else:
                picks, off = avail, (N_SLICE - len(avail)) // 2
            for c, p in enumerate(picks):
                img = render(files[p], flip)
                if img is None:
                    img = render(files[min(len(files) - 1, p + 1)], flip)
                if img is not None:
                    out[s_i, off + c] = (img * 255).astype(np.uint8)
            mask[s_i] = len(picks)
    return idx, out, mask

# %%
# ---- the frozen encoder ----
MASK_IDX = 0
class ViTSlotToken(nn.Module):
    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, "num_prefix_tokens", 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, "attn", None)
            if a is not None and hasattr(a, "num_prefix_tokens"):
                a.num_prefix_tokens = a.num_prefix_tokens + 1
    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)
    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, "patch_drop", None), x)
        x = self._maybe(getattr(v, "norm_pre", None), x)
        npt = self._orig_prefix
        x = torch.cat([x[:, :npt], self.tok(cat).unsqueeze(1), x[:, npt:]], 1)
        if rope is not None:
            if getattr(v, "rope_mixed", False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
assert cfg["cond"] == "token" and cfg["n_slice"] == N_SLICE and cfg["img"] == SIZE, cfg
vit = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0, in_chans=N_SLICE, img_size=cfg["img"])
enc = ViTSlotToken(vit, N_SLOT)
miss, unexp = enc.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("enc.")}, strict=False)
assert not miss and not unexp, (miss[:4], unexp[:4])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc.eval().to(dev)
PREFIX = enc._orig_prefix + 1          # prefix tokens + the slot token: everything after this is a real patch
print(f"encoder ready: {cfg['backbone']} | fold {ck.get('fold')} | prefix {PREFIX} | device {dev}")

@torch.no_grad()
def encode(vols, masks, micro=12):
    """vols: [B, 6, 16, 336, 336] uint8 -> features [B, 6, 12*384] fp16, mask [B, 6]."""
    recs = [(b, s) for b in range(len(masks)) for s in np.nonzero(masks[b] > 0)[0]]
    feats, out_mask = None, np.zeros((len(masks), N_SLOT), np.uint8)
    for st in range(0, len(recs), micro):
        part = recs[st:st + micro]
        im = torch.from_numpy(np.stack([vols[b, s] for b, s in part])).to(dev).float().div_(255.0)
        cat = torch.tensor([int(s) + 1 for _, s in part], device=dev, dtype=torch.long)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            t = enc.forward_features(im, cat)
        patches = t[:, PREFIX:].float()
        n = patches.shape[1]
        g = int(round(math.sqrt(n)))
        assert g * g == n, (n, g)
        p = patches.reshape(len(part), g, g, -1)
        r = max(1, g // 4); c0 = g // 2
        focal = p[:, max(0, c0 - r):min(g, c0 + r + 1), max(0, c0 - r):min(g, c0 + r + 1)].mean((1, 2))
        grid = torch.stack([p[:, a[0]:a[1], b[0]:b[1]].mean((1, 2))
                            for a in _thirds(g) for b in _thirds(g)], 1)       # [B, 9, D]
        f = torch.cat([t[:, 0].float(), patches.mean(1), focal, grid.flatten(1)], 1)
        f = f.cpu().numpy().astype(np.float16)
        if feats is None:
            feats = np.zeros((len(masks), N_SLOT, f.shape[1]), np.float16)
        for row, (b, s) in enumerate(part):
            feats[b, s] = f[row]; out_mask[b, s] = 1
    if feats is None:
        feats = np.zeros((len(masks), N_SLOT, 12 * 384), np.float16)
    return feats, out_mask

def _thirds(g):
    e = [0, g // 3, 2 * g // 3, g]
    return [(e[i], e[i + 1]) for i in range(3)]

# %%
t0 = time.time()
PART = f"{OUT}/dv3_features_f0_partial.npz"
done, F, M = 0, None, None
resume_from = PART if os.path.exists(PART) else find("dv3_features_f0_partial.npz")
if resume_from:
    z = np.load(resume_from)
    F, M, done = z["features"].copy(), z["mask"].copy(), int(z["done"])
    print("resuming after", done, "studies, from", resume_from)
jobs = [(i, u, by_study.get(u, [])) for i, u in enumerate(train_ids)][done:]
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    buf = []
    for idx, vol, mask in ex.map(build_study, jobs, chunksize=1):
        buf.append((idx, vol, mask))
        if len(buf) == CHUNK:
            f, m = encode(np.stack([b[1] for b in buf]), np.stack([b[2] for b in buf]))
            if F is None:
                F = np.zeros((len(train_ids), N_SLOT, f.shape[2]), np.float16)
                M = np.zeros((len(train_ids), N_SLOT), np.uint8)
            for k, (i2, _, _) in enumerate(buf):
                F[i2], M[i2] = f[k], m[k]
            done = max(done, buf[-1][0] + 1); buf = []
            el = (time.time() - t0) / 60
            if done % (CHUNK * 10) == 0:
                np.savez(PART, features=F, mask=M, done=done)
                print(f"  {done}/{len(train_ids)}  {el:.1f} min  ~{el/max(done,1)*(len(train_ids)-done):.0f} min left")
            if el > TIME_BUDGET_MIN:
                print("time budget reached — saving what is done"); break
    if buf:
        f, m = encode(np.stack([b[1] for b in buf]), np.stack([b[2] for b in buf]))
        if F is None:
            F = np.zeros((len(train_ids), N_SLOT, f.shape[2]), np.float16)
            M = np.zeros((len(train_ids), N_SLOT), np.uint8)
        for k, (i2, _, _) in enumerate(buf):
            F[i2], M[i2] = f[k], m[k]
        done = max(done, buf[-1][0] + 1)

np.savez_compressed(f"{OUT}/dv3_features_f0.npz", features=F, mask=M,
                    ids=np.array(train_ids, dtype=object).astype(str), slots=np.array([f"{p}_{f_}" for p, f_ in SLOTS]))
if os.path.exists(PART):
    os.remove(PART)
print(f"saved {F.shape} | studies with no slot: {int((M.sum(1) == 0).sum())} | "
      f"{(time.time()-t0)/60:.1f} min | {os.path.getsize(f'{OUT}/dv3_features_f0.npz')/1e6:.0f} MB")

# %% [markdown]
# ## SUMMARY — paste this back to Claude

# %%
print(json.dumps({"studies": len(train_ids), "done": int(done), "feature_dim": int(F.shape[2]),
                  "slot_coverage": {f"{p}_{f_}": round(float(M[:, i].mean()), 3) for i, (p, f_) in enumerate(SLOTS)},
                  "no_slots": int((M.sum(1) == 0).sum()), "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
