# %% [markdown]
# # RSNA Knee — 04: Train the image model (Kaggle GPU or your laptop)
#
# **On Kaggle:** Accelerator **GPU T4 x2** (uses one) · Internet **On** (downloads pretrained weights) ·
# Inputs: competition data + outputs of notebooks **01** (folds), **02** (report grades), **03** (volumes).
# Edit `FOLD` in the config cell, then **Save Version → Save & Run All**.
#
# **On your laptop:** run `python 04_train.py --fold 0` (see the instructions Claude gave you).
#
# **What it does:** trains on one fold's training studies with soft targets made from the report grades
# (calibrated on the 58 radiologist-labelled studies), validates on the held-out fold every epoch, and saves the
# best checkpoint + out-of-fold predictions.
#
# **Outputs:** `model_f{FOLD}.pt`, `oof_f{FOLD}.csv`, `train_log_f{FOLD}.json`. Paste the SUMMARY block back to Claude.

# %%
# ---- shared model code: identical in training and submission notebooks ----
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def to_25d(x, slice_idx):
    """x: [B, S, N, H, W] uint8 -> [B, S, n, 3, H, W] float, channels = slices (i-1, i, i+1)."""
    N = x.shape[2]
    idx = torch.as_tensor(slice_idx, device=x.device)
    trip = torch.stack([(idx - 1).clamp(0, N - 1), idx, (idx + 1).clamp(0, N - 1)], 1)  # [n, 3]
    out = x[:, :, trip]  # [B, S, n, 3, H, W]
    return out.float().div_(255.0).sub_(0.5).div_(0.25)

class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0.ra_in1k", pretrained=True, n_slots=5, n_slices=20,
                 d=256, n_layers=2, n_labels=12, drop=0.2, grad_ckpt=True):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        if grad_ckpt and hasattr(self.enc, "set_grad_checkpointing"):
            self.enc.set_grad_checkpointing(True)
        nf = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(nf, d), nn.LayerNorm(d))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_slices, d))
        nn.init.trunc_normal_(self.slot_emb, std=0.02); nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, n_layers)
        self.queries = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, dropout=0.1, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head_w = nn.Parameter(torch.zeros(n_labels, d))
        self.head_b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x, mask, slice_idx):
        """x: [B, S, N, H, W] uint8 on device; mask: [B, S] (1 = slot present); slice_idx: slices to use."""
        B, S = x.shape[:2]
        n = len(slice_idx)
        imgs = to_25d(x, slice_idx)                                  # [B, S, n, 3, H, W]
        feats = self.enc(imgs.flatten(0, 2))                         # [B*S*n, nf]
        t = self.proj(feats).view(B, S, n, -1)
        pos = self.pos_emb[:, torch.as_tensor(slice_idx, device=x.device)]  # [1, n, d]
        t = t + pos.unsqueeze(0) + self.slot_emb.unsqueeze(0)       # [B, S, n, d]
        t = t.flatten(1, 2)                                          # [B, S*n, d]
        pad = (mask == 0).repeat_interleave(n, dim=1)                # [B, S*n] True = ignore
        pad = pad & ~pad.all(1, keepdim=True)                        # a study with no slots: attend to zeros
        t = self.mixer(t, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        o, _ = self.attn(q, t, t, key_padding_mask=pad)              # [B, 12, d]
        o = self.drop(o)
        return (o * self.head_w).sum(-1) + self.head_b               # [B, 12] logits

SAG_SLOTS, INPLANE_X_SLOTS = [0, 1], [2, 3, 4]   # sagittal slices run along x; coronal/axial columns run along x

def mirror(x):
    """Mirror whole studies left<->right (patient x axis). x: [B, S, N, H, W].
    Anatomy is preserved (a mirrored right knee is a valid left knee), so labels do not change."""
    x = x.clone()
    x[:, SAG_SLOTS] = x[:, SAG_SLOTS].flip(2)          # reverse sagittal slice order
    x[:, INPLANE_X_SLOTS] = x[:, INPLANE_X_SLOTS].flip(4)  # flip image columns
    return x

def predict_tta(model, x, mask, slice_idx, tta=True):
    p = torch.sigmoid(model(x, mask, slice_idx).float())
    if tta:
        p = (p + torch.sigmoid(model(mirror(x), mask, slice_idx).float())) / 2
    return p

def slice_indices(n_total, n_use):
    return np.round(np.linspace(0, n_total - 1, n_use)).astype(int).tolist()


# %%
import os, sys, glob, json, time, argparse, random
import pandas as pd
from sklearn.metrics import roc_auc_score

CFG = dict(
    fold=0,
    backbone="efficientnet_b0.ra_in1k",
    pretrained=1,          # 0 only for offline smoke tests
    epochs=20,
    batch_size=4,          # studies per step (each study = up to 5 slots x 20 slices)
    accum=2,               # gradient accumulation steps -> effective batch 8
    lr_backbone=3e-4, lr_head=1e-3, weight_decay=1e-2, warmup_frac=0.05,
    n_slices_use=20,       # of the 20 stored slices per slot
    flip_p=0.5,            # probability of mirroring a whole study left<->right during training
    tta=1,                 # average original + mirrored prediction at validation
    gold_weight=3.0,       # loss weight for the radiologist-labelled studies
    calib_k=4.0,           # smoothing strength for grade -> probability calibration
    workers=4,
    seed=42,
    max_train_studies=None,  # set e.g. 200 for a quick smoke test
    data_roots=["/kaggle/input"],
    out_dir="/kaggle/working",
)

def parse_cli(cfg):
    """Command-line overrides when run as a script on the laptop, e.g. --fold 1 --batch_size 2."""
    if "ipykernel" in sys.modules:
        return cfg
    ap = argparse.ArgumentParser()
    for k, v in cfg.items():
        if isinstance(v, list):
            ap.add_argument(f"--{k}", nargs="+", default=v)
        elif v is None:
            ap.add_argument(f"--{k}", type=int, default=None)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return vars(ap.parse_args())

LABEL_KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa",
              "effusion", "synovitis", "baker", "contusion", "fracture"]

def find(roots, name, must_col=None):
    for root in roots:
        for cur, dirs, files in os.walk(root, followlinks=True):
            if len(dirs) > 200:
                dirs[:] = []
            if name in files:
                p = os.path.join(cur, name)
                if must_col is None or must_col in pd.read_csv(p, nrows=0).columns:
                    return p
    raise FileNotFoundError(f"{name} not found under {roots} — is that input attached / downloaded?")

# %% [markdown]
# ## Data: targets + dataset

# %%
def build_targets(train, grades, folds, fold, k):
    """Soft target per study/label = P(radiologist positive | report grade), fit on training-fold gold studies only."""
    df = train[["StudyInstanceUID"] + LABELS].merge(grades[["StudyInstanceUID"] + LABEL_KEYS], on="StudyInstanceUID") \
                                             .merge(folds[["StudyInstanceUID", "fold"]], on="StudyInstanceUID")
    gold = df[LABELS].notna().all(axis=1)
    fit = df[gold & (df.fold != fold)]
    calib = {}
    for lab, key in zip(LABELS, LABEL_KEYS):
        pooled = np.array([fit.loc[fit[k2] == g, l2].sum() for g in range(5) for l2, k2 in zip(LABELS, LABEL_KEYS)]).reshape(5, -1).sum(1)
        n_pooled = np.array([(fit[k2] == g).sum() for g in range(5) for k2 in LABEL_KEYS]).reshape(5, -1).sum(1)
        prior = (pooled + 1) / (n_pooled + 2)
        p = np.array([(fit.loc[fit[key] == g, lab].sum() + k * prior[g]) / ((fit[key] == g).sum() + k) for g in range(5)])
        calib[lab] = np.maximum.accumulate(p).round(4).tolist()   # non-decreasing in grade
    T = np.stack([np.array(calib[lab])[df[key].astype(int).values] for lab, key in zip(LABELS, LABEL_KEYS)], 1)
    T[gold.values] = df.loc[gold, LABELS].values                 # true labels where we have them
    W = np.where(gold.values, 1.0, 0.0)
    proxy = (df[LABEL_KEYS].values >= 3).astype(np.float32)       # report-derived proxy positives (for validation)
    return df[["StudyInstanceUID", "fold"]].assign(is_gold=gold.values), T.astype(np.float32), W, proxy, calib

class VolDS(torch.utils.data.Dataset):
    def __init__(self, uids, shard_of, mask_of, targets=None, weights=None):
        self.uids, self.shard_of, self.mask_of = list(uids), shard_of, mask_of
        self.targets, self.weights = targets, weights
        self._h = {}
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, i):
        u = self.uids[i]
        p = self.shard_of[u]
        if p not in self._h:
            self._h[p] = np.load(p)
        vol = self._h[p][u]
        out = {"vol": torch.from_numpy(np.ascontiguousarray(vol)), "mask": torch.from_numpy(self.mask_of[u].astype(np.float32))}
        if self.targets is not None:
            out["y"] = torch.from_numpy(self.targets[i]); out["w"] = torch.tensor(self.weights[i], dtype=torch.float32)
        return out

def gpu_augment(x, flip_p=0.5):
    """x: [B, S, N, H, W] uint8 on GPU. Small random affine per slot (same for all its slices) + intensity jitter.
    Whole-study mirroring is allowed: it turns a right knee into an anatomically valid left knee (labels unchanged)."""
    B, S, N, H, W = x.shape
    if flip_p > 0:
        flip = torch.rand(B, device=x.device) < flip_p
        if flip.any():
            x = x.clone(); x[flip] = mirror(x[flip])
    xf = x.float().view(B * S, N, H, W)
    ang = (torch.rand(B * S, device=x.device) - 0.5) * 2 * math.radians(10)
    sc = 1 + (torch.rand(B * S, device=x.device) - 0.5) * 0.2
    tx, ty = [(torch.rand(B * S, device=x.device) - 0.5) * 0.12 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B * S, N, H, W), align_corners=False)
    xf = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gain = 1 + (torch.rand(B * S, 1, 1, 1, device=x.device) - 0.5) * 0.3
    gamma = torch.exp((torch.rand(B * S, 1, 1, 1, device=x.device) - 0.5) * 0.4)
    xf = ((xf / 255).clamp(0, 1) ** gamma * gain * 255).clamp(0, 255)
    return xf.view(B, S, N, H, W).to(torch.uint8)

def macro_auc(y, p):
    aucs = {}
    for j, lab in enumerate(LABELS):
        if 0 < y[:, j].sum() < len(y):
            aucs[lab] = roc_auc_score(y[:, j], p[:, j])
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), aucs

# %% [markdown]
# ## Train

# %%
def main(cfg):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    roots, fold = cfg["data_roots"], cfg["fold"]
    train = pd.read_csv(find(roots, "train.csv", "Report"))
    grades = pd.read_csv(find(roots, "report_grades.csv"))
    folds = pd.read_csv(find(roots, "folds.csv"))
    vidx_path = find(roots, "vol_index.csv")
    vidx = pd.read_csv(vidx_path)
    vol_dir = os.path.join(os.path.dirname(vidx_path), "vols")
    if not os.path.isdir(vol_dir):
        vol_dir = os.path.dirname(glob.glob(os.path.join(os.path.dirname(vidx_path), "**", "shard_*.npz"), recursive=True)[0])
    shard_of = {u: os.path.join(vol_dir, f"shard_{s:04d}.npz") for u, s in zip(vidx.StudyInstanceUID, vidx.shard)}
    mcols = [c for c in vidx.columns if c.startswith("m_")]
    mask_of = {u: m for u, m in zip(vidx.StudyInstanceUID, vidx[mcols].values)}

    meta, T, W, proxy, calib = build_targets(train, grades, folds, fold, cfg["calib_k"])
    keep = meta.StudyInstanceUID.isin(shard_of).values
    meta, T, W, proxy = meta[keep].reset_index(drop=True), T[keep], W[keep], proxy[keep]
    W = 1.0 + (cfg["gold_weight"] - 1.0) * W
    tr, va = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
    if cfg["max_train_studies"]:
        tr, va = tr[:cfg["max_train_studies"]], va[:max(cfg["max_train_studies"] // 4, 8)]
    print(f"fold {fold}: train {len(tr)} | val {len(va)} (gold in val: {int(meta.is_gold.values[va].sum())})")
    print("calibration P(pos | grade 0..4):", json.dumps({k: v for k, v in calib.items()}))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 only on GPUs with native support (Ampere+, e.g. RTX 30/40). T4 (Turing) emulates bf16 very slowly -> use fp16.
    amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    use_scaler = dev.type == "cuda" and amp_dtype == torch.float16
    print("device:", torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "| amp:", amp_dtype)

    mk = lambda ix, tgt: VolDS(meta.StudyInstanceUID.values[ix], shard_of, mask_of, T[ix] if tgt else None, W[ix] if tgt else None)
    kw = dict(num_workers=cfg["workers"], pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0)
    dl_tr = torch.utils.data.DataLoader(mk(tr, True), batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **kw)
    dl_va = torch.utils.data.DataLoader(mk(va, False), batch_size=cfg["batch_size"], shuffle=False, **kw)

    model = KneeNet(cfg["backbone"], pretrained=bool(cfg["pretrained"])).to(dev)
    sidx = slice_indices(20, cfg["n_slices_use"])
    enc_params = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr_backbone"]},
                             {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(dl_tr) // cfg["accum"]
    warm = max(1, int(steps * cfg["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    def predict(dl):
        model.eval(); P = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            for b in dl:
                P.append(predict_tta(model, b["vol"].to(dev, non_blocking=True), b["mask"].to(dev), sidx, bool(cfg["tta"])).cpu().numpy())
        return np.concatenate(P)

    log, best, t0 = [], -1, time.time()
    for ep in range(cfg["epochs"]):
        model.train(); tl, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(dl_tr):
            x = gpu_augment(b["vol"].to(dev, non_blocking=True), cfg["flip_p"])
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
                logits = model(x, b["mask"].to(dev), sidx)
            loss = (F.binary_cross_entropy_with_logits(logits.float(), b["y"].to(dev), reduction="none").mean(1)
                    * b["w"].to(dev)).mean() / cfg["accum"]
            scaler.scale(loss).backward()
            if (i + 1) % cfg["accum"] == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            tl += loss.item() * cfg["accum"]; nb += 1
            if ep == 0 and i == 9:
                print(f"  first 10 steps: {(time.time()-t0)/10:.2f}s/step -> ~{(time.time()-t0)/10*len(dl_tr)/60:.1f} min/epoch")
        P = predict(dl_va)
        proxy_auc, _ = macro_auc(proxy[va], P)
        g = meta.is_gold.values[va]
        gold_auc, _ = macro_auc(T[va][g], P[g]) if g.sum() >= 4 else (float("nan"), {})
        rec = {"epoch": ep + 1, "train_loss": round(tl / nb, 4), "val_report_auc": round(proxy_auc, 4),
               "val_gold_auc": round(gold_auc, 4), "minutes": round((time.time() - t0) / 60, 1)}
        log.append(rec); print(rec)
        if proxy_auc > best:
            best = proxy_auc
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "calib": calib, "epoch": ep + 1},
                       os.path.join(cfg["out_dir"], f"model_f{fold}.pt"))
            pd.DataFrame(P, columns=LABELS).assign(StudyInstanceUID=meta.StudyInstanceUID.values[va],
                                                   is_gold=g).to_csv(os.path.join(cfg["out_dir"], f"oof_f{fold}.csv"), index=False)
    # per-label report for the best epoch
    oof = pd.read_csv(os.path.join(cfg["out_dir"], f"oof_f{fold}.csv"))
    P = oof[LABELS].values
    rep_macro, rep_per = macro_auc(proxy[va], P)
    summary = {"fold": fold, "backbone": cfg["backbone"], "best_epoch": max(log, key=lambda r: r["val_report_auc"])["epoch"],
               "val_report_auc": round(rep_macro, 4), "val_report_auc_per_label": {k: round(v, 3) for k, v in rep_per.items()},
               "val_gold_auc_best_epoch": max(log, key=lambda r: r["val_report_auc"])["val_gold_auc"],
               "epochs": log, "total_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(cfg["out_dir"], f"train_log_f{fold}.json"), "w"), indent=1)
    print("\nSUMMARY\n" + json.dumps(summary, indent=1))
    return summary

# %% [markdown]
# ## Run — on Kaggle, change the fold number on the next line (0-4) before each run

# %%
CFG["fold"] = 0
if __name__ == "__main__":
    CFG = parse_cli(CFG)
    main(CFG)
