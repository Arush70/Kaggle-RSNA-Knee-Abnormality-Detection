"""Fine-tune the public knee-trained DINOv3 encoder on our own 336 px volumes.

The public `knee-mri-fold-weights` encoder is a DINOv3 ViT-S/16 that already understands
knee MRI: 16 slices go in as the 16 input channels of one image, one image per slot, and a
learned slot token tells it which view it is looking at. We start from those weights instead
of from a generic model, and adapt them with the public LLM labels.

  python finetune_dv3.py --fold 1                 # one fold
  python finetune_dv3.py --fold 1 --epochs 8 --freeze_blocks 8   # faster / less memory

Needs: data/vols336/ (notebook 03hr output), data/folds.csv, data/train.csv,
       pub/llm_labels/report_labels_v2.csv, pub/fold_weights/m_f0.pt
Writes: <out_dir>/model_f{fold}.pt, oof_f{fold}.csv, train_log_f{fold}.json
"""
import os, glob, json, time, math, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.utils.checkpoint, timm

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# our slots, and which of the encoder's 6 slot tokens each one is closest to
OUR_SLOTS = ["sag_fs", "sag_nfs", "cor_fs", "cor_nfs", "ax_fs"]
SLOT_TOKEN = [1, 2, 3, 4, 5]
N_CH = 16            # the encoder takes 16 slices as 16 channels
SAG = [0, 1]         # sagittal slots: mirroring reverses slice order
INPLANE = [2, 3, 4]  # coronal/axial: mirroring flips image columns

p = argparse.ArgumentParser()
p.add_argument("--fold", type=int, required=True)
p.add_argument("--vols", default="data/vols336")
p.add_argument("--ckpt", default="pub/fold_weights/m_f0.pt")
p.add_argument("--labels", default="pub/llm_labels/report_labels_v2.csv")
p.add_argument("--init_head", default="", help="optional runs_head/head_f{fold}.pt to start the head from")
p.add_argument("--out_dir", default="runs_dv3ft")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--freeze_blocks", type=int, default=6)
p.add_argument("--lr_backbone", type=float, default=1e-5)
p.add_argument("--lr_head", type=float, default=5e-4)
p.add_argument("--weight_decay", type=float, default=1e-2)
p.add_argument("--accum", type=int, default=8)
p.add_argument("--drop", type=float, default=0.3)
p.add_argument("--d", type=int, default=256)
p.add_argument("--gold_weight", type=float, default=3.0)
p.add_argument("--flip_p", type=float, default=0.5)
p.add_argument("--tta", type=int, default=1)
p.add_argument("--max_studies", type=int, default=0, help=">0: tiny run, for checking it works")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().to_numpy(); n1 = int(y.sum()); n0 = int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---------------- data ----------------
idx = pd.concat([pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in sorted(glob.glob(f"{a.vols}/**/vol_index*.csv", recursive=True))])
shard_dir = os.path.dirname(sorted(glob.glob(f"{a.vols}/**/shard_*.npz", recursive=True))[0])
shard_of = {u: os.path.join(shard_dir, f"shard_{s:04d}.npz") for u, s in zip(idx.StudyInstanceUID, idx.shard)}
mask_of = dict(zip(idx.StudyInstanceUID, idx[[f"m_{n}" for n in OUR_SLOTS]].to_numpy(np.float32)))
print(f"volumes: {len(shard_of)} studies in {len(set(shard_of.values()))} shards")

train = pd.read_csv("data/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
folds = pd.read_csv("data/folds.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
pub = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
gold_all = train[LABELS].notna().all(axis=1)

rows = []
for u in shard_of:
    if u not in folds.index or u not in pub.index:
        continue
    g = bool(gold_all.get(u, False))
    y = train.loc[u, LABELS].to_numpy(np.float32) if g else pub.loc[u, LABELS].to_numpy(np.float32)
    rows.append((u, int(folds.loc[u, "fold"]), g, y))
if a.max_studies:
    rows = rows[:a.max_studies]
tr_rows = [r for r in rows if r[1] != a.fold]
va_rows = [r for r in rows if r[1] == a.fold]
print(f"fold {a.fold}: train {len(tr_rows)} | val {len(va_rows)} (gold in val: {sum(r[2] for r in va_rows)})")

class DS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows; self.cache = {}
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        u, _, g, y = self.rows[i]
        path = shard_of[u]
        z = self.cache.get(path)
        if z is None:
            z = np.load(path, mmap_mode="r"); self.cache = {path: z}
        vol = np.asarray(z[u])                                   # [5, 20, 336, 336] uint8
        sl = np.round(np.linspace(0, vol.shape[1] - 1, N_CH)).astype(int)
        return (torch.from_numpy(vol[:, sl].copy()), torch.from_numpy(mask_of[u]),
                torch.from_numpy(y), torch.tensor(a.gold_weight if g else 1.0))

def mirror(x):                       # x: [B, 5, 16, H, W]
    x = x.clone()
    x[:, SAG] = x[:, SAG].flip(2)            # reverse slice order
    x[:, INPLANE] = x[:, INPLANE].flip(4)    # flip columns
    return x

# ---------------- model ----------------
MASK_IDX = 0
class ViTSlotToken(nn.Module):
    def __init__(self, vit, n_cat):
        super().__init__()
        self.vit = vit
        self.tok = nn.Embedding(n_cat + 1, vit.embed_dim, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, "num_prefix_tokens", 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            at = getattr(blk, "attn", None)
            if at is not None and hasattr(at, "num_prefix_tokens"):
                at.num_prefix_tokens = at.num_prefix_tokens + 1
    def forward_features(self, x, cat, freeze_blocks=0, grad_ckpt=True):
        v = self.vit
        with torch.set_grad_enabled(torch.is_grad_enabled() and freeze_blocks == 0):
            t = v.patch_embed(x)
            pos = v._pos_embed(t)
            rope = None
            if isinstance(pos, tuple):
                t, rope = pos
            else:
                t = pos
            for m in (getattr(v, "patch_drop", None), getattr(v, "norm_pre", None)):
                if m is not None:
                    t = m(t)
            npt = self._orig_prefix
            t = torch.cat([t[:, :npt], self.tok(cat).unsqueeze(1), t[:, npt:]], 1)
            for blk in v.blocks[:freeze_blocks]:
                t = blk(t, rope=rope) if rope is not None else blk(t)
        for blk in v.blocks[freeze_blocks:]:
            if grad_ckpt and self.training and torch.is_grad_enabled():
                t = torch.utils.checkpoint.checkpoint(blk, t, rope, use_reentrant=False) if rope is not None \
                    else torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t, rope=rope) if rope is not None else blk(t)
        return v.norm(t)

def thirds(g):
    e = [0, g // 3, 2 * g // 3, g]
    return [(e[i], e[i + 1]) for i in range(3)]

class Net(nn.Module):
    def __init__(self, enc, prefix, f_in, n_slot, d, drop):
        super().__init__()
        self.enc, self.prefix = enc, prefix
        # same standardisation the frozen-feature head was trained with, so we can warm-start from it
        self.register_buffer("mu", torch.zeros(f_in))
        self.register_buffer("sd", torch.ones(f_in))
        self.proj = nn.Sequential(nn.Linear(f_in, d), nn.LayerNorm(d), nn.GELU())
        self.slot = nn.Parameter(torch.zeros(n_slot, d)); nn.init.trunc_normal_(self.slot, std=0.02)
        self.q = nn.Parameter(torch.randn(12, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.w = nn.Parameter(torch.zeros(12, d)); self.b = nn.Parameter(torch.zeros(12))
    def features(self, x, freeze_blocks):
        B, S = x.shape[:2]
        im = x.flatten(0, 1).float().div_(255.0)                        # [B*S, 16, H, W]
        cat = torch.tensor(SLOT_TOKEN, device=x.device).repeat(B)
        t = self.enc.forward_features(im, cat, freeze_blocks)
        patches = t[:, self.prefix:].float()
        n = patches.shape[1]; g = int(round(math.sqrt(n)))
        pg = patches.reshape(len(patches), g, g, -1)
        r = max(1, g // 4); c = g // 2
        focal = pg[:, max(0, c - r):c + r + 1, max(0, c - r):c + r + 1].mean((1, 2))
        grid = torch.stack([pg[:, u0:u1, v0:v1].mean((1, 2)) for u0, u1 in thirds(g) for v0, v1 in thirds(g)], 1)
        f = torch.cat([t[:, 0].float(), patches.mean(1), focal, grid.flatten(1)], 1)
        return ((f - self.mu) / self.sd).view(B, S, -1)
    def forward(self, x, m, freeze_blocks=0):
        t = self.proj(self.features(x, freeze_blocks)) + self.slot
        pad = m == 0
        pad = pad & ~pad.all(1, keepdim=True)
        o, _ = self.attn(self.q.expand(len(t), -1, -1), t, t, key_padding_mask=pad)
        return (self.drop(o) * self.w).sum(-1) + self.b

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
vit = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0, in_chans=N_CH, img_size=cfg["img"])
enc = ViTSlotToken(vit, 6)
miss, unexp = enc.load_state_dict({k[4:]: v for k, v in ck["state_dict"].items() if k.startswith("enc.")}, strict=False)
assert not miss and not unexp, (miss[:4], unexp[:4])
model = Net(enc, enc._orig_prefix + 1, 12 * vit.num_features, len(OUR_SLOTS), a.d, a.drop).to(dev)
if a.init_head and os.path.exists(a.init_head):
    hk = torch.load(a.init_head, map_location="cpu", weights_only=False)
    with torch.no_grad():
        model.mu.copy_(torch.as_tensor(hk["mu"], dtype=torch.float32))
        model.sd.copy_(torch.as_tensor(hk["sd"], dtype=torch.float32))
    want = model.state_dict()
    take = {}
    for k, v in hk["state_dict"].items():
        if k.startswith("enc.") or k not in want:
            continue
        if v.shape != want[k].shape:
            # the frozen-feature head had one slot row per encoder slot (6); we use 5
            if v.dim() == want[k].dim() and v.shape[1:] == want[k].shape[1:] and v.shape[0] > want[k].shape[0]:
                v = v[:want[k].shape[0]]
            else:
                continue
        take[k] = v
    model.load_state_dict(take, strict=False)
    print(f"head warm-started from {a.init_head}: {len(take)}/{len(want)} tensors reused")
for blk in enc.vit.blocks[:a.freeze_blocks]:
    for prm in blk.parameters():
        prm.requires_grad_(False)
for prm in enc.vit.patch_embed.parameters():
    prm.requires_grad_(a.freeze_blocks == 0)
enc_params = [q for n, q in model.named_parameters() if n.startswith("enc.") and q.requires_grad]
head_params = [q for n, q in model.named_parameters() if not n.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_backbone},
                         {"params": head_params, "lr": a.lr_head}], weight_decay=a.weight_decay)
print(f"device {dev} | amp {AMP} | frozen blocks {a.freeze_blocks} | trainable encoder tensors {len(enc_params)}")

# num_workers=0: on Windows, worker processes re-import this script and crash (freeze_support).
# Reading a study is ~0.1 s of decompression against ~2 s on the GPU, so this costs almost nothing.
tl = torch.utils.data.DataLoader(DS(tr_rows), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
vl = torch.utils.data.DataLoader(DS(va_rows), batch_size=1, shuffle=False, num_workers=0)
steps = max(1, len(tl) // a.accum) * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=steps, pct_start=0.1)
bce = nn.BCEWithLogitsLoss(reduction="none")
scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16 and dev.type == "cuda")

@torch.no_grad()
def evaluate():
    model.eval(); P = []
    for x, m, y, w in vl:
        x, m = x.to(dev), m.to(dev)
        with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
            pr = torch.sigmoid(model(x, m, a.freeze_blocks).float())
            if a.tta:
                pr = (pr + torch.sigmoid(model(mirror(x), m, a.freeze_blocks).float())) / 2
        P.append(pr.cpu().numpy())
    return np.concatenate(P)

t0, log, best, step = time.time(), [], -1, 0
Yva = np.stack([r[3] for r in va_rows]); gva = np.array([r[2] for r in va_rows])
Ygold = train.loc[[r[0] for r in va_rows if r[2]], LABELS].to_numpy(float) if gva.any() else None
for ep in range(1, a.epochs + 1):
    model.train(); tot = n = 0
    opt.zero_grad(set_to_none=True)
    for i, (x, m, y, w) in enumerate(tl):
        x, m, y, w = x.to(dev, non_blocking=True), m.to(dev), y.to(dev), w.to(dev)
        if a.flip_p and np.random.rand() < a.flip_p:
            x = mirror(x)
        with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
            loss = (bce(model(x, m, a.freeze_blocks), y).mean(1) * w).sum() / w.sum() / a.accum
        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
        tot += float(loss) * a.accum; n += 1
        if (i + 1) % a.accum == 0:
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if step < steps - 1:
                sched.step()
            step += 1
        if i == 19:
            print(f"  first 20 studies: {(time.time()-t0)/20:.2f}s each -> ~{(time.time()-t0)/20*len(tl)/60:.0f} min/epoch", flush=True)
    P = evaluate()
    rep = float(np.nanmean([auc(Yva[:, j] > 0.5, P[:, j]) for j in range(12)]))
    gld = float(np.nanmean([auc(Ygold[:, j], P[gva][:, j]) for j in range(12)])) if Ygold is not None else float("nan")
    log.append({"epoch": ep, "train_loss": round(tot / max(n, 1), 4), "val_label_auc": round(rep, 4),
                "val_gold_auc": round(gld, 4), "minutes": round((time.time() - t0) / 60, 1)})
    print(log[-1], flush=True)
    if rep > best:
        best = rep
        torch.save({"state_dict": model.state_dict(), "cfg": vars(a), "enc_cfg": cfg, "epoch": ep,
                    "labels": LABELS}, f"{a.out_dir}/model_f{a.fold}.pt")
        pd.DataFrame({"StudyInstanceUID": [r[0] for r in va_rows],
                      **{L: P[:, j] for j, L in enumerate(LABELS)}}).to_csv(f"{a.out_dir}/oof_f{a.fold}.csv", index=False)
summary = {"fold": a.fold, "best_label_auc": round(best, 4),
           "best_gold_auc": max(e["val_gold_auc"] for e in log), "epochs": log,
           "total_minutes": round((time.time() - t0) / 60, 1), "args": vars(a)}
json.dump(summary, open(f"{a.out_dir}/train_log_f{a.fold}.json", "w"), indent=1)
print("\nSUMMARY\n" + json.dumps({k: v for k, v in summary.items() if k != "epochs"}, indent=1))
