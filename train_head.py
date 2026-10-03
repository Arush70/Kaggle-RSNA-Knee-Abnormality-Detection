"""Train our own head on the frozen DINOv3 features (06-dv3-features output).

  python train_head.py                          # all 5 folds, default settings
  python train_head.py --epochs 60 --seeds 3    # more training / seed averaging

Writes  runs_head/head_f{fold}.pt,  runs_head/oof_head.csv,  runs_head/head_log.json
"""
import os, json, time, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="pub/dv3/dv3_features_f0.npz")
ap.add_argument("--labels", default="pub/llm_labels/report_labels_v2.csv")
ap.add_argument("--train_csv", default="data/train.csv")
ap.add_argument("--folds", default="data/folds.csv")
ap.add_argument("--out_dir", default="runs_head")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight_decay", type=float, default=1e-2)
ap.add_argument("--d", type=int, default=256)
ap.add_argument("--drop", type=float, default=0.3)
ap.add_argument("--gold_weight", type=float, default=3.0)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--only_fold", type=int, default=-1)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().to_numpy(); n1 = int(y.sum()); n0 = int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---------------- data ----------------
z = np.load(a.features, allow_pickle=False)
ids = z["ids"].astype(str); X = z["features"]; MK = z["mask"].astype(np.float32)
print(f"features {X.shape} | slots {z['slots'].tolist()}")

train = pd.read_csv(a.train_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
pub = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
folds = pd.read_csv(a.folds, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

gold_mask = train[LABELS].notna().all(axis=1)
Y = np.full((len(ids), 12), np.nan, np.float32)
G = np.zeros(len(ids), bool)
F = np.full(len(ids), -1, np.int64)
for i, u in enumerate(ids):
    if u in pub.index:
        Y[i] = pub.loc[u, LABELS].to_numpy(np.float32)
    if u in train.index and gold_mask.get(u, False):
        Y[i] = train.loc[u, LABELS].to_numpy(np.float32); G[i] = True
    if u in folds.index:
        F[i] = int(folds.loc[u, "fold"])
keep = ~np.isnan(Y).any(1) & (F >= 0)
print(f"usable studies {keep.sum()}/{len(ids)} | gold {int(G[keep].sum())} | folds {np.bincount(F[keep])}")

# report-grade proxy, the same yardstick the earlier runs were judged on
proxy = None
if os.path.exists("data/report_grades.csv"):
    KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa",
            "effusion", "synovitis", "baker", "contusion", "fracture"]
    gr = pd.read_csv("data/report_grades.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    proxy = np.full((len(ids), 12), np.nan, np.float32)
    for i, u in enumerate(ids):
        if u in gr.index:
            proxy[i] = (gr.loc[u, KEYS].to_numpy(np.float32) >= 3).astype(np.float32)

# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, f_in, n_slot, d, drop):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(f_in, d), nn.LayerNorm(d), nn.GELU())
        self.slot = nn.Parameter(torch.zeros(n_slot, d))
        nn.init.trunc_normal_(self.slot, std=0.02)
        self.q = nn.Parameter(torch.randn(12, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.w = nn.Parameter(torch.zeros(12, d)); self.b = nn.Parameter(torch.zeros(12))
    def forward(self, x, m):
        t = self.proj(x) + self.slot                      # [B, S, d]
        pad = m == 0
        pad = pad & ~pad.all(1, keepdim=True)
        o, _ = self.attn(self.q.expand(len(x), -1, -1), t, t, key_padding_mask=pad)
        return (self.drop(o) * self.w).sum(-1) + self.b

def run_fold(fold, seed):
    tr = keep & (F != fold); va = keep & (F == fold)
    mu, sd = X[tr].astype(np.float32).mean((0, 1)), X[tr].astype(np.float32).std((0, 1)) + 1e-3
    def prep(sel):
        x = (X[sel].astype(np.float32) - mu) / sd
        return (torch.from_numpy(x), torch.from_numpy(MK[sel]),
                torch.from_numpy(Y[sel]), torch.from_numpy(np.where(G[sel], a.gold_weight, 1.0).astype(np.float32)))
    xt, mt, yt, wt = prep(tr)
    xv, mv, yv, _ = prep(va)
    torch.manual_seed(seed); np.random.seed(seed)
    model = Head(X.shape[2], X.shape[1], a.d, a.drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = max(1, len(xt) // a.batch_size) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    xt, mt, yt, wt = xt.to(dev), mt.to(dev), yt.to(dev), wt.to(dev)
    xv, mv = xv.to(dev), mv.to(dev)
    step = 0
    for ep in range(a.epochs):
        model.train()
        order = torch.randperm(len(xt), device=dev)
        for i in range(0, len(order) - a.batch_size + 1, a.batch_size):
            ix = order[i:i + a.batch_size]
            loss = (bce(model(xt[ix], mt[ix]), yt[ix]).mean(1) * wt[ix]).sum() / wt[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xv, mv).float()).cpu().numpy()
    return model, p, va, (mu, sd)

t0 = time.time()
oof = np.zeros((len(ids), 12), np.float32)
log = {"folds": {}, "args": vars(a)}
for fold in sorted(set(F[keep].tolist())):
    if a.only_fold >= 0 and fold != a.only_fold:
        continue
    ps = []
    for s in range(a.seeds):
        model, p, va, norm = run_fold(fold, 42 + s)
        ps.append(p)
    p = np.mean(ps, 0)
    oof[va] = p
    yv, gv = Y[va], G[va]
    a_pub = float(np.nanmean([auc(yv[:, j] > 0.5, p[:, j]) for j in range(12)]))
    a_gold = float(np.nanmean([auc(yv[gv][:, j], p[gv][:, j]) for j in range(12)])) if gv.sum() else float("nan")
    a_proxy = float(np.nanmean([auc(proxy[va][:, j], p[:, j]) for j in range(12)])) if proxy is not None else float("nan")
    torch.save({"state_dict": model.state_dict(), "mu": norm[0], "sd": norm[1], "fold": fold,
                "cfg": vars(a), "labels": LABELS}, f"{a.out_dir}/head_f{fold}.pt")
    log["folds"][fold] = {"n_val": int(va.sum()), "gold_in_val": int(gv.sum()),
                          "auc_vs_public_labels": round(a_pub, 4), "auc_vs_report_proxy": round(a_proxy, 4),
                          "auc_gold": round(a_gold, 4)}
    print(f"fold {fold}: public-label AUC {a_pub:.4f} | report-proxy AUC {a_proxy:.4f} | gold {a_gold:.4f} "
          f"({int(gv.sum())} studies) | {(time.time()-t0)/60:.1f} min")

done = oof.sum(1) != 0
gm = G & done
log["overall"] = {
    "auc_vs_public_labels": round(float(np.nanmean([auc(Y[done][:, j] > 0.5, oof[done][:, j]) for j in range(12)])), 4),
    "auc_vs_report_proxy": (round(float(np.nanmean([auc(proxy[done][:, j], oof[done][:, j]) for j in range(12)])), 4)
                            if proxy is not None else None),
    "auc_gold": round(float(np.nanmean([auc(Y[gm][:, j], oof[gm][:, j]) for j in range(12)])), 4),
    "gold_n": int(gm.sum()), "minutes": round((time.time() - t0) / 60, 1)}
log["gold_per_label"] = {L: round(float(auc(Y[gm][:, j], oof[gm][:, j])), 3) for j, L in enumerate(LABELS)}
pd.DataFrame({"StudyInstanceUID": ids[done], **{L: oof[done][:, j] for j, L in enumerate(LABELS)}}) \
  .to_csv(f"{a.out_dir}/oof_head.csv", index=False)
json.dump(log, open(f"{a.out_dir}/head_log.json", "w"), indent=1)
print("\nSUMMARY\n" + json.dumps(log["overall"], indent=1))
print(json.dumps(log["gold_per_label"], indent=1))
