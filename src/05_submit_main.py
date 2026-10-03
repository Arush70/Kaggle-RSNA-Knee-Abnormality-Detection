# %% [markdown]
# # RSNA Knee — 05: Submission (runs on Kaggle's hidden test set, internet OFF)
#
# **Settings:** Accelerator **GPU T4 x2** · Internet **Off** (required for submission) ·
# Inputs: competition data + the output of notebook **04** (or the dataset holding your `model_f*.pt` files).
#
# It preprocesses each test study exactly like notebook 03, runs every `model_f*.pt` it finds, averages them,
# and writes `submission.csv`. When you commit it, it runs on the 3 sample test studies; when you **Submit**,
# Kaggle reruns it on the ~1,300 hidden studies.

# %%
#@@PREP_COMMON@@

# %%
#@@MODEL_COMMON@@

# %%
import os, glob, time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")
OUT = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
BATCH = 4
TTA = True   # average original + mirrored prediction
t0 = time.time()

def find(name):
    for cur, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        if len(dirs) > 200:
            dirs[:] = []
        if name in files:
            return os.path.join(cur, name)

DATA = os.path.dirname(find("test_series.csv"))
test_ser = pd.read_csv(os.path.join(DATA, "test_series.csv"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
test_ids = pd.read_csv(os.path.join(DATA, "test.csv")).StudyInstanceUID.tolist()
groups = {s: list(zip(g.SeriesInstanceUID, g.Fluid_Sensitive)) for s, g in test_ser.groupby("StudyInstanceUID")}
IMG_ROOT = os.path.join(DATA, "test_series")
print(f"test studies: {len(test_ids)}")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpts = sorted(glob.glob(os.environ.get("KNEE_MODEL_GLOB", f"{INPUT_ROOT}/**/model_f*.pt"), recursive=True))
assert ckpts, "No model_f*.pt found — attach notebook 04's output (or your model dataset) as an input."
models, sidx_list = [], []
for c in ckpts:
    ck = torch.load(c, map_location="cpu", weights_only=False)
    cf = ck["cfg"]
    m = KneeNet(cf["backbone"], pretrained=False, grad_ckpt=False, freeze_blocks=cf.get("freeze_blocks", 0),
                img_size=cf.get("img_size") if ("vit" in cf["backbone"] or "dinov2" in cf["backbone"]) else None)
    m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
    models.append(m); sidx_list.append(slice_indices(N_SLICES, ck["cfg"]["n_slices_use"]))
    print("loaded", os.path.relpath(c, INPUT_ROOT), "| backbone", ck["cfg"]["backbone"], "| epoch", ck.get("epoch"))
amp_dtype = torch.float16 if dev.type == "cuda" else torch.bfloat16

def prep(uid):
    try:
        vol, mask, _ = process_study(os.path.join(IMG_ROOT, uid), groups.get(uid, []))
    except Exception:
        vol = np.zeros((len(SLOT_NAMES), N_SLICES, IMG, IMG), np.uint8); mask = np.zeros(len(SLOT_NAMES), np.uint8)
    return uid, vol, mask

# %%
preds = {}
def flush(batch):
    x = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
    mk = torch.from_numpy(np.stack([b[2] for b in batch]).astype(np.float32)).to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
        p = np.mean([predict_tta(m, x, mk, s, TTA).cpu().numpy() for m, s in zip(models, sidx_list)], 0)
    for (uid, _, msk), row in zip(batch, p):
        preds[uid] = row if msk.sum() > 0 else None

batch = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    for i, item in enumerate(ex.map(prep, test_ids)):
        batch.append(item)
        if len(batch) == BATCH:
            flush(batch); batch = []
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_ids)}  {(time.time()-t0)/60:.1f} min")
if batch:
    flush(batch)

ok = [v for v in preds.values() if v is not None]
fallback = np.mean(ok, 0) if ok else np.full(len(LABELS), 0.3)
rows = [{"StudyInstanceUID": u, **dict(zip(LABELS, preds.get(u) if preds.get(u) is not None else fallback))} for u in test_ids]
sub = pd.DataFrame(rows)[sample.columns]
os.makedirs(OUT, exist_ok=True)
sub.to_csv(os.path.join(OUT, os.environ.get("KNEE_OUT_NAME", "submission.csv")), index=False)
print(sub.shape, "| studies with no usable images:", sum(v is None for v in preds.values()),
      f"| total {(time.time()-t0)/60:.1f} min")
print(sub.head())
