# %% [markdown]
# # RSNA Knee — 03: Build compact training volumes from the DICOMs (CPU only)
#
# **Settings:** Accelerator **None (CPU)** · Internet **On** (installs JPEG decoders) · Input: the competition data.
# Expect roughly 1.5–4 hours. Use **Save Version → Save & Run All**.
#
# **What it does:** for every training study, picks the best series for each of 5 slots
# (sagittal fluid / sagittal non-fluid / coronal fluid / coronal non-fluid / axial fluid), puts every series into the same
# standard orientation, crops a constant 140 mm field of view, resizes to 256×256, keeps 20 evenly spaced slices,
# and stores the result as uint8. 570 GB of DICOMs → about 10–12 GB the training notebook can read quickly.
#
# **Outputs** in `/kaggle/working/`: `vols/shard_XXX.npz` (one array per study), `vol_index.csv`, `prep_summary.json`,
# and `preview.png` (a few slices, for you to eyeball — don't share it, it is competition image data).
#
# If the run would exceed Kaggle's 12-hour limit, set `N_PARTS = 2` and run it twice (PART = 0, then a copy with PART = 1).

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls"],
               check=False)

# %%
#@@PREP_COMMON@@

# %%
import json, time, collections
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

N_PARTS, PART = 1, 0
SHARD_SIZE = 100
OUT = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")
os.makedirs(f"{OUT}/vols", exist_ok=True)

def find_file(name):
    for cur, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        if len(dirs) > 200:
            dirs[:] = []
        if name in files:
            return os.path.join(cur, name)

series_csv = find_file("train_series.csv")
assert series_csv, "Competition data not attached"
DATA = os.path.dirname(series_csv)
IMG_ROOT = os.path.join(DATA, "train_series")
ser = pd.read_csv(series_csv)
groups = {s: list(zip(g.SeriesInstanceUID, g.Fluid_Sensitive)) for s, g in ser.groupby("StudyInstanceUID")}
studies = sorted(groups)
studies = [s for i, s in enumerate(studies) if i % N_PARTS == PART]
shards = [studies[i:i + SHARD_SIZE] for i in range(0, len(studies), SHARD_SIZE)]
print(f"{len(studies)} studies in this part -> {len(shards)} shards. CPUs: {os.cpu_count()}")

def work(uid):
    t = time.time()
    try:
        vol, mask, info = process_study(os.path.join(IMG_ROOT, uid), groups[uid])
        info["error"] = None
    except Exception as ex:
        vol = np.zeros((len(SLOT_NAMES), N_SLICES, IMG, IMG), np.uint8)
        mask = np.zeros(len(SLOT_NAMES), np.uint8)
        info = {"error": f"{type(ex).__name__}: {str(ex)[:80]}", "ts": set(), "bad_pixels": 0, "slot_n": [0] * 5, "n_series": 0}
    info["sec"] = time.time() - t
    return uid, vol, mask, info

# %%
INDEX = f"{OUT}/vol_index.csv"
index = pd.read_csv(INDEX).to_dict("records") if os.path.exists(INDEX) else []
ts_counter, t0, done = collections.Counter(), time.time(), 0
with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
    for k, shard in enumerate(shards):
        sid = PART * 1000 + k
        path = f"{OUT}/vols/shard_{sid:04d}.npz"
        if os.path.exists(path):
            done += len(shard); continue
        arrays, rows = {}, []
        for uid, vol, mask, info in ex.map(work, shard):
            arrays[uid] = vol
            ts_counter.update(info["ts"])
            rows.append({"StudyInstanceUID": uid, "shard": sid, **{f"m_{n}": int(mask[i]) for i, n in enumerate(SLOT_NAMES)},
                         **{f"n_{n}": info["slot_n"][i] for i, n in enumerate(SLOT_NAMES)},
                         "bad_pixels": info["bad_pixels"], "error": info["error"], "sec": round(info["sec"], 2)})
        np.savez_compressed(path, **arrays)
        index += rows
        pd.DataFrame(index).to_csv(INDEX, index=False)
        done += len(shard)
        el = time.time() - t0
        print(f"shard {k+1}/{len(shards)}  studies {done}/{len(studies)}  {el/60:.1f} min  "
              f"~{el/max(done,1)*(len(studies)-done)/60:.0f} min left  shard {os.path.getsize(path)/1e6:.0f} MB")

# %% [markdown]
# ## Preview (look at it yourself; don't share it)
# Check: **sagittal** — kneecap on the left, thigh at the top. **Coronal** — thigh at the top, shin at the bottom.
# **Axial** — kneecap at the top. Every row should look the same way round.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
first = np.load(sorted(glob.glob(f"{OUT}/vols/shard_*.npz"))[0])
keys = list(first.keys())[:4]
fig, axes = plt.subplots(len(keys), 5, figsize=(15, 3 * len(keys)))
for r, kk in enumerate(keys):
    v = first[kk]
    for c in range(5):
        axes[r, c].imshow(v[c, N_SLICES // 2], cmap="gray"); axes[r, c].axis("off")
        if r == 0:
            axes[r, c].set_title(SLOT_NAMES[c])
plt.tight_layout(); plt.savefig(f"{OUT}/preview.png", dpi=70); plt.show()

# %% [markdown]
# ## SUMMARY — paste this back to Claude

# %%
idx = pd.DataFrame(index)
summary = {
    "studies_done": int(len(idx)), "parts": [PART, N_PARTS],
    "errors": int(idx.error.notna().sum()), "error_examples": idx.error.dropna().value_counts().head(5).to_dict(),
    "slot_coverage": {n: round(float(idx[f"m_{n}"].mean()), 3) for n in SLOT_NAMES},
    "studies_with_no_slots": int((idx[[f"m_{n}" for n in SLOT_NAMES]].sum(axis=1) == 0).sum()),
    "slices_per_chosen_series_median": {n: float(idx.loc[idx[f"m_{n}"] == 1, f"n_{n}"].median()) for n in SLOT_NAMES},
    "studies_with_unreadable_slices": int((idx.bad_pixels > 0).sum()),
    "transfer_syntaxes": dict(ts_counter.most_common(8)),
    "sec_per_study_median": float(idx.sec.median()),
    "total_minutes": round((time.time() - t0) / 60, 1),
    "output_gb": round(sum(os.path.getsize(p) for p in glob.glob(f"{OUT}/vols/*.npz")) / 1e9, 2),
}
json.dump(summary, open(f"{OUT}/prep_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
