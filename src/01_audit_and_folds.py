# %% [markdown]
# # RSNA Knee — 01: Data audit + site-grouped CV folds
#
# **How to run:** Kaggle → New Notebook → File → Import Notebook (this file) → Add Input → the competition data.
# CPU session is fine (no GPU needed). Internet can be off. Expect roughly 10–25 min (DICOM header scan).
#
# **What it does**
# 1. Finds the data and describes every CSV (shapes, column names, dtypes — never row contents).
# 2. Maps the 12 label columns and reports how many studies are fully labelled + prevalence.
# 3. Reads one DICOM header per series (no pixels) → plane, laterality, scanner fingerprint, pixel spacing.
# 4. Builds 5 folds grouped by scanner fingerprint *and* patient, balanced on the gold-labelled studies.
#
# **Outputs** in `/kaggle/working/`: `series_headers.csv`, `study_meta.csv`, `folds.csv`, `summary.json`.
# Save the notebook version so later notebooks can attach these outputs as input.
#
# **Privacy:** prints counts only — no report text, no rows. Paste the final SUMMARY block back to Claude.

# %%
import os, glob, json, re, time, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom

N_FOLDS = 5
SEED = 42
OUT = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)

LABELS = {  # canonical name -> normalized column-name keys to match
    "ACL": ["acl"], "MCL": ["mcl"],
    "Medial Meniscus": ["medialmeniscus"], "Lateral Meniscus": ["lateralmeniscus"],
    "Medial OA": ["medialoa"], "Lateral OA": ["lateraloa"], "PF OA": ["pfoa", "patellofemoraloa"],
    "Effusion": ["effusion"], "Synovitis": ["synovitis"], "Baker's": ["baker", "bakers"],
    "Contusion": ["contusion"], "Fracture": ["fracture"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")

def show_tree(root, max_depth=3, max_entries=15):
    """Print folder layout (names only) so a missing/odd data mount can be diagnosed."""
    if not os.path.exists(root):
        print(f"{root} does not exist"); return
    for cur, dirs, files in os.walk(root, followlinks=True):
        depth = cur[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []; continue
        dirs.sort()
        ind = "  " * depth
        print(f"{ind}{os.path.basename(cur) or cur}/  ({len(dirs)} dirs, {len(files)} files)")
        for f in sorted(files)[:max_entries]:
            print(f"{ind}  {f}")
        if len(dirs) > max_entries:  # don't descend into thousands of study folders
            print(f"{ind}  ... {len(dirs)} subfolders (not expanded)")
            dirs[:] = []

def find_data_dir():
    if os.environ.get("KNEE_DATA_DIR"):
        return os.environ["KNEE_DATA_DIR"]
    hits = []
    for cur, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        depth = cur[len(INPUT_ROOT):].count(os.sep)
        if depth > 5 or len(dirs) > 200:   # stop descending into image folders
            dirs[:] = []
        for f in files:
            if f.lower().endswith(".csv"):
                hits.append(os.path.join(cur, f))
    def score(p):
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except Exception:
            return (0, 0, 0)
        return ("StudyInstanceUID" in cols, "train" in os.path.basename(p).lower(), "knee" in p.lower())
    if hits:
        best = max(hits, key=score)
        if score(best)[0]:
            return os.path.dirname(best)
    print("Could not find the competition CSVs. Here is what is under /kaggle/input:\n")
    show_tree(INPUT_ROOT)
    raise FileNotFoundError(
        "Competition data not found. In the notebook's right-hand panel click 'Add Input' -> "
        "Competitions -> 'RSNA Knee Abnormality Detection' (you must have joined and accepted the rules), "
        "then Save Version again. If it IS attached, paste the folder listing above back to Claude.")

DATA = find_data_dir()
print("DATA_DIR:", DATA)
for e in sorted(os.scandir(DATA), key=lambda e: e.name):
    if e.is_dir():
        print(f"  [dir]  {e.name}/  ({sum(1 for _ in os.scandir(e.path))} entries)")
    else:
        print(f"  [file] {e.name}  ({e.stat().st_size/1e6:.1f} MB)")

# %% [markdown]
# ## 1. CSV schema

# %%
csvs = {os.path.basename(p): pd.read_csv(p, low_memory=False) for p in sorted(glob.glob(os.path.join(DATA, "*.csv")))}
for name, df in csvs.items():
    print(f"\n== {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    for c in df.columns:
        print(f"   {c:40s} {str(df[c].dtype):8s} nonnull={df[c].notna().sum():6d} nunique={df[c].nunique():6d}")

def label_map(df):
    m = {}
    for canon, keys in LABELS.items():
        for c in df.columns:
            n = norm(c)
            if any(n == k or n.startswith(k) for k in keys):
                m[canon] = c
                break
    return m

study_csvs = {n: d for n, d in csvs.items() if "StudyInstanceUID" in d.columns}
train_name = max(study_csvs, key=lambda n: (len(label_map(study_csvs[n])), "train" in n.lower()))
train = study_csvs[train_name].copy()
LMAP = label_map(train)
print(f"\nTrain table: {train_name}")
print("Label column mapping:", json.dumps(LMAP, indent=1))
missing = [k for k in LABELS if k not in LMAP]
if missing:
    print("WARNING — unmatched labels:", missing)

# report column = object column with the longest average text (stats only, never content)
obj_cols = [c for c in train.columns if train[c].dtype == object and c != "StudyInstanceUID"]
REPORT_COL = max(obj_cols, key=lambda c: train[c].dropna().astype(str).str.len().mean()) if obj_cols else None
if REPORT_COL:
    wc = train[REPORT_COL].dropna().astype(str).str.split().str.len()
    print(f"Report column guess: '{REPORT_COL}'  nonnull={train[REPORT_COL].notna().sum()}  "
          f"words median={wc.median():.0f} p90={wc.quantile(.9):.0f}")

lab_cols = [LMAP[k] for k in LABELS if k in LMAP]
gold_mask = train[lab_cols].notna().all(axis=1) if lab_cols else pd.Series(False, index=train.index)
gold_ids = set(train.loc[gold_mask, "StudyInstanceUID"])
print(f"\nStudies: {len(train)}  fully labelled (gold): {len(gold_ids)}  partially labelled: "
      f"{int((train[lab_cols].notna().any(axis=1) & ~gold_mask).sum()) if lab_cols else 0}")
prevalence = {k: round(float(train.loc[gold_mask, LMAP[k]].mean()), 3) for k in LABELS if k in LMAP}
print("Gold prevalence:", prevalence)

series_csv_name = next((n for n, d in csvs.items() if "SeriesInstanceUID" in d.columns), None)
if series_csv_name:
    s = csvs[series_csv_name]
    print(f"\nSeries table: {series_csv_name}")
    for c in s.columns:
        if s[c].nunique() <= 20:
            print(f"   {c}: {s[c].value_counts(dropna=False).to_dict()}")
    cat = [c for c in s.columns if 2 <= s[c].nunique() <= 5]
    if len(cat) >= 2:
        print("   cross-tab of low-cardinality columns:")
        print(s.groupby(cat).size().to_string())

# %% [markdown]
# ## 2. DICOM header scan (one header per series, no pixels)

# %%
TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "PatientID", "PatientSex", "PatientAge",
        "Laterality", "ImageLaterality", "BodyPartExamined", "SeriesDescription",
        "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "MagneticFieldStrength",
        "ReceiveCoilName", "ImagingFrequency", "InstitutionName", "StationName",
        "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "Rows", "Columns",
        "RepetitionTime", "EchoTime", "ImageOrientationPatient"]

def find_image_root(split):
    # image folder named like train_images / test / test_images, directly under DATA or one level down
    cands = []
    for e in os.scandir(DATA):
        if e.is_dir():
            if split in e.name.lower():
                cands.append(e.path)
            else:
                cands += [s.path for s in os.scandir(e.path) if s.is_dir() and split in s.name.lower()]
    return sorted(cands, key=len)[0] if cands else None

def tagval(ds, t):
    v = getattr(ds, t, None)
    if v is None or v == "":
        return None
    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        return "\\".join(str(x) for x in v)
    return str(v)

def plane_from_iop(iop):
    try:
        v = np.array([float(x) for x in iop.split("\\")])
        n = np.cross(v[:3], v[3:6])
        return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(n)))]
    except Exception:
        return None

def scan_study(study_dir):
    rows = []
    for root, _, files in os.walk(study_dir):
        dcms = sorted(f for f in files if f.lower().endswith(".dcm"))
        if not dcms:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root, dcms[len(dcms) // 2]), stop_before_pixels=True)
            r = {t: tagval(ds, t) for t in TAGS}
        except Exception as ex:
            r = {t: None for t in TAGS}
            r["read_error"] = type(ex).__name__
        r["study_dir"] = os.path.basename(study_dir)
        r["series_dir"] = os.path.basename(root)
        r["n_slices"] = len(dcms)
        rows.append(r)
    return rows

def scan_split(split):
    root = find_image_root(split)
    if root is None:
        print(f"No {split} image dir found."); return pd.DataFrame()
    studies = [e.path for e in os.scandir(root) if e.is_dir()]
    print(f"{split}: {root}  studies={len(studies)}")
    t0, rows = time.time(), []
    with ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(scan_study, studies)):
            rows.extend(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(studies)} studies  {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df["split"] = split
    df["StudyInstanceUID"] = df["StudyInstanceUID"].fillna(df["study_dir"])
    df["iop_plane"] = df["ImageOrientationPatient"].map(lambda x: plane_from_iop(x) if x else None)
    print(f"  done: {len(df)} series in {time.time()-t0:.0f}s")
    return df

hdr = pd.concat([scan_split("train"), scan_split("test")], ignore_index=True)
hdr.to_csv(f"{OUT}/series_headers.csv", index=False)

# %% [markdown]
# ## 3. Study-level metadata: fingerprint, laterality, patient

# %%
def fp(r):
    fs = r["MagneticFieldStrength"]
    try: fs = f"{float(fs):.1f}"
    except Exception: pass
    return "|".join(str(r[t]) for t in ["Manufacturer", "ManufacturerModelName", "SoftwareVersions",
                                        "ReceiveCoilName", "InstitutionName", "StationName"]) + f"|{fs}"

hdr["fingerprint"] = hdr.apply(fp, axis=1)

def lat_set(g):
    vals = set(g["Laterality"].dropna()) | set(g["ImageLaterality"].dropna())
    return "".join(sorted(v for v in vals if v in ("L", "R"))) or "?"

study = hdr.groupby("StudyInstanceUID").apply(lambda g: pd.Series({
    "split": g["split"].iloc[0],
    "fingerprint": g["fingerprint"].mode().iloc[0],
    "n_fingerprints": g["fingerprint"].nunique(),
    "PatientID": g["PatientID"].dropna().mode().iloc[0] if g["PatientID"].notna().any() else None,
    "PatientSex": g["PatientSex"].dropna().mode().iloc[0] if g["PatientSex"].notna().any() else None,
    "laterality": lat_set(g),
    "n_series": len(g),
    "planes": ",".join(sorted(g["iop_plane"].dropna().unique())),
    "min_pixel_spacing": pd.to_numeric(g["PixelSpacing"].str.split("\\").str[0], errors="coerce").min(),
})).reset_index()
study["is_gold"] = study["StudyInstanceUID"].isin(gold_ids)
study.to_csv(f"{OUT}/study_meta.csv", index=False)

tr = study[study.split == "train"]
print("Train studies with headers:", len(tr), "| in train csv:", tr.StudyInstanceUID.isin(train.StudyInstanceUID).sum())
print("Laterality:", tr.laterality.value_counts().to_dict())
print("Distinct fingerprints (train):", tr.fingerprint.nunique(),
      "| top-20 cover:", f"{tr.fingerprint.value_counts().head(20).sum()/len(tr):.1%}",
      "| largest:", f"{tr.fingerprint.value_counts().iloc[0]/len(tr):.1%}")
pid_counts = tr.PatientID.value_counts()
print("Patients with >1 train study:", int((pid_counts > 1).sum()), "| max studies per patient:", int(pid_counts.max()) if len(pid_counts) else 0)
te = study[study.split == "test"]
if len(te):
    unseen = (~te.fingerprint.isin(tr.fingerprint)).mean()
    print(f"Test studies visible: {len(te)} | with fingerprint unseen in train: {unseen:.1%}")

# %% [markdown]
# ## 4. Grouped folds (union of scanner fingerprint + patient), balanced on gold studies

# %%
tr = tr.copy()
# ignore PatientIDs that look like placeholders (shared by >2% of studies)
bad_pid = set(pid_counts[pid_counts > max(5, 0.02 * len(tr))].index)

def build_groups(link_patients):
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for _, r in tr.iterrows():
        parent[find("S:" + r.StudyInstanceUID)] = find("F:" + r.fingerprint)
        if link_patients and r.PatientID and r.PatientID not in bad_pid:
            parent[find("S:" + r.StudyInstanceUID)] = find("P:" + r.PatientID)
    return tr.StudyInstanceUID.map(lambda s: find("S:" + s))

# Prefer fingerprint+patient groups; if that chains into a mega-group (> 1/(N_FOLDS+1) of data), fall back to fingerprint only.
GROUP_MODE = "fingerprint+patient"
tr["group"] = build_groups(True)
if tr.group.value_counts().iloc[0] / len(tr) > 1 / (N_FOLDS + 1):
    GROUP_MODE = "fingerprint_only"
    tr["group"] = build_groups(False)
print("Grouping mode:", GROUP_MODE)

g = tr.groupby("group").agg(n=("StudyInstanceUID", "size"), gold=("is_gold", "sum")).sample(frac=1, random_state=SEED)
g = g.sort_values(["gold", "n"], ascending=False)
n_tot, gold_tot = len(tr), max(int(tr.is_gold.sum()), 1)
load = np.zeros((N_FOLDS, 2))
assign = {}
for grp, row in g.iterrows():
    cost = (load[:, 0] + row.n) / n_tot + (load[:, 1] + row.gold) / gold_tot
    f = int(np.argmin(cost))
    assign[grp] = f
    load[f] += [row.n, row.gold]
tr["fold"] = tr.group.map(assign)
tr[["StudyInstanceUID", "fold", "group", "fingerprint", "laterality", "is_gold"]].to_csv(f"{OUT}/folds.csv", index=False)
print("Groups:", tr.group.nunique(), "| largest group share:", f"{g.n.max()/n_tot:.1%}")
print(tr.groupby("fold").agg(studies=("StudyInstanceUID", "size"), gold=("is_gold", "sum"),
                              groups=("group", "nunique")).to_string())

# %% [markdown]
# ## SUMMARY — paste this block back to Claude

# %%
summary = {
    "data_dir_entries": sorted(os.listdir(DATA)),
    "csvs": {n: {"shape": list(d.shape), "columns": list(d.columns)} for n, d in csvs.items()},
    "train_csv": train_name, "label_map": LMAP, "report_col": REPORT_COL,
    "n_train": len(train), "n_gold": len(gold_ids), "gold_prevalence": prevalence,
    "n_series_scanned": int(len(hdr)), "header_read_errors": int(hdr.get("read_error", pd.Series()).notna().sum()),
    "tag_coverage": {t: round(float(hdr[t].notna().mean()), 3) for t in TAGS},
    "laterality": tr.laterality.value_counts().to_dict(),
    "n_fingerprints": int(tr.fingerprint.nunique()), "n_groups": int(tr.group.nunique()), "group_mode": GROUP_MODE,
    "patients_multi_study": int((pid_counts > 1).sum()),
    "largest_group_share": round(float(g.n.max() / n_tot), 3),
    "fold_sizes": tr.groupby("fold").size().to_dict(), "fold_gold": tr.groupby("fold").is_gold.sum().astype(int).to_dict(),
    "slices_per_series": hdr.n_slices.describe().round(1).to_dict(),
    "min_pixel_spacing_quantiles": tr.min_pixel_spacing.quantile([.05, .5, .95]).round(3).to_dict(),
    "n_test_visible": int(len(te)),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
