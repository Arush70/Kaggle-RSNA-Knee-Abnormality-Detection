# ---- shared preprocessing: identical code is used for training volumes and at submission time ----
import os, glob
import numpy as np
import pydicom
import cv2
from concurrent.futures import ThreadPoolExecutor

# 5 slots = plane x (fluid-sensitive / not). Axial non-fluid exists in only ~19% of studies -> used only as axial fallback.
SLOT_NAMES = ["sag_fs", "sag_nfs", "cor_fs", "cor_nfs", "ax_fs"]
SLOT_INDEX = {("sagittal", 1): 0, ("sagittal", 0): 1, ("coronal", 1): 2, ("coronal", 0): 3, ("axial", 1): 4}
N_SLICES = 20          # slices kept per slot (evenly spaced through the series)
IMG = 256              # output pixels per side
CROP_MM = 140.0        # constant physical field of view -> 0.55 mm/px at 256
MAX_SERIES_SLICES = 80 # prefer 2D series over very long 3D series when both exist

# Standard radiological display per plane, in DICOM patient coords (x = patient left, y = posterior, z = head):
# (axis that image columns run along, its sign, axis that image rows run along, its sign, through-plane slice axis)
TARGET = {"sagittal": (1, +1, 2, -1, 0),   # front of knee on image left, head up
          "coronal":  (0, +1, 2, -1, 1),   # patient's right on image left, head up
          "axial":    (0, +1, 1, +1, 2)}   # patient's right on image left, front up

def plane_of(iop):
    v = np.asarray(iop, dtype=float)
    n = np.cross(v[:3], v[3:6])
    return ["sagittal", "coronal", "axial"][int(np.argmax(np.abs(n)))]

def _read_header(f):
    try:
        ds = pydicom.dcmread(f, stop_before_pixels=True,
                             specific_tags=["ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing",
                                            "Rows", "Columns", "InstanceNumber"])
        return f, [float(x) for x in ds.ImageOrientationPatient], [float(x) for x in ds.ImagePositionPatient], \
               [float(x) for x in ds.PixelSpacing]
    except Exception:
        return f, None, None, None

def _read_pixels(f):
    try:
        ds = pydicom.dcmread(f)
        a = ds.pixel_array.astype(np.float32)
        if a.ndim == 3:  # multi-frame or colour: take first frame / channel
            a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
        return a, str(ds.file_meta.TransferSyntaxUID) if hasattr(ds, "file_meta") else "?"
    except Exception as ex:
        return None, type(ex).__name__

def series_headers(series_dir, pool):
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm"))) or \
            sorted(f for f in glob.glob(os.path.join(series_dir, "*")) if os.path.isfile(f))
    hdrs = [h for h in pool.map(_read_header, files) if h[1] is not None]
    if not hdrs:
        return None
    # keep the majority orientation (drops localisers mixed into a series)
    keys = [tuple(np.round(h[1], 1)) for h in hdrs]
    best = max(set(keys), key=keys.count)
    hdrs = [h for h, k in zip(hdrs, keys) if k == best]
    plane = plane_of(hdrs[0][1])
    hdrs.sort(key=lambda h: h[2][TARGET[plane][4]])
    return {"plane": plane, "hdrs": hdrs, "n": len(hdrs)}

def reorient(img, r, c, spacing, plane):
    """Flip/transpose so every series is displayed in the same standard orientation."""
    ra, rs, ca, cs, _ = TARGET[plane]
    r, c = np.asarray(r), np.asarray(c)
    sp = list(spacing)  # [row spacing (between rows), column spacing]
    if int(np.argmax(np.abs(r))) != ra:
        img, r, c, sp = img.T, c, r, sp[::-1]
    if np.sign(r[ra]) != rs:
        img = img[:, ::-1]
    if np.sign(c[ca]) != cs:
        img = img[::-1, :]
    return np.ascontiguousarray(img), sp

def crop_resize(img, sp):
    h, w = img.shape
    hr, hc = CROP_MM / 2 / max(sp[0], 1e-3), CROP_MM / 2 / max(sp[1], 1e-3)
    cy, cx = h / 2, w / 2
    y0, y1, x0, x1 = int(round(cy - hr)), int(round(cy + hr)), int(round(cx - hc)), int(round(cx + hc))
    pad = max(0, -y0, -x0, y1 - h, x1 - w)
    if pad:
        img = np.pad(img, pad)
        y0, y1, x0, x1 = y0 + pad, y1 + pad, x0 + pad, x1 + pad
    crop = img[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if crop.shape[0] > IMG else cv2.INTER_LINEAR
    return cv2.resize(crop, (IMG, IMG), interpolation=interp)

def build_slot(info, pool):
    hdrs = info["hdrs"]
    idx = np.round(np.linspace(0, len(hdrs) - 1, N_SLICES)).astype(int)
    uniq = sorted(set(idx.tolist()))
    pix = dict(zip(uniq, pool.map(lambda i: _read_pixels(hdrs[i][0]), uniq)))
    out, ts = [], set()
    for i in idx:
        a, t = pix[i]
        ts.add(t)
        if a is None:
            out.append(np.zeros((IMG, IMG), np.float32)); continue
        _, iop, _, sp = hdrs[i]
        a, sp2 = reorient(a, iop[:3], iop[3:], sp, info["plane"])
        out.append(crop_resize(a, sp2))
    vol = np.stack(out)
    nz = vol[vol > 0]
    lo, hi = (np.percentile(nz, [1, 99.5]) if nz.size > 100 else (0.0, 1.0))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return vol.astype(np.uint8), ts, sum(pix[i][0] is None for i in uniq)

def process_study(study_dir, series_rows):
    """series_rows: list of (SeriesInstanceUID, Fluid_Sensitive). Returns (vol [5,N,IMG,IMG] uint8, mask [5], info)."""
    vol = np.zeros((len(SLOT_NAMES), N_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros(len(SLOT_NAMES), np.uint8)
    info = {"n_series": len(series_rows), "plane_mismatch": 0, "bad_pixels": 0, "ts": set(), "slot_n": [0] * 5}
    cands = {}
    with ThreadPoolExecutor(8) as pool:
        for uid, fluid in series_rows:
            sdir = os.path.join(study_dir, str(uid))
            if not os.path.isdir(sdir):
                hits = [d for d, _, _ in os.walk(study_dir) if os.path.basename(d) == str(uid)]
                if not hits:
                    continue
                sdir = hits[0]
            h = series_headers(sdir, pool)
            if h is None:
                continue
            key = (h["plane"], int(fluid))
            slot = SLOT_INDEX.get(key, 4 if h["plane"] == "axial" else None)
            if slot is None:
                continue
            fallback = key not in SLOT_INDEX
            score = (not fallback, h["n"] <= MAX_SERIES_SLICES, -abs(h["n"] - 30))
            if slot not in cands or score > cands[slot][0]:
                cands[slot] = (score, h)
        for slot, (_, h) in cands.items():
            v, ts, bad = build_slot(h, pool)
            vol[slot], mask[slot] = v, 1
            info["ts"] |= ts
            info["bad_pixels"] += bad
            info["slot_n"][slot] = h["n"]
    return vol, mask, info
