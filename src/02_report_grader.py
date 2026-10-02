# %% [markdown]
# # RSNA Knee — 02: Report → severity grades with an open-weights LLM (runs entirely inside Kaggle)
#
# **Settings (right-hand panel):** Accelerator **GPU T4 x2** · Internet **On** (to download the model; this is a
# development notebook, not the submission) · Add Input: the competition data, and optionally the output of notebook 01.
# Expect roughly 45–90 min. Use **Save Version → Save & Run All** so it runs in the background.
#
# **What it does**
# 1. Loads Qwen2.5-7B-Instruct (multilingual) in fp16 across both T4s.
# 2. For every report, asks for a 0–4 severity grade per finding against the host's strict image-based thresholds,
#    plus which knee (L/R/both) the report describes.
# 3. Grades the 58 gold studies first and prints per-label AUC vs the radiologist labels, so we know quickly if it works.
# 4. Grades the remaining ~4,350 reports, checkpointing as it goes.
#
# **Grades:** 0 = explicitly normal · 1 = not mentioned · 2 = present but below threshold (mild / low-grade / degenerative)
# · 3 = probable or borderline · 4 = definite and meets the threshold.
#
# **Outputs** in `/kaggle/working/`: `report_grades.csv` (StudyInstanceUID, side, 12 grades) and `grader_summary.json`.
#
# **Privacy:** report text never leaves Kaggle and is never printed. Paste the SUMMARY block back to Claude.

# %%
import os, re, json, time, glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BATCH = 16                # reports per generation batch (sorted by length, so padding is small)
MAX_REPORT_TOKENS = 1500  # longer reports are truncated (p90 is ~290 words, so this rarely bites)
MAX_NEW_TOKENS = 110      # the JSON answer is ~80 tokens
MOCK = bool(os.environ.get("KNEE_MOCK"))  # offline test mode (no model); leave unset on Kaggle
OUT = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
INPUT_ROOT = os.environ.get("KNEE_INPUT_ROOT", "/kaggle/input")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("HF_HOME", "/tmp/hf")  # model cache off the 20 GB working disk

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa",
        "effusion", "synovitis", "baker", "contusion", "fracture"]
K2L = dict(zip(KEYS, LABELS))

def find_file(name, must_have_col=None):
    hits = []
    for cur, dirs, files in os.walk(INPUT_ROOT, followlinks=True):
        if len(dirs) > 200:
            dirs[:] = []
        if name in files:
            p = os.path.join(cur, name)
            if must_have_col is None or must_have_col in pd.read_csv(p, nrows=0).columns:
                hits.append(p)
    return sorted(hits, key=len)[0] if hits else None

train_path = find_file("train.csv", "Report")
assert train_path, "Competition data not attached: Add Input -> Competitions -> RSNA Knee Abnormality Detection"
train = pd.read_csv(train_path)
gold_mask = train[LABELS].notna().all(axis=1)
print(f"train.csv: {len(train)} studies, {int(gold_mask.sum())} gold")

# %% [markdown]
# ## Prompt — the host's grading rubric, condensed

# %%
SYSTEM = """You are a senior musculoskeletal radiologist. You will read one knee MRI report (it may be in any language) and grade each finding against a STRICT image-based rubric used by expert readers. Borderline or "on the fence" findings do not meet the threshold.

Grade scale for every finding:
0 = explicitly normal / intact / absent
1 = not mentioned
2 = present but BELOW threshold (mild, minimal, small, trace, low-grade, degenerative, sprain, signal change only)
3 = probable, suspected or borderline relative to the threshold
4 = definite AND meets the threshold

Thresholds for grade 4:
- acl: high-grade partial (>50% of fibres) or complete ACL tear/rupture. Mucoid degeneration, sprain, thickening, signal change, or an intact reconstruction graft = 2. Torn graft = 4.
- mcl: high-grade partial or complete ACUTE MCL tear. Low-grade (grade 1) sprain, periligamentous oedema, chronic thickening or old injury = 2.
- med_men / lat_men (medial / lateral meniscus): tear reaching the articular surface (grade 3 signal), or a morphologic abnormality (displaced fragment, bucket-handle, flap, radial, root tear, maceration, extrusion with tear, partial meniscectomy with re-tear). Intrasubstance degeneration (grade 1-2 signal) not reaching the surface = 2.
- med_oa / lat_oa / pf_oa (medial femorotibial / lateral femorotibial / patellofemoral incl. patella and trochlea): high-grade cartilage loss (>50% thickness: ICRS or Outerbridge grade 3-4, full-thickness defect, exposed or denuded bone) over a moderate or large area (about 1 cm or more), or advanced osteoarthritis of that compartment. Chondromalacia grade 1-2, superficial fissuring, mild thinning = 2. Small focal deep defect under 1 cm = 3.
- effusion: moderate or large joint effusion. Small, trace, minimal or physiological fluid = 2.
- synovitis: definite synovitis / synovial thickening or hypertrophy (incl. Hoffa-synovitis when described as synovitis). Mild or questionable = 2.
- baker: moderate or large popliteal (Baker) cyst, or a ruptured/leaking Baker cyst. Small = 2.
- contusion: traumatic bone marrow oedema / bone bruise without a discrete fracture line. Degenerative subchondral oedema next to cartilage loss, or reactive oedema = 2.
- fracture: acute fracture line or cortical break (incl. osteochondral, impaction, avulsion, Segond, tibial plateau). Old/healed fracture or stress reaction without a line = 2.

Also give "side": the knee described: "R", "L", "B" (both knees) or "U" (not stated).

Reply with JSON only, exactly these keys, integer grades:
{"side":"U","acl":1,"mcl":1,"med_men":1,"lat_men":1,"med_oa":1,"lat_oa":1,"pf_oa":1,"effusion":1,"synovitis":1,"baker":1,"contusion":1,"fracture":1}"""

def parse(text):
    out = {"side": "U", "parse_ok": 0, **{k: 1 for k in KEYS}}
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0))  # raises if no {...} found or JSON is malformed
        out["parse_ok"] = int(sum(k in d for k in KEYS) >= 10)
    except Exception:
        d = {}
        for k in KEYS + ["side"]:
            mm = re.search(rf'"{k}"\s*:\s*"?([0-4LRBU])', text)
            if mm:
                d[k] = mm.group(1)
        out["parse_ok"] = int(len(d) >= 10)
    for k in KEYS:
        try:
            out[k] = int(min(4, max(0, int(d.get(k, 1)))))
        except Exception:
            pass
    s = str(d.get("side", "U")).upper()[:1]
    out["side"] = s if s in "RLBU" else "U"
    return out

# %% [markdown]
# ## Model

# %%
if MOCK:
    rng = np.random.default_rng(0)
    def generate(reports):
        return [json.dumps({"side": "R", **{k: int(rng.integers(0, 5)) for k in KEYS}}) for _ in reports]
    def n_tokens(texts):
        return [len(str(t).split()) for t in texts]
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    assert torch.cuda.device_count() >= 1, "No GPU — set Accelerator to GPU T4 x2"
    local = sorted(glob.glob(f"{INPUT_ROOT}/**/Qwen2.5-7B-Instruct*/config.json", recursive=True)) + \
            sorted(glob.glob(f"{INPUT_ROOT}/qwen2.5/**/7b-instruct*/**/config.json", recursive=True))
    MODEL_PATH = os.path.dirname(local[0]) if local else MODEL_ID
    print("Loading", MODEL_PATH, "(downloading needs Internet On)" if not local else "")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    def build(report):
        ids = tok(str(report), add_special_tokens=False).input_ids[:MAX_REPORT_TOKENS]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "REPORT:\n" + tok.decode(ids)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def generate(reports):
        enc = tok([build(r) for r in reports], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def n_tokens(texts):
        return [len(x) for x in tok([str(t) for t in texts], add_special_tokens=False).input_ids]

# %% [markdown]
# ## Grade: gold studies first, then everything else (checkpointed)

# %%
CKPT = f"{OUT}/report_grades_partial.csv"
done = pd.read_csv(CKPT) if os.path.exists(CKPT) else pd.DataFrame()
done_ids = set(done.get("StudyInstanceUID", []))
print("Resuming with", len(done_ids), "already graded" if done_ids else "")

def run(df, tag):
    df = df[~df.StudyInstanceUID.isin(done_ids)].copy()
    if not len(df):
        return
    df["_len"] = n_tokens(df.Report.tolist())
    df = df.sort_values("_len")
    t0, n = time.time(), 0
    for i in range(0, len(df), BATCH):
        b = df.iloc[i:i + BATCH]
        try:
            outs = generate(b.Report.tolist())
        except Exception as ex:  # e.g. OOM on the longest batch: retry one at a time
            print("batch failed:", type(ex).__name__, "- retrying singly")
            if not MOCK:
                torch.cuda.empty_cache()
            outs = [generate([r])[0] for r in b.Report.tolist()]
        rows = [{"StudyInstanceUID": s, **parse(o)} for s, o in zip(b.StudyInstanceUID, outs)]
        pd.DataFrame(rows).to_csv(CKPT, mode="a", header=not os.path.exists(CKPT), index=False)
        n += len(b)
        if (i // BATCH) % 20 == 0 or n == len(df):
            el = time.time() - t0
            print(f"  [{tag}] {n}/{len(df)}  {el/60:.1f} min elapsed, ~{el/n*(len(df)-n)/60:.0f} min left")

def gold_report(g):
    m = g.merge(train.loc[gold_mask, ["StudyInstanceUID"] + LABELS], on="StudyInstanceUID")
    res = {}
    for k, lab in K2L.items():
        y = m[lab].values
        if 0 < y.sum() < len(y):
            res[lab] = {"auc_grade": round(roc_auc_score(y, m[k]), 3),
                        "auc_grade4": round(roc_auc_score(y, m[k] == 4), 3),
                        "auc_grade2plus": round(roc_auc_score(y, m[k] >= 2), 3),
                        "pos_rate_by_grade": {int(gr): f"{int(y[m[k] == gr].sum())}/{int((m[k] == gr).sum())}"
                                              for gr in sorted(m[k].unique())}}
    macro = {s: round(float(np.mean([r[s] for r in res.values()])), 4) for s in ["auc_grade", "auc_grade4", "auc_grade2plus"]}
    return res, macro

run(train[gold_mask], "gold")
g = pd.read_csv(CKPT)
gold_res, gold_macro = gold_report(g)
print("\nGOLD (58 studies) macro AUC:", gold_macro)
for lab, r in gold_res.items():
    print(f"  {lab:17s} auc={r['auc_grade']:.3f}  grade4-only={r['auc_grade4']:.3f}  positives/n by grade {r['pos_rate_by_grade']}")
assert g.parse_ok.mean() > 0.5 or MOCK, "Most outputs failed to parse — paste the log to Claude before spending more GPU time."

run(train[~gold_mask], "rest")

# %% [markdown]
# ## Final outputs + SUMMARY — paste this block back to Claude

# %%
grades = pd.read_csv(CKPT).drop_duplicates("StudyInstanceUID", keep="last")
grades.to_csv(f"{OUT}/report_grades.csv", index=False)

side_vs_dicom = None
meta_path = find_file("study_meta.csv")
if meta_path:
    meta = pd.read_csv(meta_path)[["StudyInstanceUID", "laterality"]]
    side_vs_dicom = pd.crosstab(grades.merge(meta, on="StudyInstanceUID").laterality,
                                grades.merge(meta, on="StudyInstanceUID").side).to_dict()

gold_res, gold_macro = gold_report(grades)
summary = {
    "model": "mock" if MOCK else MODEL_PATH,
    "n_graded": int(len(grades)), "parse_ok_rate": round(float(grades.parse_ok.mean()), 4),
    "side_counts": grades.side.value_counts().to_dict(),
    "side_report_vs_dicom": side_vs_dicom,
    "grade_distribution_all": {K2L[k]: grades[k].value_counts().sort_index().to_dict() for k in KEYS},
    "gold_macro_auc": gold_macro,
    "gold_per_label": gold_res,
}
with open(f"{OUT}/grader_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
