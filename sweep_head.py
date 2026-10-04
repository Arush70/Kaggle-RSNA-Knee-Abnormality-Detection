"""Try several head settings on the cached features and print a table."""
import subprocess, sys, json, os, itertools
BASE = [sys.executable, "train_head.py", "--quiet", "1", "--out_dir", "runs_sweep"]
CONFIGS = [
    ("baseline",            []),
    ("epochs 80",           ["--epochs", "80"]),
    ("epochs 120",          ["--epochs", "120"]),
    ("d 512",               ["--d", "512"]),
    ("drop 0.15",           ["--drop", "0.15"]),
    ("drop 0.45",           ["--drop", "0.45"]),
    ("gold_weight 1",       ["--gold_weight", "1"]),
    ("gold_weight 8",       ["--gold_weight", "8"]),
    ("cls+mean only",       ["--blocks", "0,1"]),
    ("cls+mean+focal",      ["--blocks", "0,1,2"]),
    ("grid only",           ["--blocks", "3,4,5,6,7,8,9,10,11"]),
    ("lr 3e-3",             ["--lr", "3e-3"]),
    ("lr 3e-4",             ["--lr", "3e-4"]),
]
rows = []
for name, extra in CONFIGS:
    r = subprocess.run(BASE + extra, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{name:20s} FAILED\n{r.stderr[-600:]}"); continue
    log = json.load(open("runs_sweep/head_log.json"))["overall"]
    rows.append((name, log["auc_vs_public_labels"], log["auc_gold"], log["auc_vs_report_proxy"]))
    print(f"{name:20s} public {log['auc_vs_public_labels']:.4f} | gold {log['auc_gold']:.4f} | proxy {log['auc_vs_report_proxy']:.4f}", flush=True)
print("\nbest by gold:", max(rows, key=lambda r: r[2]))
print("best by public-label:", max(rows, key=lambda r: r[1]))
