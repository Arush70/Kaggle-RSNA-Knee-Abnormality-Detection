# RSNA 2026 Knee Abnormality Detection

Private working repo. **No competition data, reports, labels, features or model weights are committed**
(see `.gitignore`) — competition rule 4.b. Keep this repo private until after the 22 Oct deadline.

## Two model families

**A. Our own model** — DINOv2-small fine-tuned on volumes we build ourselves (notebooks 01–05).
Lands around 0.80 on the leaderboard. Useful as a blend leg, not as a main model.

**B. Frozen public encoder + our head** (notebook 06 + `train_head.py`). The public
`knee-mri-fold-weights` dataset is a DINOv3 ViT-S/16 already trained on these knee MRIs.
We run it once over every study, cache a small feature vector per slot, and train only our own
head on top. Training then takes minutes instead of hours.

In reproducing their feature recipe we found their patch tokens are taken one position too early,
so the slot token is counted as a patch; the grid then isn't square and their "focal" feature
silently collapses to a copy of the mean. `src/06_dv3_features_main.py` uses the correct offset and
adds a real centre-focal and a 3×3 pooled grid.

## Pipeline
| Step | Notebook / script | Where it runs | Output |
|---|---|---|---|
| 01 | `notebooks/01_audit_and_folds.ipynb` | Kaggle CPU | `folds.csv` — 5 folds grouped by scanner fingerprint |
| 02 | `notebooks/02_report_grader.ipynb` | Kaggle GPU | `report_grades.csv` — 0–4 severity per finding (Qwen2.5-7B) |
| 03 | `notebooks/03_build_volumes.ipynb` | Kaggle CPU | `vols/shard_*.npz` — 5 slots × 20 slices × 256², ~19 GB |
| 03hr | `notebooks/03hr-part{1,2}-of-2.ipynb` | Kaggle CPU | the same at 336², in two halves (~32 GB) |
| 04 | `train_knee.py` | Laptop RTX 4070 | `model_f{fold}.pt`, `oof_f{fold}.csv`, `train_log_f{fold}.json` |
| 05 | `notebooks/05_submit.ipynb` | Kaggle, internet off | `submission.csv` from our own models |
| 06 | `notebooks/06-dv3-features.ipynb` | Kaggle CPU, ~100 min | `dv3_features_f0.npz` — frozen DINOv3 features, 183 MB |
| 07 | `train_head.py` | Laptop, minutes | `runs_head/head_f*.pt`, `oof_head.csv` |
| 08 | `blend_our_leg.py` | Kaggle GPU | our leg rank-blended into the public 0.943 notebook |

`src/` holds the sources the notebooks are built from:
`python src/build_nb.py src/04_train_main.py notebooks/04_train.ipynb`

## Running it

Train one fold on the laptop:
```
python train_knee.py --fold 1 --freeze_blocks 6 --lr_backbone 2e-5 --lr_head 5e-4 --epochs 12 \
  --n_slices_use 12 --batch_size 1 --accum 8 --workers 2 \
  --pub_labels report_labels_v2.csv --label_mix 0.5 --data_roots data --out_dir runs_mix_f1
```
All remaining folds, logged and resumable: `powershell -ExecutionPolicy Bypass -File run_folds.ps1`

Train the head on cached features (CPU, leaves the GPU free):
```
$env:CUDA_VISIBLE_DEVICES=""; python train_head.py
```

## Results log
Fold 1 unless stated. "report AUC" = against our report grades, "gold" = against the few
radiologist-labelled studies in that fold (n=12, so noisy).

| Date | Model | Report AUC | Gold AUC | Public LB |
|---|---|---|---|---|
| 10-02 | EfficientNet-B0, 12 ep, no mirroring | 0.765 | 0.858 | 0.802 |
| 10-02 | EfficientNet-B0, 20 ep, mirroring + TTA | 0.771 | 0.830 | — |
| 10-03 | DINOv2-small frozen (12 blocks) | 0.731 | 0.799 | — |
| 10-03 | DINOv2-small, 6 blocks trainable | 0.765 | 0.886 | 0.804 |
| 10-03 | ditto + 50% public LLM labels | **0.773** | **0.898** | — |

Public reference points: best single public model 0.924, public blend notebook 0.943,
leaderboard top is 0.958. Top-8 needs roughly 0.955.

## Things worth remembering
- Public LLM labels (`pilkwang/rsna-knee-llm-labels`) beat our own report grader
  (0.870 vs 0.806 gold AUC), and mixing them in helped every metric. Labels were the
  biggest single lever so far.
- Scanner fingerprint alone predicts the labels at about 0.65 AUC, so folds must be grouped
  by scanner or validation scores are inflated.
- A stalled training run has always been memory pressure: close other Python processes and
  keep `--workers 2`.
- Closing the PowerShell window kills training. `run_folds.ps1` writes logs so progress survives.
