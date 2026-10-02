# RSNA 2026 Knee Abnormality Detection

Private working repo. **No competition data, reports, labels or model weights are committed** (see `.gitignore`).

## Pipeline
| Step | Notebook | Where it runs | Output |
|---|---|---|---|
| 01 | `notebooks/01_audit_and_folds.ipynb` | Kaggle, CPU | `folds.csv` (5 folds grouped by scanner) |
| 02 | `notebooks/02_report_grader.ipynb` | Kaggle, GPU T4 x2, internet on | `report_grades.csv` (0–4 severity per finding from reports, Qwen2.5-7B) |
| 03 | `notebooks/03_build_volumes.ipynb` | Kaggle, CPU | `vols/shard_*.npz` (5 slots × 20 slices × 256², ~19 GB) |
| 04 | `04_train.py` / `notebooks/04_train.ipynb` | Laptop (RTX 4070) or Kaggle GPU | `model_f{fold}.pt`, OOF predictions |
| 05 | `notebooks/05_submit.ipynb` | Kaggle, CPU, internet off | `submission.csv` |

Train on the laptop:
```
python 04_train.py --fold 1 --batch_size 2 --accum 4 --data_roots data --out_dir runs_v2
```

`src/` holds the source the notebooks are built from (`python src/build_nb.py src/04_train_main.py notebooks/04_train.ipynb`).

## Results log
| Date | Model | Fold-1 report AUC | Fold-1 gold AUC (n=12) | Public LB |
|---|---|---|---|---|
| 2026-10-02 | EffNet-B0, 12 ep, no mirroring | 0.765 | 0.858 | 0.802 |
