# Trains the remaining folds with the recipe that won today's label test.
# Safe to re-run: a fold that already has its train_log is skipped.
Set-Location "C:\Machine Learning projects\rsna_knee"
.\.venv312\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path logs | Out-Null
foreach ($f in 0, 2, 3, 4) {
    $out = "runs_mix_f$f"
    if (Test-Path "$out\train_log_f$f.json") { Write-Host "skip fold $f (already done)"; continue }
    Write-Host "=== fold $f starting $(Get-Date -Format 'HH:mm') ==="
    python train_knee.py --fold $f --freeze_blocks 6 --lr_backbone 2e-5 --lr_head 5e-4 `
        --epochs 12 --n_slices_use 12 --batch_size 1 --accum 8 --workers 2 `
        --pub_labels report_labels_v2.csv --label_mix 0.5 `
        --data_roots data --out_dir $out 2>&1 | Tee-Object -FilePath "logs\$out.log"
}
Write-Host "ALL FOLDS DONE $(Get-Date -Format 'HH:mm')"
