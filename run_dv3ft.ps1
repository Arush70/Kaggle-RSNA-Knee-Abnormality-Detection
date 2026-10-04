# Fine-tunes the public knee-trained DINOv3 encoder on our 336 px volumes, one fold at a time.
# Safe to re-run: a fold that already has its train_log is skipped. Logs go to logs\.
Set-Location "C:\Machine Learning projects\rsna_knee"
.\.venv312\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path logs | Out-Null
foreach ($f in 1, 0, 2, 3, 4) {
    $out = "runs_dv3ft"
    if (Test-Path "$out\train_log_f$f.json") { Write-Host "skip fold $f (already done)"; continue }
    Write-Host "=== dv3 fine-tune fold $f starting $(Get-Date -Format 'HH:mm') ==="
    python finetune_dv3.py --fold $f --epochs 8 --freeze_blocks 6 --accum 8 `
        --init_head "runs_head\head_f$f.pt" --out_dir $out 2>&1 |
        Tee-Object -FilePath "logs\dv3ft_f$f.log"
}
Write-Host "ALL DV3 FOLDS DONE $(Get-Date -Format 'HH:mm')"
