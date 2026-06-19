param(
    [int]$Epochs = 2,
    [int]$Batch = 8,
    [double]$FrameDropProb = 0.15,
    [int]$MinKeepFrames = 8,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python scripts\train_shared_vqvae.py `
    --device cuda `
    --modalities BACK_IMU_acc:3,BACK_IMU_quat:4,REED_DISHWASHER_S3:1 `
    --codebook BACK_IMU_acc:128,BACK_IMU_quat:128,REED_DISHWASHER_S3:64 `
    --batch $Batch `
    --epochs $Epochs `
    --lr 1e-4 `
    --num_workers 4 `
    --data_fraction 0.10 `
    --grad_clip 1.0 `
    --use_temporal_interpolator `
    --frame_drop_prob $FrameDropProb `
    --min_keep_frames $MinKeepFrames `
    --checkpoint_dir checkpoints\vqvae_10pct_interp `
    $(if ($Amp) { "--amp" })
