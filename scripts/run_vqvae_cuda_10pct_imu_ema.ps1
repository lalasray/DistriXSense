param(
    [int]$Epochs = 10,
    [int]$Batch = 32,
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python scripts\train_shared_vqvae.py `
    --device cuda `
    --modalities BACK_IMU_acc:3,BACK_IMU_quat:4,RUA_IMU_acc:3,RUA_IMU_quat:4 `
    --codebook BACK_IMU_acc:128,BACK_IMU_quat:128,RUA_IMU_acc:128,RUA_IMU_quat:128 `
    --batch $Batch `
    --epochs $Epochs `
    --lr 1e-4 `
    --beta 0.1 `
    --quantizer ema `
    --num_workers 4 `
    --data_fraction 0.10 `
    --normalize_batch `
    --grad_clip 1.0 `
    --checkpoint_dir checkpoints\vqvae_10pct_imu_ema `
    $(if ($Amp) { "--amp" })
