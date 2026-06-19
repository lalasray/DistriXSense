param(
    [int]$Epochs = 2,
    [int]$Batch = 8,
    [switch]$Amp,
    [int]$NumWorkers = 0
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$argsList = @(
    "scripts\train_shared_vqvae.py",
    "--device", "cuda",
    "--modalities", "BACK_IMU_acc:3,BACK_IMU_quat:4,REED_DISHWASHER_S1:1",
    "--codebook", "BACK_IMU_acc:128,BACK_IMU_quat:128,REED_DISHWASHER_S1:64",
    "--batch", "$Batch",
    "--epochs", "$Epochs",
    "--num_workers", "$NumWorkers",
    "--checkpoint_dir", "checkpoints"
)

if ($Amp) {
    $argsList += "--amp"
}

& $python @argsList
