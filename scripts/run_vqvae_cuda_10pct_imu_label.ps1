param(
    [int]$Epochs = 10,
    [int]$Batch = 32,
    [double]$LabelContrastiveWeight = 0.05,
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
    --grad_clip 1.0 `
    --activity_contrastive_loss `
    --label_stream label_HL_Activity `
    --label_vocab_size 4096 `
    --label_embedding_dim 32 `
    --label_contrastive_weight $LabelContrastiveWeight `
    --checkpoint_dir checkpoints\vqvae_10pct_imu_label `
    $(if ($Amp) { "--amp" })
