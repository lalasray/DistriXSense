param(
    [int]$Epochs = 10,
    [int]$Batch = 16,
    [double]$DataFraction = 0.10,
    [switch]$ActivityAux
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$argsList = @(
    "scripts\run_full_scale_vqvae.py",
    "--epochs", "$Epochs",
    "--batch", "$Batch",
    "--data_fraction", "$DataFraction",
    "--num_workers", "4",
    "--checkpoint_dir", "checkpoints\vqvae_full_scale",
    "--device", "cuda",
    "--quantizer", "ema",
    "--stft_loss_weight", "0.03",
    "--wavelet_loss_weight", "0.05",
    "--reed_transition_loss_weight", "0.20"
)

if ($ActivityAux) {
    $argsList += "--activity_contrastive_loss"
}

& $python @argsList
