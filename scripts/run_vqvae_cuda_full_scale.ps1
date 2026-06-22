param(
    [int]$Epochs = 10,
    [int]$Batch = 16,
    [double]$DataFraction = 0.10,
    [switch]$NoActivityAux,
    [switch]$Wandb,
    [string]$WandbProject = "DistriXSense-vqvae",
    [string]$WandbEntity = "",
    [string]$WandbRunName = "",
    [string]$WandbTags = "full-scale",
    [int]$WandbLogInterval = 10,
    [switch]$WandbWatch,
    [switch]$WandbLogArtifacts
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

if ($NoActivityAux) {
    $argsList += "--no-activity_contrastive_loss"
}

if ($Wandb) {
    $argsList += @(
        "--wandb",
        "--wandb_project", "$WandbProject",
        "--wandb_tags", "$WandbTags",
        "--wandb_log_interval", "$WandbLogInterval"
    )
    if ($WandbEntity) {
        $argsList += @("--wandb_entity", "$WandbEntity")
    }
    if ($WandbRunName) {
        $argsList += @("--wandb_run_name", "$WandbRunName")
    }
    if ($WandbWatch) {
        $argsList += "--wandb_watch"
    }
    if ($WandbLogArtifacts) {
        $argsList += "--wandb_log_artifacts"
    }
}

& $python @argsList
